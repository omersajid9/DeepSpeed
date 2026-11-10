# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Device-side sampling for rollout decode loops.

A rollout decode step needs one token per row and, for PPO/OPD, the log-probability of that
token. HuggingFace's sampling loop does both on the host (logits warpers and ``multinomial``
in Python), which stalls the accelerator once per token; this module keeps the whole draw in
tensor ops so a decode loop can sample without leaving the device.

The draw is the Gumbel-max trick: ``argmax(logits / T + Gumbel)`` is an exact categorical
sampler (Maddison et al. 2014, "A* Sampling"). Temperature, top-k and top-p are therefore
three masks over one vectorised ``argmax``, with no per-row Python loop and no host
round-trip. The top-p rule keeps the token that crosses the threshold (vLLM's convention)
rather than the last token whose inclusive cumulative probability stays inside ``p``.

Log-probabilities are taken over the sampling distribution (temperature-scaled, filtered,
renormalized) -- the distribution the token was actually drawn from, which is what an
importance ratio needs.
"""

import torch


def sample(logits, temperature=1.0, top_k=-1, top_p=1.0, generator=None):
    """Draw one token per row from ``logits`` and report its log-probability.

    Args:
        logits: ``[batch, vocab]`` unnormalized scores. ``-inf`` entries can never be drawn.
        temperature: softmax temperature; ``<= 0`` draws the argmax (greedy), ignoring filters.
        top_k: keep the ``k`` highest-scoring tokens (``<= 0`` disables the filter).
        top_p: keep the smallest set of highest-scoring tokens whose probability reaches
            ``top_p`` (``>= 1`` disables the filter).
        generator: optional ``torch.Generator`` for the noise; defaults to the global one.

    Returns:
        ``(tokens, logprobs)``: ``[batch]`` int64 token ids and ``[batch]`` float32
        log-probabilities of those tokens under the sampling distribution.
    """
    logits = logits.float()

    if temperature <= 0.0:
        tokens = logits.argmax(dim=-1)
        return tokens, torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    logits = logits / temperature
    if top_k > 0:
        logits = _top_k(logits, top_k)
    if top_p < 1.0:
        logits = _top_p(logits, top_p)

    noise = torch.rand(logits.shape, generator=generator, device=logits.device, dtype=logits.dtype)
    gumbel = -torch.log(-torch.log(noise.clamp_(1.0e-7, 1.0 - 1.0e-7)))
    tokens = (logits + gumbel).argmax(dim=-1)
    logprobs = torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens, logprobs


def _top_k(logits, k):
    """Mask every score below the ``k``-th largest with ``-inf``; ties at the cut survive."""
    kth = logits.topk(min(k, logits.shape[-1]), dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def _top_p(logits, p):
    """Mask the tokens the nucleus filter drops, keeping the one that crosses ``p``.

    A token is dropped once the probability mass of the strictly better tokens has reached
    ``p``, so a 0.5/0.5 pair survives ``top_p=0.6`` instead of collapsing to one token.
    """
    sorted_logits, order = logits.sort(dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    dropped = (sorted_probs.cumsum(dim=-1) - sorted_probs) > p
    dropped[..., 0] = False
    kept = sorted_logits.masked_fill(dropped, float("-inf"))
    return torch.empty_like(logits).scatter_(-1, order, kept)

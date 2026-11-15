# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for the device-side rollout sampler."""

import pytest
import torch

from deepspeed.runtime.rollout.sampling import sample


def test_sample_follows_the_softmax_law():
    logits = torch.log(torch.tensor([[0.3, 0.3, 0.2, 0.2]]))
    rows = 100_000
    tokens, _ = sample(logits.repeat(rows, 1), generator=torch.Generator().manual_seed(7))

    frequency = torch.bincount(tokens, minlength=4).float() / rows
    assert torch.allclose(frequency, torch.softmax(logits, -1)[0], atol=0.01)


def test_sample_reports_the_logprob_of_the_drawn_token():
    logits = torch.log(torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.1, 0.2, 0.3, 0.4]]))
    tokens, logprobs = sample(logits, generator=torch.Generator().manual_seed(1))

    assert tokens.shape == logprobs.shape == (2, )
    assert tokens.dtype == torch.int64
    assert logprobs.dtype == torch.float32
    assert torch.allclose(logprobs, torch.distributions.Categorical(logits=logits).log_prob(tokens), atol=1e-6)


def test_sample_greedy_at_zero_temperature_ignores_filters():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.5]])
    tokens, logprobs = sample(logits, temperature=0.0, top_k=1, top_p=0.1)

    assert tokens.tolist() == [1]
    assert logprobs[0].item() == pytest.approx(torch.log_softmax(logits, -1)[0, 1].item())


def test_sample_top_k_limits_the_support_and_renormalizes_logprobs():
    logits = torch.log(torch.tensor([[0.3, 0.3, 0.2, 0.2]]))
    rows = 20_000
    tokens, logprobs = sample(logits.repeat(rows, 1), top_k=2, generator=torch.Generator().manual_seed(3))

    assert set(tokens.tolist()) == {0, 1}
    assert logprobs[0].item() == pytest.approx(float(torch.log(torch.tensor(0.5))), abs=1e-6)


def test_sample_top_k_one_reproduces_greedy():
    logits = torch.randn(2, 32, generator=torch.Generator().manual_seed(2))
    tokens, _ = sample(logits, temperature=1.0, top_k=1, generator=torch.Generator().manual_seed(3))

    assert torch.equal(tokens, logits.argmax(dim=-1))


def test_sample_top_p_keeps_the_token_that_crosses_the_threshold():
    """0.3/0.3/0.2/0.2 at top_p=0.5 keeps both 0.3 tokens instead of collapsing to the first."""
    logits = torch.log(torch.tensor([[0.3, 0.3, 0.2, 0.2]]))
    rows = 20_000
    tokens, logprobs = sample(logits.repeat(rows, 1), top_p=0.5, generator=torch.Generator().manual_seed(5))

    assert set(tokens.tolist()) == {0, 1}
    assert logprobs[0].item() == pytest.approx(float(torch.log(torch.tensor(0.5))), abs=1e-6)


def test_sample_top_p_zero_keeps_only_the_argmax():
    logits = torch.log(torch.tensor([[0.4, 0.3, 0.2, 0.1]]))
    tokens, _ = sample(logits, top_p=0.0, generator=torch.Generator().manual_seed(9))

    assert int(tokens[0]) == int(logits[0].argmax())


def test_sample_never_draws_a_masked_token():
    logits = torch.tensor([[0.0, float("-inf"), 1.0]])
    tokens, logprobs = sample(logits.repeat(1_000, 1), generator=torch.Generator().manual_seed(13))

    assert 1 not in tokens.tolist()
    assert torch.isfinite(logprobs).all()


def test_sample_is_reproducible_with_a_generator():
    logits = torch.randn(3, 16, generator=torch.Generator().manual_seed(0))

    first = sample(logits, temperature=0.8, top_p=0.9, generator=torch.Generator().manual_seed(21))
    again = sample(logits, temperature=0.8, top_p=0.9, generator=torch.Generator().manual_seed(21))

    assert torch.equal(first[0], again[0])
    assert torch.equal(first[1], again[1])

# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Benchmark: fused (segment-KI) vs native decode attention over a paged KV cache.

Runs the same greedy continuous-batch rollout three ways and checks that all
three produce the same response tokens:

  paged native        the paged cache's default decode forward (eager q-side
                      ops + BlockedFlashAttn over the block tables)
  paged fused         use_segki_paged_decode: the segment-KI decode_attn kernel
                      with block-table addressing (one kernel per attention
                      layer: q|gate projection chunk, per-head norms, partial
                      RoPE, KV store, split-KV softmax, sigmoid gate)
  paged flashinfer    use_flashinfer_decode: FlashInfer's paged decode backend
                      behind the same eager q-side (only when flashinfer is
                      installed)

The numbers that matter: wall ms per variant and the peak memory, plus the
token-stream digest — a fused run that changes tokens is a bug, not a
speedup.

Launch (from repo root, one GPU):

    # smoke: small random hybrid model, no weights to download
    python benchmarks/rollout/bench_paged_segki_decode.py --prompt-len 256 --max-new-tokens 16

    # real model, the handoff's cell
    python benchmarks/rollout/bench_paged_segki_decode.py \\
        --model /path/to/Qwen3.5-4B --prompt-len 3072 --max-new-tokens 1024 --batch-size 4

Arguments:
    --model           HuggingFace id or path; omit to build a small random Qwen3.5
    --prompt-len      tokens per prompt, padded to a common width (default: 256)
    --max-new-tokens  response tokens per request (default: 16)
    --batch-size      requests in the continuous batch (default: 2)
    --max-batch-size  decode slots (default: same as --batch-size)
    --dtype           bf16 (default) or fp16
    --warmup          discarded generate calls before timing (default: 1)
    --iterations      timed generate calls (default: 3)
    --variants        comma separated subset of native,fused,flashinfer
    --json            optional path to dump the results as JSON
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import (
    HybridEngineRollout,
    HybridEngineRolloutConfig,
)


def _random_hybrid_model(dtype):
    """A tiny random-weight Qwen3.5 hybrid model, so the benchmark needs no weights."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        partial_rotary_factor=0.25,
        layer_types=["linear_attention"] * 6 + ["full_attention"] * 2,
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        vocab_size=1024,
        max_position_embeddings=4096,
    )
    model = Qwen3_5ForCausalLM(config).to(dtype=dtype)
    model._supports_cache_class = True
    return model.eval()


def _load_model(model_name, dtype):
    if model_name is None:
        return _random_hybrid_model(dtype)
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    if getattr(config, "model_type", "") == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa")
        model.model.visual = None  # text-only rollout: the vision tower is never called
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa")
    model._supports_cache_class = True
    return model.eval()


def _make_request(prompt_len, batch_size, vocab_size, device):
    generator = torch.Generator().manual_seed(0)
    prompt_ids = torch.randint(1, vocab_size, (batch_size, prompt_len), generator=generator, device=device)
    prompt_mask = torch.ones_like(prompt_ids)
    return RolloutRequest(prompt_ids, prompt_mask)


def _response_ids(batch):
    rows = []
    for row in range(batch.input_ids.shape[0]):
        start = int(batch.response_start_idx[row].item())
        mask = batch.attention_mask[row, start:].bool()
        rows.append(batch.input_ids[row, start:][mask].tolist())
    return rows


def _digest(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()[:12]


def _run_variant(name, config, request, sampling, warmup, iterations):
    model = config.pop("_model")
    rollout = HybridEngineRollout(SimpleNamespace(module=model),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                  cfg=HybridEngineRolloutConfig(**config))
    with torch.no_grad():
        for _ in range(warmup):
            rollout.generate(request, sampling)
        get_accelerator().synchronize()
        get_accelerator().reset_peak_memory_stats()
        walls, rows = [], None
        for _ in range(iterations):
            start = time.perf_counter()
            batch = rollout.generate(request, sampling)
            get_accelerator().synchronize()
            walls.append((time.perf_counter() - start) * 1000.0)
            rows = _response_ids(batch)
    peak_mb = get_accelerator().max_memory_allocated() / (1024**2)
    stats = rollout.get_last_continuous_stats() or {}
    return {
        "variant": name,
        "wall_ms": [round(value, 1) for value in walls],
        "best_wall_ms": round(min(walls), 1),
        "peak_memory_mb": round(peak_mb, 1),
        "response_digest": _digest(rows),
        "response_ids": rows,
        "graph_captures": stats.get("graph_captures"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batch-size", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--variants", default="native,fused")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():  #ignore-cuda
        raise SystemExit("paged rollout benchmarks need a CUDA device")

    max_batch_size = args.max_batch_size or args.batch_size
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(get_accelerator().current_device_name())

    model = _load_model(args.model, dtype).to(device)
    vocab_size = int(getattr(model.config, "vocab_size", getattr(model.config, "text_config", None).vocab_size))
    request = _make_request(args.prompt_len, args.batch_size, vocab_size, device)
    sampling = SamplingConfig(max_new_tokens=args.max_new_tokens,
                              temperature=0.0,
                              top_p=1.0,
                              n_samples_per_prompt=1,
                              continuous_batch_size=max_batch_size)

    variants = {
        "native":
        dict(use_graph_capture=True, use_paged_cache=True, enable_paged_prefill=True),
        "fused":
        dict(use_graph_capture=True,
             use_paged_cache=True,
             enable_paged_prefill=True,
             use_segki=True,
             use_segki_paged_decode=True),
        "flashinfer":
        dict(use_graph_capture=True, use_paged_cache=True, enable_paged_prefill=True, use_flashinfer_decode=True),
    }
    results = []
    for name in args.variants.split(","):
        name = name.strip()
        if name not in variants:
            raise SystemExit(f"unknown variant {name!r}; choose from {sorted(variants)}")
        results.append(
            _run_variant(name, dict(variants[name], _model=model), request, sampling, args.warmup, args.iterations))

    baseline = results[0]
    print(f"model: {args.model or 'random-qwen3.5'}  prompt {args.prompt_len}  new {args.max_new_tokens}  "
          f"batch {args.batch_size}/{max_batch_size}  dtype {args.dtype}")
    ok = True
    for result in results:
        same = result["response_digest"] == baseline["response_digest"]
        delta = result["best_wall_ms"] / baseline["best_wall_ms"] - 1.0
        print(f"  {result['variant']:<11} {result['best_wall_ms']:>9.1f} ms  "
              f"({delta:+.1%} vs {baseline['variant']})  peak {result['peak_memory_mb']:>7.1f} MB  "
              f"captures {result['graph_captures']}  digest {result['response_digest']}  "
              f"{'match' if same else 'TOKEN MISMATCH'}")
        ok &= same or result is baseline
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"args": vars(args), "results": results}, indent=2) + "\n")
        print(f"wrote {args.json}")
    if not ok:
        raise SystemExit("response tokens differ between variants")


if __name__ == "__main__":
    main()

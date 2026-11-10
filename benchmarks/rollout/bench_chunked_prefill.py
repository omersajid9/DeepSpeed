# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Benchmark: naive vs chunked prefill for continuous-batching rollout generation.

Runs the same greedy rollout twice - once with the prompt prefilled in a single
forward (naive) and once with chunked prefill - and prints the token stream check
plus the profile breakdown that the two differ in. The captured-graph continuous
path is used when the accelerator supports graph capture; otherwise the script
falls back to the eager continuous path and says so, since the chunking machinery
is shared between them.

The numbers that matter:

  prefill forward ms / prefill forwards   the latency a *step* is blocked by while
                                          a prompt fills. One large value for naive,
                                          many small ones for chunked.
  decode forwards / decode steps          the decoding rows' work, which chunked
                                          prefill interleaves with the staging.
  tokens/s, end-to-end ms                 the throughput price of that interleaving:
                                          chunked prefill re-reads the weights once
                                          per chunk, so it can be slower overall.

Launch (from repo root):

    # offline sanity run: a small random-weight Llama, CPU, chunk 4 vs naive
    python benchmarks/rollout/bench_chunked_prefill.py --prompt-len 64 --chunk-sizes 16

    # a real model on one GPU
    python benchmarks/rollout/bench_chunked_prefill.py --model facebook/opt-1.3b \\
        --prompt-len 4096 --max-new-tokens 64 --requests 8 --max-batch-size 8 \\
        --dtype bf16 --chunk-sizes 512,1024,2048

Arguments:
    --model           HuggingFace id or path; omit to build a small random model
    --prompt-len      tokens per prompt, padded to a common width (default: 512)
    --max-new-tokens  response tokens per request (default: 32)
    --requests        prompts submitted to the rollout (default: 8)
    --max-batch-size  decode slots, i.e. concurrent requests (default: 8)
    --chunk-sizes     comma separated chunk sizes to test (default: 128,512)
    --mixed           make half the prompts effectively short via left padding
    --prefill-first   give the first requests the long prompts (order matters)
    --dtype           fp32 | fp16 | bf16 (default: bf16 on cuda, fp32 otherwise)
    --graph           auto | on | off capture of the decode graph (default: auto)
    --trials          measured repetitions per configuration, best is reported (default: 3)
    --warmup          unmeasured repetitions per configuration (default: 1)
    --layers/--hidden-size/--heads   shape of the random model

    python benchmarks/rollout/bench_chunked_prefill.py ...            (installed deepspeed)
    PYTHONPATH=$PWD python benchmarks/rollout/bench_chunked_prefill.py ...  (repo checkout)
"""

import argparse
import time
from types import SimpleNamespace

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args():
    parser = argparse.ArgumentParser(description="naive vs chunked prefill rollout benchmark")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--chunk-sizes", type=str, default="128,512")
    parser.add_argument("--mixed", action="store_true", help="make half the prompts effectively short")
    parser.add_argument("--prefill-first", action="store_true", help="give the first requests the long prompts")
    parser.add_argument("--dtype", type=str, default=None, choices=list(DTYPES))
    parser.add_argument("--graph", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def build_model(args, device, dtype):
    """Load a checkpoint, or build a small random model so the script runs offline."""
    if args.model is None:
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(vocab_size=args.vocab_size,
                             hidden_size=args.hidden_size,
                             intermediate_size=4 * args.hidden_size,
                             num_hidden_layers=args.layers,
                             num_attention_heads=args.heads,
                             num_key_value_heads=args.heads,
                             max_position_embeddings=max(2048, 4 * args.prompt_len),
                             pad_token_id=0,
                             eos_token_id=1)
        model = LlamaForCausalLM(config)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    return model.to(device=device, dtype=dtype).eval()


def build_workload(args, device, pad_token_id):
    """Left-padded prompts of a common width, the layout the continuous path requires."""
    lengths = [args.prompt_len] * args.requests
    if args.mixed:
        for index in range(1, args.requests, 2):
            lengths[index] = max(1, args.prompt_len // 8)
    if args.prefill_first:
        lengths.sort(reverse=True)

    prompt_ids = torch.full((args.requests, args.prompt_len), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(prompt_ids)
    for index, length in enumerate(lengths):
        # Real tokens sit at the right edge so causal generation appends after them.
        prompt_ids[index, args.prompt_len - length:] = torch.randint(2,
                                                                     args.vocab_size, (length, ),
                                                                     dtype=torch.long,
                                                                     device=device)
        attention[index, args.prompt_len - length:] = 1
    return RolloutRequest(prompt_ids, attention), lengths


def run_once(args, model, workload, chunk_size, graph_capture, device):
    cfg = HybridEngineRolloutConfig(use_graph_capture=graph_capture,
                                    enable_profiling=True,
                                    enable_chunked_prefill=chunk_size is not None,
                                    prefill_chunk_size=chunk_size or 512)
    rollout = HybridEngineRollout(SimpleNamespace(module=model),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=None),
                                  cfg=cfg)
    sampling = SamplingConfig(max_new_tokens=args.max_new_tokens,
                              temperature=0,
                              continuous_batch_size=args.max_batch_size)

    accelerator = get_accelerator()
    accelerator.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output = rollout.generate(workload, sampling)
    accelerator.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if output.input_ids.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    return {
        "ids": output.input_ids.cpu(),
        "elapsed_ms": elapsed_ms,
        "profile": rollout.get_last_profile() or {},
        "stats": rollout.get_last_continuous_stats() or {},
    }


def measure(args, model, workload, chunk_size, graph_capture, device):
    """Best of `--trials` runs after `--warmup` unmeasured ones.

    A fresh rollout is built per run (the graph cache and the eager cache both key on the
    batch shape and on parameter versions), so timing reflects steady-state generation
    rather than first-call initialization.
    """
    for _ in range(max(0, args.warmup)):
        run_once(args, model, workload, chunk_size=chunk_size, graph_capture=graph_capture, device=device)
    best = None
    for _ in range(max(1, args.trials)):
        run = run_once(args, model, workload, chunk_size=chunk_size, graph_capture=graph_capture, device=device)
        if best is None or run["elapsed_ms"] < best["elapsed_ms"]:
            best = run
    return best


def summarize(name, run, args):
    profile = run["profile"]
    stats = run["stats"]
    prefill_forwards = profile.get("num_prefill_forwards", 0)
    decode_forwards = profile.get("num_decode_forwards", 0)
    tokens = stats.get("num_generated_tokens") or args.requests * args.max_new_tokens
    return {
        "config": name,
        "chunks": stats.get("chunked_prefill_chunks", 0),
        "staging MB": stats.get("chunked_prefill_staging_bytes", 0) / 2**20,
        "prefill fwd": prefill_forwards,
        "prefill ms": profile.get("prefill_forward_ms", 0.0),
        "ms/fwd": profile.get("prefill_forward_ms", 0.0) / prefill_forwards if prefill_forwards else 0.0,
        "decode fwd": decode_forwards,
        "decode ms": profile.get("decode_forward_ms", 0.0),
        "sched ms": profile.get("scheduler_overhead_ms", 0.0),
        "cache ms": profile.get("cache_management_overhead_ms", 0.0),
        "steps": stats.get("decode_steps", 0),
        "total ms": run["elapsed_ms"],
        "tokens/s": tokens / (run["elapsed_ms"] / 1000.0) if run["elapsed_ms"] > 0 else 0.0,
    }


def print_table(rows):
    columns = list(rows[0])
    numeric = [
        "chunks", "staging MB", "prefill fwd", "prefill ms", "ms/fwd", "decode fwd", "decode ms", "sched ms",
        "cache ms", "steps", "total ms", "tokens/s"
    ]
    print("  ".join(f"{c:>12}" if c in numeric else f"{c:<16}" for c in columns))
    print("-" * (len(columns) * 15))
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if column in numeric:
                cells.append(f"{value:>12.2f}" if isinstance(value, float) else f"{value:>12}")
            else:
                cells.append(f"{value:<16}")
        print("  ".join(cells))


def main():
    args = parse_args()
    accelerator = get_accelerator()
    device = torch.device(accelerator.current_device_name()) if accelerator.is_available() else torch.device("cpu")
    dtype = DTYPES[args.dtype] if args.dtype else (torch.bfloat16 if device.type == "cuda" else torch.float32)

    graph_capture = device.type == "cuda" and accelerator.create_graph() is not None
    if args.graph == "on":
        graph_capture = True
    elif args.graph == "off":
        graph_capture = False
    if args.graph != "off" and not graph_capture:
        print("NOTE: graph capture is unavailable on this accelerator; using the eager continuous path. "
              "The chunking machinery is shared, so the comparison still holds.\n")

    model = build_model(args, device, dtype)
    workload, lengths = build_workload(args, device, pad_token_id=0)
    print(f"device={device} dtype={dtype} graph_capture={graph_capture} "
          f"requests={args.requests} slots={args.max_batch_size} padded_prompt={args.prompt_len} "
          f"effective_prompts={sorted(set(lengths), reverse=True)} new_tokens={args.max_new_tokens}\n")

    results = {}
    naive = measure(args, model, workload, chunk_size=None, graph_capture=graph_capture, device=device)
    results["naive"] = naive
    for size in [int(s) for s in args.chunk_sizes.split(",") if s.strip()]:
        results[f"chunked@{size}"] = measure(args,
                                             model,
                                             workload,
                                             chunk_size=size,
                                             graph_capture=graph_capture,
                                             device=device)

    print("correctness (greedy token streams must match the naive prefill):")
    for name, run in results.items():
        match = "MATCH" if torch.equal(run["ids"], naive["ids"]) else "MISMATCH"
        print(f"  {name:<16} {match}")
    print()

    print("generation profile:")
    print_table([summarize(name, run, args) for name, run in results.items()])


if __name__ == "__main__":
    main()

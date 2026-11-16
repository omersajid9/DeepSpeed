# Experimental packed GDN and residual RMSNorm kernels

This temporary branch ports the standalone kernel experiments from
`kernel_lab/b300` into the DeepSpeed repository. It does not change the installed
DeepSpeed operators or enable a new default dispatch path. The branch starts at
`bee5f1b22ebc67e5e7d2254e7041d3a40d0d379e` on `perf/paged_fused_kernels`.

These kernels are for frozen-weight, single-GPU inference experiments. They are
**not token-identical to the existing engine**, and model accuracy equivalence
has not been established. AI assistance was used in implementation and testing;
this is an experimental branch for human review, not a production feature PR.

## Implementation

- `kernels.cu`: vectorized CUDA residual add/RMSNorm, and a packed CUDA GDN
  recurrence with grouped Q/K heads, gate computation and in-place state updates.
  The norm preserves the unrounded FP32 residual sum during normalization while
  storing its BF16 rounding back to the residual stream.
- `packed_triton.py`: the faster measured GDN alternative. BF16 and FP32 state
  retain the native `[batch, value_head, K, V]` layout. This removes separate
  gate, Q/K repetition and value-repacking launches. The selected `triton_match`
  variant uses the FLA division/square-root normalization expression.
- `experiment.py`: independent FP64-oracle checks and CUDA-graph microbenchmarks.
- `stress_and_streaming.py`: 1,024-step recurrence checks with extreme gates,
  followed by measurements rotating through 24 distinct layer states.
- `run_candidate.py`: explicit process-local injection into a benchmark script.
  Prefill, training, tensor parallelism and unsupported shapes retain the
  original path. Dispatch counters reject runs where a requested kernel did not
  execute. Derived gate constants assume frozen weights; do not use this runner
  across weight refreshes.

CUDA norm supports BF16 widths 2,560 and 4,096. Packed GDN supports BF16
activations, BF16 or FP32 state, K=V=128, and divisible grouped head counts.
Inputs to the experimental Triton entry point must follow the tested layout.
It is not a general-purpose public operator API. See `NOTICES.md` for source
attribution.

## Measured results before the port

The following are historical B300 measurements of the original experiment.
Formatting and output-path changes during this port do not constitute new
end-to-end performance or model-accuracy measurements.

For 4B, 3,072-token LongBench prompts followed by 1,024 greedy tokens, ignoring
EOS, one warmup:

| Batch | Corrected E control median | Final candidate median | Wall reduction | Saved vLLM reference |
|---|---:|---:|---:|---:|
| 2 | 3,394.87 ms | 3,198.17 ms | 5.79% | 2,753.38 ms |
| 32 | 7,732.24 ms | 7,118.52 ms | 7.94% | 4,684.46 ms |

Candidate medians use three iterations; controls use six iterations collected
before and after the candidate. vLLM is the saved single-iteration reference.
No new 9B full-model performance matrix was measured for these kernels.

At batch 32, the 24-layer BF16 state working set is 768 MiB. Existing GDN takes
33.34 microseconds/layer, packed CUDA takes 23.27, and selected Triton takes
16.71. CUDA residual RMSNorm takes 1.87 versus 3.23 microseconds at width 2,560,
and 2.13 versus 3.02 at width 4,096. These do not fix prefill or host overhead.

Exact-answer accuracy on 128 frozen GSM8K questions:

| Test | Previous corrected E | Final candidate | vLLM reference |
|---|---:|---:|---:|
| 4B greedy | 120/128 | 122/128 | 120/128 |
| 9B greedy | 124/128 | 123/128 | 124/128 |
| 4B sampling, four seeds | 483/512 | 477/512 | 480/512 |
| 9B sampling, four seeds | 491/512 | 488/512 | 495/512 |

Sampling uses temperature 0.7, top-p 0.9, top-k 50, seeds 0/1/2/3 and a
1,024-token cap. All candidate correct answers finished on EOS. Repetition
screening flagged 21/512 candidate 9B samples versus 14/512 previous E and
12/512 vLLM; this is a heuristic, but it prevents a claim of unchanged behavior.
There was no predefined statistical non-inferiority margin. Keep the observed
losses alongside the performance results.

`evidence/` contains compact historical measurements. Its portability source
hash identifies the original, pre-formatting CUDA source. The original kernels
compiled for SM80 (A100), SM86, SM90 (H200) and SM103 (B300). Runtime correctness
and performance were measured **only on B300**. The Triton tile choices
(64 columns/four warps for BF16, 32/four for FP32) require tuning and validation
on older GPUs.

## Reproduce in an existing isolated environment

Use a CUDA-capable PyTorch, Triton, FLA and DeepSpeed environment compatible with
the perf branch. The historical B300 environment used PyTorch 2.15.0+cu132,
FLA 0.6.0, CUDA 13.2, driver 595.71.05 and one pinned B300 SXM6 AC per run.
No script installs packages, checks out another commit or changes an environment.

The historical results used the corrected E baseline. Its separate
`evidence/baseline_compat.patch.gz` records those prerequisites relative to this
branch's parent: CUDA compatibility and rollout accuracy corrections. It is
provided for provenance and is **not automatically applied**. Using an
uncorrected baseline is not a reproduction of those model measurements.

On the original B300 host, use the already validated environment and baseline:

```bash
source /fsx/mosajid/pfuse/compat/corrected.env
export CUDA_VISIBLE_DEVICES=0
export KERNEL_LAB_OUTPUT=/fsx/mosajid/tmp/gdn-kernel-port
export TMPDIR=/fsx/mosajid/tmp
export TORCH_CUDA_ARCH_LIST=10.3

python /fsx/mosajid/deepspeed/DeepSpeed-kernel-port/benchmarks/inference/experimental_gdn/experiment.py
STATE_DTYPE=bf16 python /fsx/mosajid/deepspeed/DeepSpeed-kernel-port/benchmarks/inference/experimental_gdn/experiment.py
STATE_DTYPE=bf16 MATCH_FLA=1 python /fsx/mosajid/deepspeed/DeepSpeed-kernel-port/benchmarks/inference/experimental_gdn/stress_and_streaming.py
```

On another host, set `KERNEL_LAB_OUTPUT`, compiler/cache paths and the script
paths for that host. Use an appropriate CUDA toolkit and set
`TORCH_CUDA_ARCH_LIST=8.0` on A100 or `9.0` on H200. An unset architecture list
uses PyTorch's normal device detection. These variables affect the standalone
extension build, not production dispatch.

To inject the selected kernels into an existing benchmark or accuracy script:

```bash
KERNEL_LAB_VARIANT=triton_match python path/to/experimental_gdn/run_candidate.py \
    path/to/benchmark.py --output path/to/result.json [original benchmark arguments]
```

Retain the original arguments, model revision and compatible baseline. Start
with greedy accuracy, then sampling before interpreting speed. Use `norm` or
`triton_gdn_match` to ablate individual components. `gdn_cuda4` selects the
slower native CUDA GDN alternative. `baseline` installs no overrides.

Runtime products, generated forward preview and extension builds go under
`KERNEL_LAB_OUTPUT` (default: the ignored `results/` directory next to the
scripts). Source and historical evidence are not overwritten.

## Port verification

The three operator commands above passed on one B300 after formatting the
ported files: BF16 and FP32 state, residual norms, strided projection inputs,
in-place updates, extreme gates and 1,024 recurrent steps.
`evidence/port_validation.json` records source hashes and the validation output
directory. CUDA computation is unchanged after formatting and a local variable
rename; the normalized Triton syntax tree also matches the original experiment.
No new full-model accuracy run or A100/H200 runtime test was performed for the
port. The corrected baseline patch hash and original checkout remained unchanged.

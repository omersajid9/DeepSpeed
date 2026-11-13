# Paged Attention for Hybrid (GDN) Rollout — Handoff Status

Branch: `paged-hybrid-gdn` (based on `opti-merge` @ `d8c9f455e`, which is 3 commits ahead of `be25c8d92`).
Commits on this branch vs `be25c8d92`:

- `4f4cf326c` feat(rollout): run the segki decode kernels under continuous batching
- `c06317ec9` feat(rollout): paged attention for hybrid GDN models (4 files, +788/−57)
- `d8c9f455e` fix(rollout): expose paged layer init flag so CB graph capture succeeds

Committer: omersajid9 <omersajid.777@gmail.com> (all commits `--signoff`).

## What this work does

Adds paged (blocked) attention support to the DeepSpeed continuous-batching
(CB) rollout path for hybrid GDN models (Qwen3.5-4B: 32 layers = 24
linear_attention + 8 full_attention; GDN 32 v / 16 k heads, 128/128 dims, conv
K=4; full attention 16 q / 4 kv heads, head_dim=256, partial_rotary_factor
0.25). Previously only the dense (static) cache worked under CB; paged was
rejected by the hybrid validator.

Design guidance: AVMP (paged attention for hybrid models) and Marconi
(blocked KV for LLM serving). HF `transformers` GDN cache protocol is the
reference implementation the paged hybrid cache mirrors.

## Files

- `deepspeed/runtime/rollout/paged_cache.py` — the core of the work:
  - `PagedAttentionCache`: blocked KV pool, device-resident block table per
    row, `compact()`/`reset()`, base KV-slot remap (`num_kv_layers`,
    `_layer_to_slot`, `_kv_slot`).
  - `_PagedLayer`: per-layer view; `is_initialized` property returns `True`
    (pools are allocated with the cache — no lazy-allocation step for the CB
    capture probe). This flag is required by
    `hybrid_engine_rollout._capture_continuous_graph`, which probes
    `all(layer.is_initialized for layer in cache.layers)`; without it the paged
    bucket capture raised and every decode step fell back to eager.
  - `_apply_partial_rotary`: bit-exact vs transformers
    `apply_rotary_pos_emb` (verified by unit test).
  - `_paged_qwen3_5_forward`: HF-style attention forward (q+gate chunk, q/k
    RMSNorm, partial RoPE) run over blocked flash attention with per-step
    block-table atoms.
  - `HybridPagedAttentionCache`: HF GDN state pools + protocol
    (`update_conv_state` left-zero-pads a fresh prefill row with seq_len <
    kernel, mirroring HF `cache_utils.update_conv_state`; 2-D row copies are
    `pool[row:row+1].copy_(...)`; width-0 decode branch guarded;
    `get_recurrent_state` returns None when recording but no prefill row is
    selected). Fail-closed validation for unsupported combinations.
- `deepspeed/runtime/rollout/hybrid_engine_rollout.py`:
  - `_validate_hybrid_continuous_options()` moved before the generic paged-trim
    check in `_validate_continuous_inputs` so hybrid-specific errors win.
  - `_create_continuous_paged_cache`, `_paged_prefill` (recording window,
    `begin_prefill_row`, mask dict).
  - Bucket/eager paged mask dicts; `update_graph_atoms` /
    `_init_graph_atom_structure` call sites.
  - Graph loop uses `write_positions = {}` (dict, ~line 1083) vs the eager
    path's 1-D tensor (~line 690) — relevant to the open bug below.
- `tests/unit/runtime/rollout/test_paged_cache.py`: 6 new tests (KV
  sizing / layer surface / GDN-index rejection; prefill protocol incl.
  short-prompt left-pad; decode in-place pool-row semantics; compact moves
  pool rows + KV; fail-closed combos; partial-RoPE bit-exactness).
- `tests/unit/runtime/rollout/test_hybrid_engine_rollout.py`: updated
  `test_continuous_generation_rejects_hybrid_models_without_state_pool_support`
  to the new paged contract.

## Test / e2e status

- Unit: 12/12 `test_paged_cache.py`, 138 `test_hybrid_engine_rollout.py`,
  56 static_cache+segment_ki — all green. Pre-commit green on all 4 files.
- E2E on shared A100-40GB (Qwen3.5-4B snapshot): paged-hybrid CB greedy +
  sampling ×3 with graph capture working (no "Rollout graph capture failed"
  fallbacks).
- Content validation vs fp32/bf16 native-HF `generate` oracle (4 prompts,
  greedy): paged b=4 T3 = [17, 8, 393, 14008] matches oracle [17, 553, 393,
  14008] except T1 position where static also differs ([8] vs [553] there);
  paged b=1 row1 T3 = 553 = oracle; static b=1 row1 T3 = 8. Logit margins:
  row1 step2 553−8 = +0.1104 (bf16 tie), row2 step2 393−279 = +1.9347 →
  the *static fused decode_attn* path has a pre-existing numeric error
  (paged + old CB-native SDPA both produce the oracle-correct 393).
  Flagged, not fixed.

## Performance (A100-40GB, b=4, 3072→1024 tokens, warmup=1, ×3, segki, ignore-eos)

| Path | Greedy wall | Sampling wall | Peak mem |
|---|---|---|---|
| Paged (this branch) | 14,947 ms | 15,941 ms | 12,217 MB |
| Static (segki CB) | 14,304 ms | 15,283 ms | 15,096 MB |
| vLLM | 10,121 ms | 10,260 ms | — |

Verdict: paged is **not** a wall-time win at this size (+4% wall, −19% peak
memory). Root cause: the static kernel path fuses the q-side into segki
`decode_attn` over dense KV, while paged runs the full HF attention forward in
Python (q_proj chunk → gate + q/k norms + partial RoPE) plus
`BlockedFlashAttn` over block tables, plus per-step `update_graph_atoms()`
refreshing atom length fields from `_write_positions` before every
replay/capture, plus paged prefill through block-table atoms. Speedup levers:
fused q-side over paged KV, flashinfer-style decode/prefill paths.

## Blockers / open bugs

1. **Degeneracy (active, not root-caused).** Paged + `--batch-size 1` +
   `--max-new-tokens 3` + `--warmup 0` → `[294, 294, 294]` repetition loop
   (reproduced twice); static with the same settings is correct
   `[294, 507, 553]`; paged with mnt=1024 is correct; static mnt 3/8/16/32 all
   correct. Repro: `bench_continuous_graph.py` on a single-prompt jsonl
   (`/tmp/oracle_row1.jsonl` = line 2 of `data/longbench_20k.jsonl`).
   Investigation state: step-1 decode atoms are identical between mnt=3 and
   mnt=1024; mnt≤64 all give `max_blocks_per_row=49` so block-per-row count is
   not an obvious trigger. Was mid-read of how `_paged_prefill`'s
   `cache.set_write_position(write_positions)` validates against the base
   class's requirement of a 1-D tensor covering all rows, while the graph
   loop passes a dict `write_positions = {}` (~line 1083 of
   `hybrid_engine_rollout.py`).
2. **Pre-existing static numeric error** (see content validation above):
   static fused `decode_attn` picks token 279 where oracle/paged pick 393 on
   row 2 of T3. Not introduced by paged.
3. **No wall-time speedup** — see performance section; paged currently buys
   memory, not speed, at this problem size.

## Links followed

- AVMP — paged attention for hybrid models: https://arxiv.org/abs/2605.22416
- Marconi — blocked KV for LLM serving: https://arxiv.org/abs/2411.19379
- HF reference protocol: `transformers/cache_utils.py` (`update_conv_state`
  left-pad semantics, ~line 1192) and
  `transformers/models/qwen3_5/modeling_qwen3_5.py`
  (`Qwen3_5Attention.forward` ~line 736, `apply_rotary_pos_emb` ~line 660).
  Local copy:
  `/work/nvme/bcjw/msajid/conda/opsd/lib/python3.12/site-packages/transformers/`.

## Reproduction / run environment

- Bench harness (examples repo, intentionally **uncommitted**):
  `/u/msajid/projects/DeepSpeedExamples/training/opsd/benchmarks/bench_continuous_graph.py`
  with `--use-paged-cache --enable-paged-prefill [--save-response-ids]`
  (first N jsonl lines, left-padded to 3072; top-level `response_ids` when
  the save flag is set). Static baseline: same command without the paged
  flags.
- Perf results + extended paged section:
  `/u/msajid/projects/DeepSpeedExamples/training/opsd/runs/perf_cb_gdn.md`,
  `runs/perf_cb_b4_{greedy,sampling}_paged.json` (+ `*_pagedids.json`,
  `*_staticids.json`); oracles: `runs/oracle_paged_vs_static.py`,
  `runs/oracle_margin_probe.py`.
- Model snapshot:
  `/work/nvme/bcjw/msajid/hf_cache/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Run env: `PATH=/work/nvme/bcjw/msajid/conda/opsd/bin:$PATH`,
  `CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2`, CPATH set to
  the cccl/math_libs includes of that CUDA, `CC=/usr/bin/gcc`,
  `CXX=/usr/bin/g++`, `TORCH_CUDA_ARCH_LIST=8.0`,
  `HF_HOME=/work/nvme/bcjw/msajid/hf_cache`, `PYTHONUNBUFFERED=1`; torch
  2.15.0+cu132; segki/torch extensions cached in
  `/u/msajid/.cache/torch_extensions/py312_cu132/`; deepspeed is an editable
  install of `/u/msajid/projects/DeepSpeed`.
- Single shared A100: never run two GPU jobs concurrently.

## Housekeeping

- Untracked 6.9 GB core dump in the DeepSpeed repo
  (`core.omp.1794277609.92210.3236077.dt-login03.delta.ncsa.illinois.edu`) —
  safe to delete.

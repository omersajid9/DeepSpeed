# Paged attention for the rollout — plan

Status: phases 0-2 and 4 implemented (graph-safe blocked decode), plus sampling in chunked prefill.
Phase 3 is implemented for paged prefill and chunked paged prefill (one request at a time; mixed
prefill+decode in one forward remains open). It is opt-in via `enable_paged_prefill`; the previous
dense-staging graph prefill remains available. Phase 5 block hashing and ref-counted block sharing are
implemented; weight-version salts, `prefix_id` sessions, and cross-cache reuse remain open.
Owner: rollout/graph workstream.
Target: `deepspeed/runtime/rollout/` on the hybrid-engine path, for RL/distillation training.
Read with: `plans/../optimization_strategy.md` §3.4.2, §4.1, §9 (R2/R8/R17/R20/R21) and `cuda-graph-sampling-v2.md`.

---

## 0 · Implementation status and 6k profile

Implemented in `deepspeed/runtime/rollout/paged_cache.py`:

- `PagedAttentionCache`: `BlockedKVCache`/`BlockedAllocator`, per-row `block_ids` table, grow-on-boundary
  allocation, retire/free, dense-prefill copy, and blocked-flash decode.
- Qwen3 attention shim: decodes with `BlockedFlashAttn`; falls back to the original HF attention for the
  dense prefill/staging cache.
- `HybridEngineRolloutConfig(use_paged_cache=True, paged_block_size=None)`; the benchmark flag is
  `--use-paged-cache`. Paged decode supports graph capture/replay, requires continuous batching, and
  excludes aligned decode fronts and unaligned trimming.
- Atoms and block allocation are built once per decode step, not once per layer.

Measured on Qwen3-4B, A40, bf16, LongBench prompt truncated to 6144 tokens, batch 1, 16 new tokens,
no graph capture, warmup 1 (`DeepSpeedExamples/training/opsd/runs/paged_attention_6k.md`):

| path | prefill ms | decode ms/token | total ms | peak MiB | token SHA |
|---|---:|---:|---:|---:|---|
| dense static cache | 714.5 | 51.59 | 1513.8 | 9931.1 | `d4a324e9…` |
| paged decode | 713.5 | 45.70 | 1507.2 | 9938.7 | `d4a324e9…` |

The blocked kernel is faster per layer (0.167 ms vs 0.346 ms for dense SDPA with the decode mask), so
eager decode improves 11.4% and the response tokens are identical.

### Four-way: graph + chunked prefill + Gumbel sampler + paged

Same model and 6k prompt, now with graph capture, chunked prefill (chunk 512), and the Gumbel
device sampler (`temperature=0.7`, `top_p=0.9`, seed 0), warmup 1
(`DeepSpeedExamples/training/opsd/runs/fourway_6k.md`):

| cache | prefill ms | decode ms/token | total ms | tokens/s | peak MiB |
|---|---:|---:|---:|---:|---:|
| dense static | 1379.3 | 39.73 | 2011.7 | 7.95 | 9503.6 |
| paged | 1381.5 | **25.86** | 1820.4 | 8.79 | 9511.6 |

Paged decode is **1.54x** faster and end-to-end is 1.10x; 97/97 blocks (64 tokens each) are used.
The plan's 25 ms decode target is essentially met (within 3.4%) once graphs remove the launch/CPU
overhead. The accuracy verifier reports identical response tokens and a 1.5e-3 max logprob drift
on the two recorded positions; the current graph path records logprobs only for the final two
response steps, a pre-existing logging gap shared by both rows.

Graph capture required one non-obvious invariant: the per-width row-index and atom tensors embedded
in a captured kernel must stay alive across replays. Recreating them in `set_graph_mode` invalidated
the captured index pointers and caused a device-side `index out of bounds` on the second replay;
`PagedAttentionCache` now caches those buffers per width.

### Phase 3/5 implementation and measurement

`PagedAttentionCache` now supports `q_len > 1` attention and direct paged prefill:
`HybridEngineRollout._paged_prefill` strips left padding, splits the prompt into
`prefill_chunk_size` windows, writes K/V into the block table, and runs the blocked kernel
without a dense staging cache or a prefill-to-paged copy. Query blocks wider than the kernel's
`q_block_size` are split into one atom per query block. The completion of a prefilling chunk
returns the first response token from the last chunk's logits, so sampling and greedy rollouts
share the same first-token path.

Full-block prefix reuse is implemented with a chained blake2b hash per block, a hash→block map,
and per-block refcounts. `prepare_prefix` points a new row at the matched block chain and
increments its refs; `register_prefix` records newly written full blocks; `_release_blocks` only
returns a block to the allocator when the last row releases it. Reuse is restricted to the
non-graph-safe cache because the graph-safe cache preallocates every block table; it currently
reuses blocks shared by rows in the same cache, not blocks retained after a row retires.

Measured on Qwen3-4B, A40, bf16, no graph capture, paged cache, chunk 512, 8 new tokens
(`DeepSpeedExamples/training/opsd/runs/paged_prefill_prefix.md`):

| workload | path | prefill ms | forwards | total ms | peak MiB | token SHA |
|---|---|---:|---:|---:|---:|---|
| 6k, batch 2 | dense prefill | 3253.7 | 24 | 4509.2 | 10522.8 | `5530534d…` |
| 6k, batch 2 | paged prefill | 2081.3 | 24 | 2704.6 | 9521.3 | `5530534d…` |
| 20k, batch 1 | dense prefill | 10913.2 | 39 | 11954.9 | 13585.9 | `c4291b80…` |
| 20k, batch 1 | paged prefill | 3972.8 | 39 | 4591.5 | 10521.6 | `c4291b80…` |
| 6k, batch 2, duplicate prompts | paged, no prefix reuse | 2081.3 | 24 | 2704.6 | 9521.3 | `5530534d…` |
| 6k, batch 2, duplicate prompts | paged, prefix reuse | 1269.7 | 13 | 1875.8 | 9521.3 | `5530534d…` |

The duplicate-prefix run reused 95 blocks / 6080 tokens and cut prefill forwards from 24 to 13.
With graph capture, chunked prefill, and the sampler on the 6k four-way path, paged prefill is
822.7 ms versus 1381.6 ms for dense-staging + copy; total is 1233.8 ms versus 1809.2 ms
(tok/s 12.97 versus 8.84), with token-identical output (`d4a324e9…`).

A head-to-head vLLM 0.33.0 run (same Qwen3-4B, prompt, tokenizer, chunk 512, batch 1, sampler,
and generated-token count; vLLM KV pool capped to 1.1 GiB at 6k) is 954.8 ms / 16.76 tok/s at
6k and 3363.6 ms / 2.38 tok/s at 20k greedy. The DeepSpeed paged path is **1.29x slower at 6k**
and **1.15x slower at 20k**, with identical 6k and 20k greedy tokens; details in
`DeepSpeedExamples/training/opsd/runs/vllm_comparison.md`.
Peak memory did not drop because `PagedAttentionCache` sizes the `BlockedKVCache` pool for the
full per-row block table; prefix reuse saves prefill compute and KV writes, not pool capacity.

Limitations: prefill is one request at a time, so the phase-3 mixed prefill+decode batch is not
implemented; prefix reuse has no weight-version salt and no session id; Qwen3.5-hybrid remains
unsupported by the paged shim.

---

## 1 · Why

Three separate problems in the current rollout all reduce to "the KV cache is one contiguous
per-row buffer and attention sees a dense mask". Paging fixes all three with one change of data
plane.

| problem today | evidence | what paging buys |
|---|---|---|
| **No prefix reuse**: every rollout re-prefills its system prompt / prior turns | `use_shared_prefill` is within-batch and native-path only; no prefix cache exists | block-granular reuse: a shared prefix is *shared blocks*, not copied KV (roadmap 2.3) |
| **Attention is dense-masked, so FA2 is refused under capture and long-context decode is slow** | §8.2: 6k-context forward is 66.5 ms/step ≈ 5.8× the weight-read roofline and *identical* with/without graphs; `_validate_continuous_graph_options` raises on `flash_attention_2`; decode passes a dense additive `[B,1,1,max_cache_len]` mask | the paged/varlen kernels take per-row lengths (atoms) instead of a mask, so the mask build disappears and the fast attention path is usable — including under graphs (R2) |
| **Memory is worst-case-per-row**: capacity is `rows × (prompt + max_new_tokens)`, compaction copies live columns, trimming shifts them | `_estimate_continuous_cache_len`, `cache.compact`, `_trim_left` | capacity becomes a shared block pool: `rows` and context length decouple from fragmentation, trim/reclaim becomes block release |

Also load-bearing for later items: **2.3 needs a block table** to be cheap at all, and **R21 (one graph
with mixed prefill+decode rows)** needs a layout where rows do not have to share a front.

## 2 · Non-goals

- Not porting vLLM's scheduler policy layer: priority queues, admission fairness, 503s, KV
  connectors/offload, multi-tenant eviction tuning (all serving-shaped; see §9 "not to copy").
- Not CPU/NVMe offload (`BlockedKVCache.offload/restore` raise `NotImplementedError` today — leave it).
- Not changing the sampler, the scheduler bookkeeping, or the ZeRO-3 lockstep contract.
- Not making paging opt-out for correctness: it lands behind a config flag, greedy path first.

## 3 · Prior art, and what to take

### 3.1 This repo already ships the paged stack (this is the important finding)

`deepspeed/inference/v2/ragged/` + `kernels/ragged_ops/` contain a complete blocked-KV data plane with
device kernels. Verified surface:

| piece | anchor | shape / API |
|---|---|---|
| block allocator (free/used linked list) | `ragged/blocked_allocator.py:33-105` | `allocate(n) -> Tensor[block_ids]`, `free(blocks)`, `free_blocks` |
| blocked KV cache | `ragged/kv_cache.py:37-205` | storage `(num_caches, num_blocks, block_size, 2, heads, head_dim)`; `free_blocks`; `get_cache(cache_id, group)`; `split_kv` → `[blocks, block_size, heads, head_dim]` |
| sequence descriptor | `ragged/sequence_descriptor.py:59-280` | `seen_tokens`, `cur_allocated_blocks`, `kv_blocks_ptr`, padded `[groups, max_blocks]` block-id list |
| ragged batch | `ragged/ragged_wrapper.py:31-292` | `insert_sequence(desc, tokens)`, `finalize()`, `batch_metadata_buffer(on_device)`, `kv_ptrs`, `input_ids`, `tokens_to_seq` |
| **atom builder (device kernel)** | `kernels/ragged_ops/atom_builder/atom_builder.py:15-50` | `(atoms[max_atoms,8] int32, ragged_batch, q_block_size) -> (atoms, n_atoms)` |
| **blocked flash attention (device kernel)** | `kernels/ragged_ops/blocked_flash/blocked_flash.py:55-107` | `(out, q, k, v, atoms, softmax_scale)`; `q/out [tokens, hidden]`, `k/v [blocks, block_size, kv_heads, head_dim]`, `atoms [n,8]`; fp16/bf16 only; CC ≥ 8; `head_size % 16 == 0` |
| atom layout | `kernels/ragged_ops/blocked_flash/attention_atom.h:11-38` | `block_idx_list, q_start_idx, q_len, kv_blocks, total_extent, global_q_idx` |
| blocked KV writes + rotary (pre-RoPE'd keys into blocks, GQA ratios 1-8) | `kernels/ragged_ops/linear_blocked_kv_rotary/{blocked_kv_rotary,blocked_trained_kv_rotary,linear_blocked_kv_copy}.py` | head sizes 64/80/96/128 |
| block-size rules | `blocked_flash.py:12-53` | `get_q_block_size`, `get_kv_block_size(head_size)` → 128 / 64 / 32 |
| builder | `op_builder/ragged_ops.py` (`DS_BUILD_RAGGED_DEVICE_OPS`) | `RaggedOpsBuilder().load()`, CC gate |
| **block table is a view, not a tensor** | `ragged/ragged_manager.py:66-80`, `ragged/sequence_descriptor.py:98-133,192` | one `int32 [max_tracked_sequences, groups, max_blocks]` buffer; a row is a view, and kernels take its `data_ptr()` (`kv_blocks_ptr`) |
| pinned host shadows + one-shot H2D | `ragged/ragged_wrapper.py:79-121,184` (`finalize()`), `sequence_descriptor.py:177-235` (`extend_kv_cache`) | the R14 pattern is already implemented here: build on the host, upload once per step |
| kernel ABI structs | `kernels/ragged_ops/ragged_helpers/ragged_dtypes.h:7-35` | `RaggedBatchDescriptor`, `InflightSeqDescriptor`, `KVCacheDescriptor{int32** block_lists, block_size, n_blocks}` — the contract any metadata builder must produce |
| block→address math for the KV write | `linear_blocked_kv_rotary/blocked_kv_rotary_cuda.cu:79-95` | `global = seen + (tok - start)`; `blk = block_lists[row][global / B]`; `off = (blk*B + global % B) * stride1 + head*head_dim` |
| KV write variants | `linear_blocked_kv_rotary/{blocked_kv_rotary,blocked_trained_kv_rotary,linear_blocked_kv_copy}.py` | synthesized RoPE, trained inv_freq RoPE, and a no-RoPE copy; GQA `q_ratio = n_q / n_kv` (only `head_idx % q_ratio == 0` writes KV); requires `qkv_stride == k.stride(0) == v.stride(0)`; head sizes 64/80/96/128 |
| atom count rule | `blocked_flash/atom_builder.cpp:26-47`, `modules/implementations/attention/dense_blocked_attention.py:113` | `n_atoms = Σ_rows ceil(q_len / q_block_size)`, `max_atoms = max_ragged_sequence_count`; decode ⇒ exactly one atom per row |
| state manager (optional for a rollout) | `ragged/ragged_manager.py:55-203` | tracking-slot allocator, per-uid descriptors, `flush_sequence`; a rollout that owns its rows needs the block row, not this layer |
| reserve vs allocate, TP agreement | `ragged/kv_cache.py:141-152` | `AllocationMode.RESERVE` vs `ALLOCATE`; inside a TP group the block count is min-reduced so ranks agree |

So the plan is **not** "write paged attention". It is: *adopt the V2 ragged/blocked stack inside the
hybrid-engine rollout*, keeping the rollout's write-position protocol, bucket ladder and scheduler.

**Two gaps in that stack, which this plan has to build:**

1. **No sharing/ref-counts.** `BlockedAllocator` is a free/used list only, mutated by a Python loop
   (`:33-105`). Prefix reuse needs blocks referenced by more than one row and freed only when the last
   reference dies — vLLM's block pool carries `ref_cnt` for exactly this. Add a thin ref-count map on top of
   `BlockedAllocator` rather than forking it, and reserve in batches (per-token-boundary calls would put a
   Python loop on the step's critical path).
2. **No CPU path for tests.** `BlockedFlashAttn` requires CC ≥ 8 *and links an external `dskernels`
   library* (`-lblockedflash`, see `blocked_flash.cpp`); the ragged plumbing additionally needs
   `RaggedUtilsBuilder` (`allocate_fast_host_buffer`), not just `RaggedOpsBuilder`. CPU CI is where our
   correctness tests live, so keep a **materialise-then-dense** reference: gather a row's blocks by slot map
   into a contiguous `[1, heads, len, head_dim]` view and call SDPA, behind the same interface as the blocked
   kernel. That oracle is also what makes the build dependency non-blocking: phases 0-1 need no CUDA ops at
   all.

### 3.2 Lessons from the reference implementations (audited in §9)

- **vLLM**: block-granular hashing with a chained key (`hash_block_tokens(parent, block_tokens, extra_keys)`,
  `kv_cache_utils.py:703-733`), LRU free queue, `ref_cnt` sharing, hash lookup once per admission; keys carry
  **no weight hash** — their own RLHF comment plus `reset_prefix_cache` (which preempts every running
  request) is the failure mode to avoid. Take the hash chain and the ref-counted pool; replace the flush with
  a **weight-version salt in `extra_keys`**.
- **vLLM (metadata)**: `block_table`/`slot_mapping` are ordinary device tensors built by a kernel/host step
  and consumed by attention; nothing about the layout requires a scheduler thread — we can build them in our
  own loop. Specifics worth matching or deliberately diverging from:
  - page layout: theirs is per layer `[num_blocks, num_kv_heads, block_size, 2*head_size]` with K and V
    **interleaved in the page tail**, and FA views it as `transpose(1,2).split(head_size)` →
    `[num_blocks, block_size, num_kv_heads, head_size]`. Ours (the in-repo kernel's) is
    `[num_blocks, block_size, 2, n_kv_heads, head_size]` split by `split_kv` with k/v passed separately — keep
    ours, do not copy the interleaving;
  - block size: their manager default is **64 tokens** (`config/cache.py:185,481-482`) and FA accepts any
    multiple of 16; ours comes from `get_kv_block_size(head_size)` (128/64/32). Their manager block can be
    larger than the kernel block and the kernel table is *derived*, not stored — we should just pick one
    block size and skip that layer (`v1/worker/utils.py:371-410`);
  - `slot_mapping` is `int64 [num_kv_groups, max_num_batched_tokens]` — **shape over tokens, not rows** —
    with `PAD_SLOT_ID = -1` and `NULL_BLOCK_ID = 0` (`attention/backends/utils.py:51-52`), which matches our
    `slot_map [max_tokens]` design; padding is written as `-1` by a kernel rather than by Python;
  - `block_table` is per group `int32 [max_num_reqs, max_num_blocks]`, appended CPU→UVA in staged writes, and
    the forward reads a **gathered** copy (`gather_block_tables`) per group — we can index rows directly
    because we own the loop, so the gather is unnecessary for us;
  - decode and prefill/chunk are **the same paged call**, differing only in `query_start_loc`/`max_query_len`
    (`flash_attn.py:586-620,1355-1381`) — independent confirmation that one code path covers both, which is
    what phase 3 assumes;
  - the mask disappears entirely: `causal=True` plus per-row `seqused_k` replaces the 4-D additive mask, and
    `window_size` gives sliding-window attention for free (`flash_attn.py:428,1355-1381`) — this is the direct
    answer to the §8.2 anomaly, and it also closes the roadmap's sliding-window appendix item;
  - capture: their flashinfer backend records the **addresses** of `seq_lens` and the page table at capture
    time, preallocates dummy block tables/slot mappings, writes them in place every step, zeroes stale rows
    and pads with `-1` (`flashinfer_ts.py:203-245`, `block_table.py:get_dummy_*`) — the same discipline our
    phase 4 needs;
  - the free-queue + hash→block prefix cache + refcounts live in one pool object
    (`core/block_pool.py:112-190`), which is the shape of the ref-count layer §3.1 says we must add.
- **neuron-rl-core**: block tables and slot mappings are built **on the host in numpy** and shipped as one
  int32 upload per step (`vllm_lite/host_step.py:19-59,80-101`); a *single* paged kernel launch serves a mixed
  prefill+decode batch, with the entry-point variant chosen **on the host** from cheap counters
  (`decode_only = counts_prefill == 0 and counts_decode > 0`, plus a chunk-slice choice from the maximum
  `nblk_vis`; `core/kernels/paged_attention/kernel.py:556-580`, `kernel_decode.py:1265-1356`), and their mixed
  kernel's decodes are **bit-identical** to the decode-only entry. Portable mechanics worth copying:
  - a table invariant: `int32 [R_MAX, NB_MAX]`, `-1` marks an unused tail, and there must be **no `-1` hole
    inside the used prefix** `[0, ceil(seq_len / block_size))` (`vllm_lite/models/attention.py:449-499`);
  - the packed query dim is the row-major concatenation of each row's queries, so
    `q_start_idx == cumsum(q_len)` — atoms (and any varlen call) inherit that invariant;
  - the KV write for a step is **one scatter per pool**: slot `blk*B + pos % B` → flat row
    `(slot // B) * n_kv_heads * B + head * B + slot % B`, applied to a `view(-1, head_dim)` of the pool
    (`attention.py:178-215,685`) — not a per-row Python loop;
  - a **sink page**: padded/`-1` slots write to a dedicated never-referenced block (`sink_slot = NPHYS * B`,
    `attention.py:167-169`), so rows that must keep a stable table entry (parked bucket rows) have somewhere
    harmless to point;
  - rows with no item leave the attention output **uninitialised**, not zeroed — a consumer must gather only
    active rows (the same rule as R1).

## 4 · Target design

### 4.1 Data structures (rollout-owned)

```
blocks      : BlockedKVCache over KVCacheConfig(block_size = get_kv_block_size(head_dim))
              storage (layers, num_blocks, block_size, 2, kv_heads, head_dim)
block_ids   : [max_rows, max_blocks_per_row] int32, device-resident, STATIC   # the block table
row_len     : [max_rows] int32, device-resident  (context length per row; reuse write_position)
slot_map    : [max_tokens] int32 per step   -> block_ids[row, pos // B] * B + pos % B
atoms       : [max_atoms, 8] int32, device-resident (built by AtomBuilder)
row_refs    : block -> ref count (NEW, see §3.1 gap 1)
```

Invariants to keep from today's cache: per-row write position advanced by the caller; rows are
independent; a row's blocks are append-only until it retires (no compaction copies).

New invariants this design adds (from the neuron contract, `attention.py:449-499`):
`block_ids` is `int32 [max_rows, max_blocks]` with `-1` only in the unused tail and no `-1` hole inside a
row's used prefix; the packed token dim is the row-major concatenation of each row's queries, so
`q_start_idx == cumsum(q_len)` per step; the step's KV is written **before** the attention launch, as one
scatter per pool over a `view(-1, head_dim)` (never a per-row Python loop); and attention output rows for
rows without an atom are left uninitialised, so the logits gather must read only active rows (R1).

### 4.2 Step flow

**Decode step (R rows, one token each):** the slot for each row's new position is
`block_ids[row, p // B] * B + p % B` (the same arithmetic the write kernel does internally,
`blocked_kv_rotary_cuda.cu:79-95`); the K/V write (with RoPE, three variants available) goes through
`KVCacheDescriptor`; `AtomBuilder` emits **one atom per row** for `q_len == 1`
(`n_atoms = Σ ceil(q_len / q_block_size)`); one `BlockedFlashAttn` call over the packed query rows
produces the attention output. No mask tensor is built.

**Prefill / staged chunk (C rows × chunk tokens):** same, with `q_len = chunk` and `q_start_idx` pointing
into the packed token dim; the atom's `kv_blocks` covers the row's already-written prefix, `total_extent`
the partially filled head block. This is where the current dense per-chunk mask disappears.

**Mixed step (prefill chunks + decoding rows together, R21):** one atom list, one launch. This is the
end-state the bucketed continuous loop wants.

**Allocation:** when `row_len` crosses a block boundary, `reserve(1)` and append to `block_ids[row]`;
when a row retires, `free` (or decrement refs). Cache capacity = `num_blocks × block_size` tokens shared
by all rows, replacing `_estimate_continuous_cache_len`'s worst-case sizing.

### 4.3 Attention integration — the one real design fork

Paging is only real if the *model's* attention reads through the block table. Today the rollout runs HF
model code with `past_key_values=<transformers Cache>` and HF's own SDPA path (`_generate_graph`,
`_generate_continuous*`). Three ways to get `BlockedFlashAttn` in:

| option | how | pro | con |
|---|---|---|---|
| **A. attention shim (prototype)** | wrap/replace each attention module's `forward` while the rollout runs (the same trick `module_inject.replace_module` uses, scoped to the rollout) | small, reversible, no engine-mode switch; keeps HF masks/rope for the reference path | monkey-patching HF internals; must handle per-model differences (GQA ratios, sliding-window layers, Qwen3.5 hybrid state) |
| **B. hybrid-engine kernel injection (target)** | extend the native containers so batching can stay inside the fused path with a `Cache`-object-aware forward (roadmap 4.1.2a) | one supported path, no patching, aligns with the hybrid engine's fusion story | the native path is tied to the workspace + legacy tuple caches; larger change, and it must keep the graph path alive |
| **C. keep HF attention, materialise per row** | gather blocks into a contiguous per-row view each step, call SDPA | zero risk, useful as an oracle | copies O(context) KV per step — defeats the purpose; only for tests |

**Recommendation:** build A first (it is what makes phases 2-3 testable), migrate to B once the cache and
atoms are proven, and keep C permanently as the CPU oracle behind the same `PagedAttention` interface.

### 4.4 Lifecycle, trimming, capacity

- `cache.compact(keep_slots)` → drop/renumber table rows (no data movement).
- `_trim_left` → release whole blocks, and either forbid trimming inside the head block or pay one ≤B-token
  copy (keep it as an explicit, measured option).
- `cache_memory_bytes`, `peak_cache_length` → recompute from the block pool; add `blocks_in_use`,
  `block_utilization`, `reused_tokens` counters (R11/R19 scoreboard).

### 4.5 Prefix reuse on top (the 2.3 payoff)

Per-block hash chain (vLLM's `hash_block_tokens` shape), key = `(parent_hash, block tokens, weight_version)`
where `weight_version` is the **counter that `sync_weights(step)` must start maintaining** (R17). Lookup on
admission maps a prompt prefix to an existing block list → the new row's `block_ids` starts as a *copy of the
list with refs incremented*; only the uncached tail is prefilled. Multi-turn sessions reuse the same
mechanism with a session id instead of re-hashing the conversation (R16). Partial (sub-block) hits: optional,
copy-on-write of one block, only if measurement shows it matters.

### 4.6 Graph capture

The capture-friendly shape is: allocator/table/atom buffers are **static tensors updated in place**, and
`AtomBuilder` runs *outside* the captured region (or inside it if its output buffer is static — the
experiment in phase 4). Bucket keys become `(rows, uniform_q_len, variant)` (R20), and the attention op is
left out of the captured segment (R2's piecewise route) so a paged kernel needs no capturability.

## 5 · Phases

Each phase is independently shippable behind `HybridEngineRolloutConfig(use_paged_cache=...)`; each has a
rollback (flag off) and a metric.

| phase | scope | files | tests / metric |
|---|---|---|---|
| **0 · interface + oracle** | `PagedCache` protocol (`slot_map`, `block_ids`, `row_len`, `write(k,v,layer,slots)`, `materialize(row)`) with the existing contiguous cache as one implementation and a dense-view + SDPA oracle as another | new `runtime/rollout/paged_cache.py`, tests | CPU unit tests: slots ↔ positions for arbitrary row lengths; oracle equals `DeepSpeedStaticCache` output token-for-token (`top_k=1`) |
| **1 · blocked container** | adopt `BlockedKVCache`/`BlockedAllocator` behind the protocol; grow-on-boundary allocation; retire frees blocks; capacity from the pool | `paged_cache.py`, `hybrid_engine_rollout.py` (cache construction, stats) | parity vs phase 0 oracle on the existing bucketed continuous tests; `blocks_in_use`, utilisation counters |
| **2 · atoms + blocked attention (shim)** | `AtomBuilder` + `BlockedFlashAttn` in the decode path via option A; dense masks deleted from the paged path | shim module, `hybrid_engine_rollout.py` | GPU: logits parity vs oracle at 1/8/64 rows, 1k/6k/32k context; **the §8.2 experiment**: 6k decode ms/step and achieved DRAM GB/s vs 66.5 ms baseline |
| **3 · prefill + mixed** | chunked prefill writes through the table; one atom list for prefill+decode rows (R21) | shim + loops | mixed-batch ITL flat vs today; staged-chunk tests stay token-exact |
| **4 · graphs** | static atom/table buffers, `AtomBuilder` outside capture, bucket key gains the variant (R20); attention excluded from the segment (R2) | graph paths | capture count per bucket; replay parity; no re-capture per optimizer step (2.4.1) |
| **5 · prefix reuse** | hash chain + weight-version salt + ref counts; `prefix_id` sessions (R16) | scheduler + cache + `base.py` (`RolloutRequest.prefix_id`) | prefill tokens served/served-token → ~0 on repeat prefixes; stale-version batches = 0 (R22) |

Suggested order of *value*: phase 2 is the one that addresses the measured anomaly; phase 5 is the one that
pays for agentic workloads; phases 0-1 are the price of both. Do not start phase 5 before R17's version
counter exists.

## 6 · Risks and open questions

1. **Does `BlockedFlashAttn` handle `q_len > 1` per atom with a partially filled head block** (staged chunk
   ending mid-block)? The atom struct implies yes (`q_len`, `total_extent`); confirm with a unit test before
   phase 3.
2. **Is `AtomBuilder` replay-safe** (static input buffer, no host sync, deterministic)? This decides whether
   phase 4 needs the attention op outside the graph only, or atoms too.
3. **CC ≥ 8 requirement** (`blocked_flash.py:12-53`): no paged path on pre-Ampere, and no CPU path — the phase-0
   oracle is mandatory for CI.
4. **Head-size/GQA coverage**: blocked rotary supports head sizes 64-128 and q-ratios 1-8; a model outside
   that set needs either a fallback to the oracle path or a kernel extension. Qwen3.5-hybrid (linear layers)
   is out of scope until its state is cacheable.
5. **Injection vs shim** (§4.3): if option B turns out to be required for the fused path, phases 2-4 should be
   re-planned around the container contract rather than the shim.
6. **Allocator contention**: allocation happens on the host today (`reserve` returns a tensor built in
   Python). Under graph capture and many rows, block growth must be pre-reserved or done outside capture.
7. **Memory accounting**: graphs + pools + block pool must be budgeted before the KV allocation
   (vLLM's `profile_cudagraph_memory` pattern), otherwise the first capture can OOM what the pool needed.
8. **Build surface**: `BlockedFlashAttn` needs the external `dskernels`/`-lblockedflash` link, and the ragged
   batch plumbing needs `RaggedUtilsBuilder` alongside `RaggedOpsBuilder`. Verify both build on the target box
   before phase 2; phases 0-1 are deliberately op-free.
9. **Stride coupling on the write side**: the blocked write requires
   `qkv_stride == k.stride(0) == v.stride(0)`, i.e. the KV write must consume the same packed-QKV layout the
   V2 kernels expect (or pay a copy). Decide this when wiring the shim (option A) — it constrains how the
   attention module is replaced.
10. **TP**: `BlockedKVCache` min-reduces block counts across a TP group so ranks agree on pool size
    (`kv_cache.py:141-152`). Our rollout paths are single-rank today; if the rollout is ever TP-sharded the
    pool sizing becomes a collective, and `AllocationMode` matters.
11. **Allocator granularity**: `allocate` walks a pinned host list in Python; pre-reserve a row's blocks (or
    batch allocations per step) instead of allocating at every block boundary.
12. **Parked rows in captured buckets need valid table entries.** Our bucket ladder replays a width wider than
    the active rows; a captured kernel that walks a row's block list will read whatever the table holds there.
    Two mitigations, both cheap: emit atoms only for active rows (preferred - the kernel then never touches the
    parked rows), and keep a **sink page** the parked rows point at so their entries stay stable across replays.
13. **Mixed vs decode-only parity.** neuron's mixed entry is bit-identical to its decode-only entry, which is
    the property to test for us too: once phase 3 has one atom list for prefill+decode, assert the decode rows
    produce the same logits as the decode-only path at the same width (a cheap regression test that catches
    atom/packing mistakes the token-stream comparison would hide behind sampling).

## 7 · What this replaces in the roadmap

- **3.4.2 "introduce a block table"** — this plan *is* that item; it also absorbs 3.4.1 (capacity) and the
  compaction/trim copies (3.4's "verify" bullets).
- **4.1 (varlen attention)** — split into: 4.1a "varlen without paging" (a smaller change that may already fix
  the 6k forward: per-row `cu_seqlens` over the existing contiguous cache, no block table) and 4.1b paging
  (this plan). If 4.1a recovers the forward time, phases 0-2 lose their *latency* justification and keep only
  the reuse/memory one — worth measuring first, since it is a fraction of the work.
- **2.3 (prefix caching)** — implemented on top here (phase 5), with R17's version salt as a gate.
- Enables **R8/R16/R17/R20/R21** and removes the mask-rebuild item (3.3) on the paged path.

## 8 · Success and abort criteria

Success (phases 0-2): 6k-context decode step recovers most of the gap to the weight-read roofline
(target: 66.5 ms → ≤25 ms at batch 1 on the A40), token-exact parity with the oracle, no dense mask built,
and the same tests passing on CPU through the oracle.

Success (phase 5): shared-prefix prefill tokens served from cache; `prefix_tokens_reused / total_prefill`
tracked; zero stale-version batches.

Abort / re-scope:
- If **4.1a (varlen, no paging)** already lands the 6k recovery, ship that first and re-derive this plan's
  value from prefix reuse + memory only.
- If the **shim vs injection** fork cannot be resolved without the container rewrite, stop after phase 1
  (the block pool still buys capacity accounting and the prefix-cache substrate).

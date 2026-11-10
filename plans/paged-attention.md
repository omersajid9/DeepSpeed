# Paged attention for the rollout — plan

Status: plan, nothing implemented. Owner: rollout/graph workstream.
Target: `deepspeed/runtime/rollout/` on the hybrid-engine path, for RL/distillation training.
Read with: `plans/../optimization_strategy.md` §3.4.2, §4.1, §9 (R2/R8/R17/R20/R21) and `cuda-graph-sampling-v2.md`.

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

So the plan is **not** "write paged attention". It is: *adopt the V2 ragged/blocked stack inside the
hybrid-engine rollout*, keeping the rollout's write-position protocol, bucket ladder and scheduler.

**Two gaps in that stack, which this plan has to build:**

1. **No sharing/ref-counts.** `BlockedAllocator` is a free/used list only (`:31-105`). Prefix reuse needs
   blocks to be referenced by more than one row and freed only when the last reference dies — vLLM's
   block pool carries `ref_cnt` for exactly this. Add a thin ref-count map (or block-state table) on top of
   `BlockedAllocator` rather than forking it.
2. **No CUDA path for tests.** `BlockedFlashAttn` requires CC ≥ 8, and CPU CI is where our correctness
   tests live. Keep a **materialise-then-dense** reference: gather a row's blocks by slot map into a
   contiguous `[1, heads, len, head_dim]` view and call `SDPA`/`BlockedFlashAttn` behind one interface, so
   the same tests run on CPU (oracle) and GPU (target).

### 3.2 Lessons from the reference implementations (audited in §9)

- **vLLM**: block-granular hashing with a chained key (`hash_block_tokens(parent, block_tokens, extra_keys)`,
  `kv_cache_utils.py:703-733`), LRU free queue, `ref_cnt` sharing, hash lookup once per admission; keys carry
  **no weight hash** — their own RLHF comment plus `reset_prefix_cache` (which preempts every running
  request) is the failure mode to avoid. Take the hash chain and the ref-counted pool; replace the flush with
  a **weight-version salt in `extra_keys`**.
- **vLLM (metadata)**: `block_table`/`slot_mapping` are ordinary device tensors built by a kernel/host step
  and consumed by attention; nothing about the layout requires a scheduler thread — we can build them in our
  own loop.
- **neuron-rl-core**: block tables and slot mappings are built **on the host in numpy** and shipped as one
  int32 upload per step (`vllm_lite/host_step.py:19-59,80-101`), and a *single* paged kernel launch serves a
  mixed prefill+decode batch with host-selected decode-only entry points (`core/kernels/paged_attention/
  kernel.py:557`, `kernel_decode.py:1309,1334`; decode-only entry saves ~20-40 µs/launch). That is the shape
  our step should end up with: host builds metadata once, one launch serves the mix.

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
independent; a row's blocks are append-only until it retires (no compaction copies); padding rows are
never referenced.

### 4.2 Step flow

**Decode step (R rows, one token each):** host (or device kernel) appends the new position's slot to
`slot_map`; `BlockedKVWrite` writes the token's K/V (with RoPE) into its block slot; `AtomBuilder`
turns `(row_len, kv_blocks, global_q_idx)` into one atom per row; one `BlockedFlashAttn` call over the
packed query rows produces the logits input. No mask tensor is built.

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

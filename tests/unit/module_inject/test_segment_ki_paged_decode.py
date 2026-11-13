# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Paged (block-table) decode attention: references, kernel contract, wiring.

The paged variant of the segment-KI decode-attention kernel must produce the
same result as its dense sibling when the block pool holds the same K/V. Four
layers of checks:

1. Reference equivalence (CPU): the two executable specs agree once the pool
   is materialized densely, including when the page table is shuffled.
2. Fused prologue vs HF (CPU): q|gate split, (1+w) per-head RMSNorm and
   partial rotate-half RoPE against the transformers Qwen3.5 attention
   module they replay.
3. CUDA kernel vs reference (GPU): block boundaries, GQA ratios, 64/128/256
   head dims, bare and fully fused.
4. Rollout wiring (CPU): a paged cache reaches the block-table kernel with
   the cache's page table and write positions, and every unsupported case
   reaches the native forward instead.
"""

from types import SimpleNamespace

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.kernel_reference import decode_attn as ref_decode_attn
from deepspeed.module_inject.kernel_reference import decode_attn_paged as ref_decode_attn_paged
from deepspeed.module_inject.kernel_reference import qwen3_5_fused_prologue as ref_prologue
from deepspeed.module_inject.kernel_reference import write_paged_kv
from deepspeed.module_inject.segment_ki import _paged_decode_attn_forward

DEV = get_accelerator().device_name()


def _kernel_op_available() -> bool:
    """True when this accelerator registers the segment-KI op builder."""
    try:
        builder = get_accelerator().get_op_builder("FusedGLUBuilder")
    except Exception:
        return False
    return builder is not None and builder.__name__ != "NotImplementedBuilder"


GPU_AVAILABLE = _kernel_op_available()


def _blocks_for(length, block_size):
    return (length + block_size - 1) // block_size


def _paged_pool(batch, num_kv_heads, head_dim, length, block_size, seed=0, device=DEV, dtype=torch.float32):
    """A block pool plus a page table whose per-row block order is shuffled.

    Shuffling is what makes the addressing test real: a contiguous walk over
    the pool cannot reproduce the reference unless the table is honored.
    """
    torch.manual_seed(seed)
    blocks_per_row = _blocks_for(length, block_size)
    pool_k = torch.randn(batch * blocks_per_row, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    pool_v = torch.randn_like(pool_k)
    table = torch.arange(batch * blocks_per_row, dtype=torch.int32).view(batch, blocks_per_row)
    for row in range(batch):
        table[row] = table[row][torch.randperm(blocks_per_row)]
    return pool_k.view(-1, num_kv_heads, head_dim), pool_v.view(-1, num_kv_heads, head_dim), table


def _materialize(pool, table, block_size, row, positions):
    """Gather a row's valid KV out of the pool into a dense [kv, len, hd] plane."""
    slots = [int(table[row, i // block_size]) * block_size + i % block_size for i in range(positions)]
    return pool[slots]


# ---------------------------------------------------------------------------
# 1. Reference equivalence (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [4, 16])
def test_paged_reference_matches_dense_reference(block_size):
    """Same K/V, block-table addressing: the paged reference equals the dense one."""
    batch, num_heads, num_kv_heads, head_dim = 2, 8, 2, 32
    positions = torch.tensor([13, 5], dtype=torch.long)
    length = int(positions.max()) + 1
    pool_k, pool_v, table = _paged_pool(batch, num_kv_heads, head_dim, length, block_size, device="cpu")
    query = torch.randn(batch, num_heads, head_dim, dtype=torch.float32)

    paged = ref_decode_attn_paged(query, pool_k, pool_v, table, positions, block_size)

    dense_k = torch.zeros(batch, num_kv_heads, length, head_dim)
    dense_v = torch.zeros_like(dense_k)
    for row in range(batch):
        valid = int(positions[row]) + 1
        dense_k[row, :, :valid] = _materialize(pool_k, table, block_size, row, valid).transpose(0, 1)
        dense_v[row, :, :valid] = _materialize(pool_v, table, block_size, row, valid).transpose(0, 1)
    dense = ref_decode_attn(query, dense_k, dense_v, positions)

    torch.testing.assert_close(paged, dense, atol=1e-6, rtol=1e-6)


def test_fused_prologue_matches_hf_attention():
    """The fused prologue replays HF's q|gate split, per-head norms and partial RoPE."""
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5Attention, Qwen3_5TextRotaryEmbedding,
                                                              apply_rotary_pos_emb)

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(hidden_size=64,
                            intermediate_size=128,
                            num_attention_heads=4,
                            num_key_value_heads=2,
                            head_dim=64,
                            num_hidden_layers=2,
                            layer_types=["full_attention", "linear_attention"],
                            partial_rotary_factor=0.25,
                            rope_theta=10000.0,
                            vocab_size=128,
                            linear_num_value_heads=4,
                            linear_num_key_heads=2,
                            linear_key_head_dim=32,
                            linear_value_head_dim=32,
                            linear_conv_kernel_dim=4)
    attn = Qwen3_5Attention(cfg, layer_idx=0).eval()
    positions = torch.tensor([3, 7], dtype=torch.long)
    hidden = torch.randn(2, 1, cfg.hidden_size)
    tables = Qwen3_5TextRotaryEmbedding(config=cfg)(hidden, positions.unsqueeze(-1))
    cos, sin = tables[0][:, 0, :], tables[1][:, 0, :]  # [b, rope_dim] for the decode step

    q, gate, k, v = ref_prologue(hidden[:, 0, :], attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight,
                                 attn.q_norm.weight, attn.k_norm.weight, cos, sin, attn.head_dim, attn.q_norm.eps)

    qg = attn.q_proj(hidden).view(2, 1, -1, attn.head_dim * 2)
    q_hf, gate_hf = torch.chunk(qg, 2, dim=-1)
    q_hf = attn.q_norm(q_hf)
    k_hf = attn.k_norm(attn.k_proj(hidden).view(2, 1, -1, attn.head_dim))
    v_hf = attn.v_proj(hidden).view(2, 1, -1, attn.head_dim)
    q_hf, k_hf = apply_rotary_pos_emb(q_hf, k_hf, cos.unsqueeze(1), sin.unsqueeze(1))

    torch.testing.assert_close(q, q_hf[:, 0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k, k_hf[:, 0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gate, gate_hf[:, 0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(v, v_hf[:, 0], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3. CUDA kernel vs reference (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GPU_AVAILABLE, reason="non-CPU device required")
@pytest.mark.parametrize("head_dim,num_q_heads,num_kv_heads", [(64, 8, 2), (128, 16, 4), (256, 16, 4)])
def test_paged_decode_kernel_matches_reference(head_dim, num_q_heads, num_kv_heads):
    """Bare block-table kernel against the reference, at block boundaries."""
    from deepspeed.ops.module_inject import get_fused_glu_op

    torch.manual_seed(0)
    batch, block_size = 2, 64
    positions = torch.tensor([block_size - 1, 2 * block_size + 5], dtype=torch.long, device=DEV)
    length = int(positions.max()) + 1
    pool_k, pool_v, table = _paged_pool(batch,
                                        num_kv_heads,
                                        head_dim,
                                        length,
                                        block_size,
                                        device=DEV,
                                        dtype=torch.bfloat16)
    query = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device=DEV)

    out = torch.empty(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device=DEV)
    get_fused_glu_op().decode_attn_paged(query,
                                         pool_k.view(-1, num_kv_heads * head_dim),
                                         pool_v.view(-1, num_kv_heads * head_dim),
                                         table,
                                         positions,
                                         out,
                                         num_q_heads,
                                         num_kv_heads,
                                         head_dim,
                                         block_size)

    expected = ref_decode_attn_paged(query.float(), pool_k.float(), pool_v.float(), table, positions, block_size)
    torch.testing.assert_close(out.float(), expected, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="non-CPU device required")
def test_paged_decode_kernel_fused_matches_reference():
    """Fused paged kernel: q/k norms, partial RoPE, KV store, split-KV softmax, gate."""
    from deepspeed.ops.module_inject import get_fused_glu_op

    torch.manual_seed(0)
    batch, num_q_heads, num_kv_heads, head_dim, block_size = 2, 8, 2, 64, 64
    hidden_size, rope_dim = 64, 16
    positions = torch.tensor([block_size - 1, 3 * block_size], dtype=torch.long, device=DEV)
    length = int(positions.max()) + 1
    pool_k, pool_v, table = _paged_pool(batch,
                                        num_kv_heads,
                                        head_dim,
                                        length,
                                        block_size,
                                        device=DEV,
                                        dtype=torch.bfloat16)
    hidden = torch.randn(batch, 1, hidden_size, dtype=torch.bfloat16, device=DEV)
    q_w = torch.randn(num_q_heads * head_dim * 2, hidden_size, dtype=torch.float32, device=DEV).bfloat16()
    k_w = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=torch.float32, device=DEV).bfloat16()
    v_w = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=torch.float32, device=DEV).bfloat16()
    q_norm_w = torch.randn(head_dim, dtype=torch.float32, device=DEV)
    k_norm_w = torch.randn(head_dim, dtype=torch.float32, device=DEV)
    # full-length tables indexable by the GPU-resident write position
    cos_t = torch.randn(length, rope_dim, dtype=torch.float32, device=DEV)
    sin_t = torch.randn(length, rope_dim, dtype=torch.float32, device=DEV)

    q_flat = torch.nn.functional.linear(hidden[:, 0, :], q_w)
    gate = q_flat.view(batch, -1, head_dim * 2)[..., head_dim:]
    k_raw = torch.nn.functional.linear(hidden[:, 0, :], k_w).view(batch, num_kv_heads, head_dim)
    v_raw = torch.nn.functional.linear(hidden[:, 0, :], v_w).view(batch, num_kv_heads, head_dim)

    out = torch.empty(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device=DEV)
    get_fused_glu_op().decode_attn_paged(
        q_flat.view(batch, -1, head_dim * 2)[..., :head_dim].reshape(batch, num_q_heads, head_dim),
        pool_k.view(-1, num_kv_heads * head_dim),
        pool_v.view(-1, num_kv_heads * head_dim),
        table,
        positions,
        out,
        num_q_heads,
        num_kv_heads,
        head_dim,
        length,
        block_size,
        q_norm_w=1.0 + q_norm_w,
        cos_t=cos_t,
        sin_t=sin_t,
        gate=gate.reshape(batch, num_q_heads, head_dim),
        k_raw=k_raw.reshape(batch, num_kv_heads, head_dim),
        v_raw=v_raw.reshape(batch, num_kv_heads, head_dim),
        k_norm_w=1.0 + k_norm_w,
        eps=1e-6,
        rope_dim=rope_dim)

    # reference: prologue -> KV store through the page table -> attention -> gate
    ref_pool_k, ref_pool_v = pool_k.float().clone(), pool_v.float().clone()
    q_ref, gate_ref, k_ref, v_ref = ref_prologue(hidden[:, 0, :].float(), q_w.float(), k_w.float(), v_w.float(),
                                                 q_norm_w, k_norm_w, cos_t[positions], sin_t[positions], head_dim,
                                                 1e-6)
    for row in range(batch):
        write_paged_kv(ref_pool_k, ref_pool_v, table, int(positions[row]), block_size, k_ref[row], v_ref[row])
    attn = ref_decode_attn_paged(q_ref, ref_pool_k, ref_pool_v, table, positions, block_size)
    expected = attn * torch.sigmoid(gate_ref.float())

    torch.testing.assert_close(out.float(), expected, atol=5e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# 4. Rollout wiring (CPU)
# ---------------------------------------------------------------------------


class _PoolStub:
    """The cache surface ``_paged_decode_attn_forward`` reads."""

    is_paged_cache = True
    block_size = 8

    def __init__(self, num_slots, num_kv_heads, head_dim):
        self.block_ids = torch.zeros(1, 4, dtype=torch.int32)
        self._key_views = [torch.zeros(num_slots, num_kv_heads, head_dim)]
        self._value_views = [torch.zeros(num_slots, num_kv_heads, head_dim)]
        self._write_position = None
        self._record_past = False
        self.note_calls = 0

    def _kv_slot(self, layer_idx):
        return 0

    def note_step_written(self):
        self.note_calls += 1


class _OpStub:

    def __init__(self, paged=True):
        self.calls = []
        if paged:
            self.decode_attn_paged = self._record

    def _record(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _AttnStub:

    head_dim = 64
    layer_idx = 0

    def __init__(self, cache, op):
        self.q_proj = torch.nn.Linear(64, 8 * 64 * 2, bias=False).bfloat16()
        self.k_proj = torch.nn.Linear(64, 2 * 64, bias=False).bfloat16()
        self.v_proj = torch.nn.Linear(64, 2 * 64, bias=False).bfloat16()
        self.o_proj = torch.nn.Linear(8 * 64, 64, bias=False).bfloat16()
        self.q_norm = SimpleNamespace(weight=torch.zeros(64), eps=1e-6)
        self.k_norm = SimpleNamespace(weight=torch.zeros(64), eps=1e-6)
        self._ki_attn_op = op
        self._ki_qkv_op = None
        self._ki_num_q_heads = 8
        self._ki_num_kv_heads = 2
        self._ki_cos_t = torch.zeros(64, 16)
        self._ki_sin_t = torch.zeros(64, 16)
        self._ki_write_pos = cache._write_position
        self._ki_orig_calls = []
        self._ki_orig_attn_forward = self._orig_forward

    def _orig_forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None,
                      **kwargs):
        self._ki_orig_calls.append(hidden_states.shape)
        return torch.zeros_like(hidden_states), None


def _stub_attn(paged_op=True):
    cache = _PoolStub(num_slots=64, num_kv_heads=2, head_dim=64)
    cache._write_position = torch.zeros(1, dtype=torch.long)
    op = _OpStub(paged=paged_op)
    return cache, _AttnStub(cache, op), op


def test_paged_forward_calls_block_table_kernel():
    """A decoding step hands the kernel the cache's page table, pool views and positions."""
    cache, attn, op = _stub_attn()
    hidden = torch.randn(1, 1, 64, dtype=torch.bfloat16)

    out, _ = _paged_decode_attn_forward(attn, hidden, (None, None), {"full_attention": None}, cache)

    assert len(op.calls) == 1
    args, kwargs = op.calls[0]
    assert args[3] is cache.block_ids
    assert args[4] is cache._write_position
    assert args[9] == attn._ki_cos_t.shape[0]  # table length for the cos/sin bounds check
    assert args[10] == cache.block_size
    assert kwargs["k_raw"].shape == (1, 2, 64)
    assert kwargs["gate"].shape == (1, 8, 64)
    assert kwargs["rope_dim"] == 16
    assert cache.note_calls == 1
    assert not attn._ki_orig_calls
    assert out.shape == hidden.shape


def test_paged_forward_falls_back_without_the_paged_kernel():
    """A build without decode_attn_paged keeps the native paged forward."""
    cache, attn, _ = _stub_attn(paged_op=False)
    _paged_decode_attn_forward(attn, torch.randn(1, 1, 64, dtype=torch.bfloat16), (None, None), None, cache)
    assert attn._ki_orig_calls
    assert cache.note_calls == 0


def test_paged_forward_falls_back_outside_decode():
    """Prefill and recording windows stay on the native forward."""
    cache, attn, _ = _stub_attn()
    _paged_decode_attn_forward(attn, torch.randn(1, 3, 64, dtype=torch.bfloat16), (None, None), None, cache)
    cache._record_past = True
    _paged_decode_attn_forward(attn, torch.randn(1, 1, 64, dtype=torch.bfloat16), (None, None), None, cache)
    assert len(attn._ki_orig_calls) == 2
    assert cache.note_calls == 0

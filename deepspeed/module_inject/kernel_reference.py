# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Reference implementations (executable specifications) for segment-KI kernels.

Each function here defines the mathematically correct output for its
corresponding interface method (the m.def entries on the loaded extension).
Custom kernels (CUDA today, SYCL/XPU in the future) must produce results
equivalent to these functions within bf16 tolerance. Three roles in one:

1. **Specification**: an agent porting kernels to a new backend reads these
   to understand what each interface method must compute.
2. **Runtime fallback**: the forward paths in segment_ki call these when no
   custom kernel is available (e.g. CPU-only, XPU without compiled kernels).
3. **Test oracle**: unit tests compare custom kernel output against these
   functions on identical inputs.

All implementations use only standard PyTorch ops — no custom kernels — so
they are correct by construction and run on any device.
"""

import math

import torch
import torch.nn.functional as F


def grouped_linear(x, weights, glu_epilogue=False):
    """Unified projection: matmul against each weight, optional GLU epilogue.

    The portable interface's single entry point for all GEMV/GEMM paths
    (flat single-row, small-batch grouped, large-batch composite). Returns
    a list of per-weight outputs; the GLU variant folds silu(gate)*up into
    the first output.
    """
    outs = [F.linear(x, w) for w in weights]
    if glu_epilogue:
        gate = outs[0].float()
        up = outs[1].float()
        outs = [(gate / (1.0 + torch.exp(-gate)) * up).to(x.dtype)]
    return outs


def gdn_gates(a, b, a_log, dt_bias):
    """GDN gating: beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias)."""
    beta = torch.sigmoid(b)
    g = -torch.exp(a_log.float()) * F.softplus(a.float() + dt_bias.float())
    return beta, g.to(b.dtype)


def decode_attn(q, K, V, pos, kv_start=None):
    """Single-query attention over valid KV positions, batched.

    q: [b, num_heads, head_dim]
    K, V: [b, num_kv_heads, max_len, head_dim]
    pos: int or LongTensor [b] (valid positions, 0-indexed inclusive;
         an int applies to every row)
    kv_start: optional int or LongTensor [b], first valid position per row
         (left-padded prompts, or a row behind a moved decode front)

    Returns [b, num_heads, head_dim] with GQA (K/V heads shared across query
    heads).
    """
    num_q_heads = q.shape[-2]
    num_kv_heads = K.shape[-3]
    head_dim = q.shape[-1]
    n_rep = num_q_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    lens = (pos + 1).tolist() if torch.is_tensor(pos) else [pos + 1] * q.shape[0]
    if kv_start is None:
        starts = [0] * q.shape[0]
    else:
        starts = kv_start.tolist() if torch.is_tensor(kv_start) else [kv_start] * q.shape[0]
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for r in range(q.shape[0]):
        K_valid = K[r, :, starts[r]:lens[r], :]  # [nkv, len, hd]
        V_valid = V[r, :, starts[r]:lens[r], :]
        if n_rep > 1:
            K_valid = K_valid.repeat_interleave(n_rep, dim=0)
            V_valid = V_valid.repeat_interleave(n_rep, dim=0)
        scores = torch.einsum('hd,hld->hl', q[r].float(), K_valid.float()) * scale
        weights = torch.softmax(scores, dim=-1)
        out[r] = torch.einsum('hl,hld->hd', weights, V_valid.float())
    return out.to(q.dtype)


def paged_kv_slot(block_table, row, pos, block_size):
    """Page-table lookup: the block holding ``pos`` for ``row``.

    block_table: [rows, blocks_per_row] block ids for the pool.
    Returns the block id; the physical slot is block * block_size + pos % block_size.
    """
    return int(block_table[row, int(pos) // block_size])


def write_paged_kv(K_pool, V_pool, block_table, pos, block_size, k, v, row=0):
    """Store one token's post-norm/post-RoPE K/V into the block pool in place.

    K_pool, V_pool: [num_slots, num_kv_heads, head_dim]
    k, v: [num_kv_heads, head_dim] (or [1, num_kv_heads, head_dim]) for one row
    pos: the token's position within the row (int)
    row: the pool row whose page table resolves ``pos``
    Returns the physical slot written.
    """
    k = k.reshape(k.shape[-2], k.shape[-1])
    v = v.reshape(v.shape[-2], v.shape[-1])
    slot = paged_kv_slot(block_table, row, pos, block_size) * block_size + int(pos) % block_size
    K_pool[slot].copy_(k)
    V_pool[slot].copy_(v)
    return slot


def decode_attn_paged(q, K_pool, V_pool, block_table, pos, block_size, kv_start=None):
    """Paged single-query attention: the block-table sibling of ``decode_attn``.

    q: [b, num_heads, head_dim]
    K_pool, V_pool: [num_slots, num_kv_heads, head_dim] (slot = block * block_size + offset)
    block_table: [b, blocks_per_row] int; rows address the pool through it
    pos: int or LongTensor [b] (valid positions, 0-indexed inclusive)
    block_size: tokens per block

    Same math as ``decode_attn`` — a row's valid K/V are gathered through the
    page table instead of read from a dense plane — so both must agree when
    the pool holds the same values.
    """
    num_q_heads = q.shape[-2]
    num_kv_heads = K_pool.shape[-2]
    head_dim = q.shape[-1]
    n_rep = num_q_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    lens = (pos + 1).tolist() if torch.is_tensor(pos) else [pos + 1] * q.shape[0]
    if kv_start is None:
        starts = [0] * q.shape[0]
    else:
        starts = kv_start.tolist() if torch.is_tensor(kv_start) else [kv_start] * q.shape[0]
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for r in range(q.shape[0]):
        slots = [
            paged_kv_slot(block_table, r, i, block_size) * block_size + i % block_size
            for i in range(starts[r], lens[r])
        ]
        K_valid = K_pool[slots]  # [len, nkv, hd]
        V_valid = V_pool[slots]
        if n_rep > 1:
            K_valid = K_valid.repeat_interleave(n_rep, dim=1)
            V_valid = V_valid.repeat_interleave(n_rep, dim=1)
        scores = torch.einsum('hd,lhd->hl', q[r].float(), K_valid.float()) * scale
        weights = torch.softmax(scores, dim=-1)
        out[r] = torch.einsum('hl,lhd->hd', weights, V_valid.float())
    return out.to(q.dtype)


def qwen3_5_norm_rope(q_raw, k_raw, q_norm_w, k_norm_w, cos, sin, eps):
    """Per-head (1+w) RMSNorm plus partial rotate-half RoPE on projected q/k.

    q_raw/k_raw: [..., heads, head_dim] pre-norm projections.
    q_norm_w/k_norm_w: [head_dim] HF weights (the kernel multiplies by 1 + w).
    cos/sin: [..., rope_dim] tables at the rows' positions.
    Returns (q, k) post-norm/post-RoPE — the fused kernel's q-side/k-side
    contract, shared by the dense and paged references.
    """
    head_dim = q_raw.shape[-1]
    rope_dim = cos.shape[-1]

    def rms_norm(x, weight):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (xf * (1.0 + weight.float())).to(x.dtype)

    query = rms_norm(q_raw, q_norm_w)
    key = rms_norm(k_raw, k_norm_w)
    if rope_dim < head_dim:
        q_rot, q_pass = query[..., :rope_dim], query[..., rope_dim:]
        k_rot, k_pass = key[..., :rope_dim], key[..., rope_dim:]
    else:
        q_rot, k_rot = query, key
        q_pass = k_pass = None
    half = rope_dim // 2
    c1, c2 = cos[..., :half].unsqueeze(-2), cos[..., half:].unsqueeze(-2)
    s1, s2 = sin[..., :half].unsqueeze(-2), sin[..., half:].unsqueeze(-2)
    q1, q2 = q_rot[..., :half], q_rot[..., half:]
    k1, k2 = k_rot[..., :half], k_rot[..., half:]
    q_rot = torch.cat([q1 * c1 - q2 * s1, q2 * c2 + q1 * s2], dim=-1)
    k_rot = torch.cat([k1 * c1 - k2 * s1, k2 * c2 + k1 * s2], dim=-1)
    if q_pass is not None:
        query = torch.cat([q_rot, q_pass], dim=-1)
        key = torch.cat([k_rot, k_pass], dim=-1)
    else:
        query, key = q_rot, k_rot
    return query, key


def qwen3_5_fused_prologue(hidden, q_proj_w, k_proj_w, v_proj_w, q_norm_w, k_norm_w, cos, sin, head_dim, eps):
    """The fused decode_attn prologue: q|gate split, per-head q/k RMSNorm, partial RoPE.

    hidden: [b, hidden_size]
    q_proj_w: [2 * num_q_heads * head_dim, hidden_size] (query then gate per head)
    k_proj_w / v_proj_w: [num_kv_heads * head_dim, hidden_size]
    q_norm_w / k_norm_w: [head_dim] HF weights (the kernel uses 1 + w)
    cos, sin: [b, rope_dim] tables at the rows' positions
    Returns (q, gate, k, v) with q/gate [b, num_q_heads, head_dim] and k/v [b, num_kv_heads, head_dim],
    all post-norm and (q, k) post-RoPE — the kernel's fused-q/fused-k contract.
    """
    num_q_heads = q_proj_w.shape[0] // (2 * head_dim)
    num_kv_heads = k_proj_w.shape[0] // head_dim
    batch = hidden.shape[0]
    qg = F.linear(hidden, q_proj_w).view(batch, num_q_heads, head_dim * 2)
    query, gate = qg[..., :head_dim], qg[..., head_dim:]
    key = F.linear(hidden, k_proj_w).view(batch, num_kv_heads, head_dim)
    value = F.linear(hidden, v_proj_w).view(batch, num_kv_heads, head_dim)
    query, key = qwen3_5_norm_rope(query, key, q_norm_w, k_norm_w, cos, sin, eps)
    return query, gate, key, value


def silu_mul(gu, n):
    """GLU epilogue on a [M, 2n] buffer: silu(g) * u."""
    g = gu[:, :n].float()
    u = gu[:, n:].float()
    return (g / (1.0 + torch.exp(-g)) * u).to(gu.dtype)


def repeat_qk(q, k, ratio):
    """Head-repeat for the GDN scan's v-head layout: out[...,hv,:] = in[...,hv/ratio,:]."""
    return q.repeat_interleave(ratio, dim=-2), k.repeat_interleave(ratio, dim=-2)


def gated_norm(x, z, weight, eps=1e-5):
    """GDN z-gated output norm: (weight * rms(x)) * silu(z), weight direct."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * weight.float() * torch.nn.functional.silu(z.float())).to(x.dtype)


def fused_add_norm(hidden, residual, weight, eps=1e-5):
    """Residual add + RMSNorm: rms_norm(hidden + residual) * weight."""
    combined = hidden.float() + residual.float()
    rms = torch.rsqrt(combined.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (combined * rms * weight.float()).to(hidden.dtype)

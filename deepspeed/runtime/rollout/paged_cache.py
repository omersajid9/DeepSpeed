# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Paged KV cache and attention shim for the hybrid-engine rollout.

The cache keeps one block table per row over a shared ``BlockedKVCache`` pool.
Prefill still uses the dense static cache (the model's own attention); its KV is
copied into blocks once per prompt. Decode writes one token per row into the
block table and runs ``BlockedFlashAttn`` over the atoms, so it never builds the
dense per-step attention mask.

The attention shim is scoped to Qwen3's attention modules. It falls back to the
module's original forward whenever ``past_key_values`` is not a
``PagedAttentionCache``, which is what keeps the dense prefill path intact.
"""

from types import MethodType
from typing import Optional

import torch

from deepspeed.inference.v2.inference_utils import DtypeEnum
from deepspeed.inference.v2.kernels.ragged_ops.blocked_flash.blocked_flash import (
    BlockedFlashAttn,
    get_kv_block_size,
)
from deepspeed.inference.v2.ragged.kv_cache import BlockedKVCache, split_kv
from deepspeed.inference.v2.ragged.manager_configs import (
    AllocationMode,
    KVCacheConfig,
    MemoryConfig,
)


class _PagedLayer:
    """Dense-looking views of one layer's blocked cache for memory accounting."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys = keys
        self.values = values


class PagedAttentionCache:
    """Blocked KV cache with a device-resident block table per row.

    ``write`` takes post-RoPE K/V and a per-row write position. Blocks are
    allocated when a row first crosses a block boundary and released when the
    row retires (``compact``/``reset``). ``attention`` builds the atom list on
    the host, uploads it, and calls the blocked flash kernel.
    """

    def __init__(
        self,
        config,
        batch_size: int,
        max_cache_len: int,
        device,
        dtype,
        block_size: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("paged cache requires a positive batch size")
        if max_cache_len <= 0:
            raise ValueError("paged cache requires a positive capacity")

        text_config = getattr(config, "text_config", config)
        self.num_layers = int(getattr(text_config, "num_hidden_layers"))
        self.num_kv_heads = int(
            getattr(text_config, "num_key_value_heads", getattr(text_config, "num_attention_heads")))
        self.head_dim = int(
            getattr(text_config, "head_dim", None)
            or (getattr(text_config, "hidden_size") // getattr(text_config, "num_attention_heads")))
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_cache_len = int(max_cache_len)
        self.max_batch_size = int(batch_size)
        expected_block_size = get_kv_block_size(self.head_dim)
        if block_size is None:
            self.block_size = expected_block_size
        else:
            self.block_size = int(block_size)
            if self.block_size != expected_block_size:
                raise ValueError(
                    f"blocked attention requires block_size={expected_block_size} for head_dim={self.head_dim}, "
                    f"got {self.block_size}")
        self.max_blocks_per_row = (self.max_cache_len + self.block_size - 1) // self.block_size
        num_blocks = self.max_batch_size * self.max_blocks_per_row

        cache_config = KVCacheConfig(
            block_size=self.block_size,
            cache_shape=(self.num_layers, self.num_kv_heads, self.head_dim),
            cache_dtype=DtypeEnum(dtype),
        )
        memory_config = MemoryConfig(mode=AllocationMode.ALLOCATE, size=num_blocks)
        self.kv_cache = BlockedKVCache((cache_config, ), memory_config)
        self._key_views, self._value_views = [], []
        for layer_idx in range(self.num_layers):
            key, value = split_kv(self.kv_cache.get_cache(layer_idx))
            self._key_views.append(key)
            self._value_views.append(value)

        self.block_ids = torch.full(
            (self.max_batch_size, self.max_blocks_per_row),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.row_len = torch.zeros((self.max_batch_size, ), dtype=torch.int32, device=self.device)
        self._block_counts = [0] * self.max_batch_size
        self._write_positions: Optional[torch.Tensor] = None
        self._active_rows = torch.arange(self.max_batch_size, dtype=torch.long, device=self.device)
        self._kernel = BlockedFlashAttn(self.head_dim, DtypeEnum(dtype))
        self._atom_count = self.max_batch_size
        self._atoms_cpu = torch.zeros((self._atom_count, 8), dtype=torch.int32)
        self._atoms_device = torch.zeros((self._atom_count, 8), dtype=torch.int32, device=self.device)
        self._step_atoms: Optional[torch.Tensor] = None
        self._step_prepared = False
        self._scale = self.head_dim**-0.5

    @property
    def layers(self) -> list[_PagedLayer]:
        return [_PagedLayer(key, value) for key, value in zip(self._key_views, self._value_views)]

    @property
    def memory_bytes(self) -> int:
        storage = sum(key.numel() * key.element_size() + value.numel() * value.element_size()
                      for key, value in zip(self._key_views, self._value_views))
        metadata = self.block_ids.numel() * self.block_ids.element_size()
        metadata += self.row_len.numel() * self.row_len.element_size()
        return storage + metadata

    @property
    def blocks_in_use(self) -> int:
        return sum(self._block_counts)

    @property
    def layer_count(self) -> int:
        return self.num_layers

    def set_write_position(self, pos: torch.Tensor) -> None:
        if not isinstance(pos, torch.Tensor) or pos.dim() != 1:
            raise ValueError("paged cache write positions must be a 1-D tensor")
        if pos.numel() != self.max_batch_size:
            raise ValueError("paged cache write positions must cover every row")
        self._write_positions = pos
        self._step_prepared = False
        self._step_atoms = None

    def set_active_rows(self, rows) -> None:
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        if rows.dim() != 1:
            raise ValueError("active rows must be a 1-D tensor")
        self._active_rows = rows
        self._step_prepared = False
        self._step_atoms = None

    def _ensure_blocks(self, row: int, num_tokens: int) -> None:
        needed = (num_tokens + self.block_size - 1) // self.block_size
        if needed > self.max_blocks_per_row:
            raise ValueError(f"row {row} needs {needed} blocks, table holds {self.max_blocks_per_row}")
        count = self._block_counts[row]
        if needed <= count:
            return
        new_blocks = self.kv_cache.reserve(needed - count).to(self.device, dtype=torch.int32)
        self.block_ids[row, count:needed].copy_(new_blocks)
        self._block_counts[row] = needed

    def _free_row(self, row: int) -> None:
        count = self._block_counts[row]
        if count:
            blocks = self.block_ids[row, :count].tolist()
            self.kv_cache.free(blocks)
            self.block_ids[row].fill_(-1)
            self._block_counts[row] = 0
            self.row_len[row] = 0

    def _slots(self, rows: torch.Tensor, positions: torch.Tensor, q_len: int) -> torch.Tensor:
        """Return ``[rows, q_len]`` physical slot ids for the block-table rows."""
        positions = positions[:, None] + torch.arange(q_len, device=self.device)[None, :]
        block_indices = positions // self.block_size
        blocks = self.block_ids[rows].gather(1, block_indices)
        return blocks * self.block_size + positions % self.block_size

    def copy_from_dense(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        target_row: int,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Copy one dense prefill row into a clean block-table row.

        ``keys``/``values`` are ``[heads, prompt_len, head_dim]``. A left-padded
        prompt passes ``valid_mask``; only its real tokens are copied, so the
        paged row starts at column zero and decode positions stay compact.
        """
        if valid_mask is not None:
            keys = keys[:, valid_mask]
            values = values[:, valid_mask]
        prompt_len = keys.shape[1]
        self._free_row(target_row)
        self._ensure_blocks(target_row, prompt_len)
        positions = torch.arange(prompt_len, device=self.device)
        slots = self._slots(torch.tensor([target_row], device=self.device), positions[:1], prompt_len)[0]
        key_view = self._key_views[layer_idx].view(-1, self.num_kv_heads, self.head_dim)
        value_view = self._value_views[layer_idx].view(-1, self.num_kv_heads, self.head_dim)
        key_view[slots] = keys.transpose(0, 1)
        value_view[slots] = values.transpose(0, 1)
        self.row_len[target_row] = prompt_len
        if self._write_positions is not None:
            self._write_positions[target_row] = prompt_len

    def _prepare_step(self, q_len: int) -> None:
        if self._step_prepared:
            return
        if self._write_positions is None:
            raise RuntimeError("paged cache write position was not set")
        rows = self._active_rows
        positions = self._write_positions[rows]
        for local_row, row in enumerate(rows.tolist()):
            self._ensure_blocks(row, int(positions[local_row]) + q_len)
        self._step_prepared = True
        self._step_atoms = None

    def write(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Write post-RoPE K/V for the active rows at their current positions."""
        if self._write_positions is None:
            raise RuntimeError("paged cache write position was not set")
        rows = self._active_rows
        q_len = key_states.shape[2]
        self._prepare_step(q_len)
        positions = self._write_positions[rows]
        slots = self._slots(rows, positions, q_len)
        key_view = self._key_views[layer_idx].view(-1, self.num_kv_heads, self.head_dim)
        value_view = self._value_views[layer_idx].view(-1, self.num_kv_heads, self.head_dim)
        key_view[slots.reshape(-1)] = key_states.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
        value_view[slots.reshape(-1)] = value_states.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
        self.row_len[rows] = (positions + q_len).to(torch.int32)

    def _build_atoms(self) -> torch.Tensor:
        if self._step_atoms is not None:
            return self._step_atoms
        rows = self._active_rows
        positions = self._write_positions[rows]
        num_atoms = rows.numel()
        atoms = self._atoms_cpu[:num_atoms]
        pointers = torch.tensor([self.block_ids[int(row)].data_ptr() for row in rows.tolist()], dtype=torch.int64)
        atoms.view(torch.int64)[:, 0].copy_(pointers)
        # AttentionAtom: pointer (8 bytes), q_start, q_len, kv_blocks, total_extent, global_q_idx, unused.
        atoms[:, 2] = torch.arange(num_atoms, dtype=torch.int32)
        atoms[:, 3] = 1
        atoms[:, 4] = ((positions + 1 + self.block_size - 1) // self.block_size).to(torch.int32)
        atoms[:, 5] = (positions + 1).to(torch.int32)
        atoms[:, 6] = positions.to(torch.int32)
        atoms[:, 7] = 0
        self._atoms_device[:num_atoms].copy_(atoms, non_blocking=True)
        self._step_atoms = self._atoms_device[:num_atoms]
        return self._step_atoms

    def attention(self, layer_idx: int, query_states: torch.Tensor, scale: float) -> torch.Tensor:
        """Run blocked flash attention for one decode step.

        ``query_states`` is ``[rows, heads, 1, head_dim]`` for the active rows;
        the return has the same shape.
        """
        num_rows, num_heads, q_len, head_dim = query_states.shape
        if q_len != 1:
            raise ValueError("paged decode attention expects one query token per row")
        atoms = self._build_atoms()
        output = torch.empty((num_rows, num_heads * head_dim), dtype=query_states.dtype, device=query_states.device)
        query = query_states.transpose(1, 2).reshape(num_rows, num_heads * head_dim)
        self._kernel(output, query, self._key_views[layer_idx], self._value_views[layer_idx], atoms, scale)
        return output.view(num_rows, 1, num_heads, head_dim).transpose(1, 2)

    def materialize(self, layer_idx: int, row: int, length: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather a row's blocks into dense ``[heads, length, head_dim]`` K/V for tests."""
        if length is None:
            length = int(self.row_len[row])
        positions = torch.arange(length, device=self.device)
        slots = self._slots(torch.tensor([row], device=self.device), positions[:1], length)[0]
        keys = self._key_views[layer_idx].view(-1, self.num_kv_heads, self.head_dim)[slots]
        values = self._value_views[layer_idx].view(-1, self.num_kv_heads, self.head_dim)[slots]
        return keys.transpose(0, 1), values.transpose(0, 1)

    def compact(self, active_indices: torch.Tensor) -> None:
        """Compact the block table after rows retire; no KV is copied."""
        keep = active_indices.to(device=self.device, dtype=torch.long)
        if keep.dim() != 1:
            raise ValueError("compact indices must be 1-D")
        if keep.numel() and (keep.min() < 0 or keep.max() >= self.max_batch_size):
            raise ValueError("compact indices out of range")
        if keep.unique().numel() != keep.numel():
            raise ValueError("compact indices must not contain duplicates")
        kept = set(keep.tolist())
        for row in range(self.max_batch_size):
            if self._block_counts[row] and row not in kept:
                self._free_row(row)
        count = keep.numel()
        if count:
            self.block_ids[:count].copy_(self.block_ids[keep].clone())
            self.row_len[:count].copy_(self.row_len[keep].clone())
            self._block_counts[:count] = [self._block_counts[row] for row in keep.tolist()]
        if count < self.max_batch_size:
            self.block_ids[count:].fill_(-1)
            self.row_len[count:].zero_()
            self._block_counts[count:] = [0] * (self.max_batch_size - count)
        if self._write_positions is not None:
            positions = self._write_positions
            compacted = positions[keep].clone() if count else positions[:0]
            positions[:count].copy_(compacted)
            positions[count:].fill_(-1)

    def reset(self) -> None:
        for row in range(self.max_batch_size):
            self._free_row(row)
        self.row_len.zero_()

    def trim_left(self, count: int) -> None:
        """Release a block-aligned dead prefix from every row."""
        if count < 0:
            raise ValueError("trim count must be non-negative")
        if count % self.block_size:
            raise ValueError("paged cache trimming must be block aligned")
        blocks = count // self.block_size
        if blocks == 0:
            return
        for row in range(self.max_batch_size):
            row_blocks = self._block_counts[row]
            if row_blocks <= blocks:
                self._free_row(row)
                continue
            released = self.block_ids[row, :blocks].tolist()
            self.kv_cache.free(released)
            remaining = row_blocks - blocks
            self.block_ids[row, :remaining].copy_(self.block_ids[row, blocks:row_blocks].clone())
            self.block_ids[row, remaining:].fill_(-1)
            self._block_counts[row] = remaining
            self.row_len[row] = torch.clamp(self.row_len[row] - count, min=0)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if self._write_positions is None:
            return 0
        return int(self._write_positions.max().item())

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.max_cache_len

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        return self.max_cache_len, 0

    def __len__(self) -> int:
        return self.num_layers


def _paged_qwen3_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):
    original_forward = self._paged_original_forward
    if not isinstance(past_key_values, PagedAttentionCache):
        return original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
    if self.sliding_window is not None:
        raise RuntimeError("paged attention does not support sliding-window Qwen3 layers yet")

    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    past_key_values.write(self.layer_idx, key_states, value_states)
    attn_output = past_key_values.attention(self.layer_idx, query_states, self.scaling)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def install_paged_attention(module) -> list:
    """Patch Qwen3 attention modules to use paged attention when handed a paged cache."""
    patched = []
    for submodule in module.modules():
        if type(submodule).__name__ != "Qwen3Attention":
            continue
        if getattr(submodule, "_paged_original_forward", None) is not None:
            continue
        submodule._paged_original_forward = submodule.forward
        submodule.forward = MethodType(_paged_qwen3_forward, submodule)
        patched.append(submodule)
    if not patched:
        raise ValueError("paged cache requires a Qwen3Attention model")
    return patched

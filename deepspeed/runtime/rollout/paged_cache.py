# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Paged KV cache and attention shim for the hybrid-engine rollout.

The cache keeps one block table per row over a shared ``BlockedKVCache`` pool.
Prefill can write directly into the table in chunk windows (no dense staging or
copy), and decode writes one token per row and runs ``BlockedFlashAttn`` over the
atoms, so neither path builds the dense attention mask. Full prompt-prefix blocks
can be shared between rows with per-block refcounts.

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
    get_q_block_size,
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
        use_flashinfer_prefill: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("paged cache requires a positive batch size")
        if max_cache_len <= 0:
            raise ValueError("paged cache requires a positive capacity")

        text_config = getattr(config, "text_config", config)
        self.num_layers = int(getattr(text_config, "num_hidden_layers"))
        self.num_kv_heads = int(
            getattr(text_config, "num_key_value_heads", getattr(text_config, "num_attention_heads")))
        self.num_q_heads = int(getattr(text_config, "num_attention_heads"))
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
        self.q_block_size = get_q_block_size(self.head_dim)
        self._atom_count = self.max_batch_size * ((max_cache_len + self.q_block_size - 1) // self.q_block_size)
        self._atoms_cpu = torch.zeros((self._atom_count, 8), dtype=torch.int32)
        self._atoms_device = torch.zeros((self._atom_count, 8), dtype=torch.int32, device=self.device)
        self._step_atoms: Optional[torch.Tensor] = None
        self._step_q_len = 1
        self._step_prepared = False
        self._scale = self.head_dim**-0.5
        self._graph_safe = False
        self._graph_mode = False
        self._graph_width = 0
        self._graph_active_rows: Optional[torch.Tensor] = None
        self._graph_atoms: Optional[torch.Tensor] = None
        self._graph_outputs: Optional[torch.Tensor] = None
        self._graph_row_buffers: dict = {}
        self._graph_atom_views: dict = {}
        # Block sharing for prompt-prefix reuse. A block can be referenced by several
        # rows; it returns to the allocator only when the last row releases it.
        self._block_refs: dict[int, int] = {}
        self._block_hash: dict[bytes, int] = {}
        self._block_hash_by_id: dict[int, bytes] = {}
        self.prefix_reuse_blocks = 0
        self.prefix_reuse_tokens = 0
        self.use_flashinfer_prefill = use_flashinfer_prefill
        self._flashinfer_prefill_wrapper = None
        self._flashinfer_workspace = None

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
        self._step_q_len = 1

    def set_active_rows(self, rows) -> None:
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        if rows.dim() != 1:
            raise ValueError("active rows must be a 1-D tensor")
        self._active_rows = rows
        self._step_prepared = False
        self._step_atoms = None

    def enable_graph_safe(self) -> None:
        """Preallocate the whole block table so replay never allocates or reads host state.

        A graph replay executes only device work. The paged decode path therefore cannot
        reserve blocks or walk host lists per step; every row owns its full block span from
        ``reset`` to ``reset`` and the atom fields are updated from device tensors.
        """
        if self._graph_safe:
            return
        self._graph_safe = True
        self._allocate_all_blocks()
        self._zero_storage()
        self._init_graph_atom_structure()
        if self._graph_outputs is None:
            self._graph_outputs = torch.empty((self.num_layers, self.max_batch_size, self.num_q_heads * self.head_dim),
                                              dtype=self.dtype,
                                              device=self.device)

    def set_eager_mode(self) -> None:
        """Return to eager writes/attention after a decode graph has run."""
        self._graph_mode = False
        self._graph_width = 0
        self._graph_atoms = None
        self._step_prepared = False
        self._step_atoms = None
        self._step_q_len = 1

    def set_graph_mode(self, width: int) -> None:
        """Address exactly ``width`` rows from device-resident buffers for a replay step."""
        if not self._graph_safe:
            raise RuntimeError("paged cache must be graph safe before graph replay")
        if width <= 0 or width > self.max_batch_size:
            raise ValueError(f"graph width {width} is outside [1, {self.max_batch_size}]")
        # A captured graph embeds the addresses of the tensors it indexes. Keep one
        # row/atom tensor per width instead of recreating them on every replay, or the
        # captured index kernels would dereference freed memory after the first step.
        self._graph_mode = True
        self._graph_width = width
        rows = self._graph_row_buffers.get(width)
        if rows is None:
            rows = torch.arange(width, dtype=torch.long, device=self.device)
            self._graph_row_buffers[width] = rows
        atoms = self._graph_atom_views.get(width)
        if atoms is None:
            atoms = self._atoms_device[:width]
            self._graph_atom_views[width] = atoms
        self._graph_active_rows = rows
        self._graph_atoms = atoms
        # Eager prefill may have overwritten the shared atom buffer with multi-token
        # atoms. Restore the decode atom structure before every capture/replay.
        self._init_graph_atom_structure()
        self._graph_atoms = self._atoms_device[:width]
        self._active_rows = rows
        self._step_prepared = True
        self._step_atoms = None
        self._step_q_len = 1

    def _allocate_all_blocks(self) -> None:
        for row in range(self.max_batch_size):
            self._ensure_blocks(row, self.max_cache_len)

    def _zero_storage(self) -> None:
        for key, value in zip(self._key_views, self._value_views):
            key.zero_()
            value.zero_()

    def update_graph_atoms(self) -> None:
        """Refresh the length fields of the static atom table outside the captured graph."""
        if not self._graph_mode or self._graph_atoms is None:
            return
        positions = self._write_positions[:self._graph_width]
        atoms = self._graph_atoms
        atoms[:, 4] = ((positions + 1 + self.block_size - 1) // self.block_size).to(torch.int32)
        atoms[:, 5] = (positions + 1).to(torch.int32)
        atoms[:, 6] = positions.to(torch.int32)

    def _init_graph_atom_structure(self) -> None:
        """Fill the atom fields that do not change across replay steps.

        The block-table pointer is fixed for the cache's lifetime; the length fields are
        recomputed on device in ``attention`` from ``_write_positions``.
        """
        atoms = self._atoms_cpu
        pointers = torch.tensor([self.block_ids[row].data_ptr() for row in range(self.max_batch_size)],
                                dtype=torch.int64)
        atoms.view(torch.int64)[:self.max_batch_size, 0].copy_(pointers)
        atoms[:, 2] = torch.arange(self._atom_count, dtype=torch.int32)
        atoms[:, 3] = 1
        atoms[:, 4:7] = 0
        atoms[:, 7] = 0
        self._atoms_device.copy_(atoms)

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

    def clear_row(self, row: int) -> None:
        """Release a row's block chain before it is reused for a new prompt."""
        self._free_row(row)

    def _release_blocks(self, blocks) -> None:
        for block in blocks:
            refs = self._block_refs.get(block)
            if refs is None or refs <= 1:
                if refs is not None:
                    self._block_refs.pop(block, None)
                block_hash = self._block_hash_by_id.pop(block, None)
                if block_hash is not None:
                    self._block_hash.pop(block_hash, None)
                self.kv_cache.free([block])
            else:
                self._block_refs[block] = refs - 1

    def _free_row(self, row: int) -> None:
        count = self._block_counts[row]
        if count:
            self._release_blocks(self.block_ids[row, :count].tolist())
            self.block_ids[row].fill_(-1)
            self._block_counts[row] = 0
            self.row_len[row] = 0

    def prepare_prefix(self, hashes, target_row: int) -> int:
        """Point ``target_row`` at an existing block chain and return its block count.

        Only full blocks are reused. The caller prefills the remaining suffix; shared
        blocks keep their refcount so retiring either row cannot free KV that the other
        row still reads.
        """
        if self._graph_safe:
            raise RuntimeError("prefix reuse is not compatible with a graph-safe cache")
        self._free_row(target_row)
        reused = 0
        for block_hash in hashes:
            block = self._block_hash.get(block_hash)
            if block is None:
                break
            self.block_ids[target_row, reused] = block
            self._block_refs[block] = self._block_refs.get(block, 1) + 1
            reused += 1
        self._block_counts[target_row] = reused
        self.row_len[target_row] = reused * self.block_size
        return reused

    def register_prefix(self, hashes, target_row: int, count: int) -> None:
        """Record the target row's full blocks as reusable prefix blocks."""
        if self._graph_safe:
            return
        for index, block_hash in enumerate(hashes[:count]):
            if block_hash in self._block_hash:
                continue
            block = int(self.block_ids[target_row, index])
            self._block_hash[block_hash] = block
            self._block_hash_by_id[block] = block_hash
            self._block_refs.setdefault(block, 1)

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
        if not self._graph_safe:
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
        if not self._graph_mode:
            self._prepare_step(q_len)
        elif q_len != 1:
            raise ValueError("graph replay paged writes expect one token per row")
        positions = self._write_positions[rows]
        slots = self._slots(rows, positions, q_len)
        key_view = self._key_views[layer_idx].view(-1, self.num_kv_heads, self.head_dim)
        value_view = self._value_views[layer_idx].view(-1, self.num_kv_heads, self.head_dim)
        key_view[slots.reshape(-1)] = key_states.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
        value_view[slots.reshape(-1)] = value_states.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
        self.row_len[rows] = (positions + q_len).to(torch.int32)

    def _build_atoms(self, q_len: int = 1) -> torch.Tensor:
        if self._step_atoms is not None and self._step_q_len == q_len:
            return self._step_atoms
        rows = self._active_rows
        positions = self._write_positions[rows]
        num_rows = rows.numel()
        if q_len == 1:
            atoms = self._atoms_cpu[:num_rows]
            pointers = torch.tensor([self.block_ids[int(row)].data_ptr() for row in rows.tolist()], dtype=torch.int64)
            atoms.view(torch.int64)[:, 0].copy_(pointers)
            # AttentionAtom: pointer (8 bytes), q_start, q_len, kv_blocks, total_extent, global_q_idx, unused.
            atoms[:, 2] = torch.arange(num_rows, dtype=torch.int32)
            atoms[:, 3] = 1
            atoms[:, 4] = ((positions + 1 + self.block_size - 1) // self.block_size).to(torch.int32)
            atoms[:, 5] = (positions + 1).to(torch.int32)
            atoms[:, 6] = positions.to(torch.int32)
            atoms[:, 7] = 0
            num_atoms = num_rows
        else:
            # The kernel consumes one atom per query block; a prefilling row wider than
            # q_block_size must be split or the kernel reads past the atom's query block.
            q_blocks_per_row = (q_len + self.q_block_size - 1) // self.q_block_size
            if num_rows * q_blocks_per_row > self._atom_count:
                raise ValueError("query block count exceeds the paged atom table")
            atom_index = torch.arange(num_rows * q_blocks_per_row, device=positions.device)
            row_index = atom_index // q_blocks_per_row
            local_block = atom_index % q_blocks_per_row
            local_start = local_block * self.q_block_size
            local_len = torch.clamp(q_len - local_start, max=self.q_block_size)
            atom_row = rows[row_index]
            atom_start = positions[row_index] + local_start
            atoms = self._atoms_cpu[:atom_index.numel()]
            pointers = torch.tensor([self.block_ids[int(row)].data_ptr() for row in atom_row.tolist()],
                                    dtype=torch.int64)
            atoms.view(torch.int64)[:, 0].copy_(pointers)
            atoms[:, 2] = (row_index * q_len + local_start).to(torch.int32).cpu()
            atoms[:, 3] = local_len.to(torch.int32).cpu()
            atoms[:, 4] = ((atom_start + local_len + self.block_size - 1) // self.block_size).to(torch.int32).cpu()
            atoms[:, 5] = (atom_start + local_len).to(torch.int32).cpu()
            atoms[:, 6] = atom_start.to(torch.int32).cpu()
            atoms[:, 7] = 0
            num_atoms = atom_index.numel()
        self._atoms_device[:num_atoms].copy_(atoms, non_blocking=True)
        self._step_atoms = self._atoms_device[:num_atoms]
        self._step_q_len = q_len
        return self._step_atoms

    def _flashinfer_prefill_attention(self, layer_idx: int, query_states: torch.Tensor, scale: float) -> torch.Tensor:
        """Prefill a single prompt row through FlashInfer's paged-KV backend."""
        if query_states.shape[0] != 1:
            raise ValueError("FlashInfer prefill currently supports one active row")
        if self._graph_mode:
            raise RuntimeError("FlashInfer prefill cannot run inside a graph replay")

        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        if self._flashinfer_prefill_wrapper is None:
            self._flashinfer_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            self._flashinfer_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(self._flashinfer_workspace,
                                                                                   "NHD",
                                                                                   use_cuda_graph=False)

        row = int(self._active_rows[0])
        total_len = int(self.row_len[row])
        block_count = (total_len + self.block_size - 1) // self.block_size
        last_page_len = total_len - (block_count - 1) * self.block_size
        if last_page_len == 0:
            last_page_len = self.block_size
        indices = self.block_ids[row, :block_count].to(torch.int32)
        indptr = torch.tensor([0, block_count], dtype=torch.int32, device=self.device)
        last_page = torch.tensor([last_page_len], dtype=torch.int32, device=self.device)
        q = query_states.transpose(1, 2).reshape(-1, query_states.shape[1], query_states.shape[3])
        qo_indptr = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=self.device)
        wrapper = self._flashinfer_prefill_wrapper
        wrapper.plan(qo_indptr,
                     indptr,
                     indices,
                     last_page,
                     query_states.shape[1],
                     self.num_kv_heads,
                     self.head_dim,
                     self.block_size,
                     causal=True,
                     sm_scale=scale,
                     q_data_type=query_states.dtype,
                     kv_data_type=query_states.dtype)
        output = wrapper.run(q, (self._key_views[layer_idx], self._value_views[layer_idx]))
        return output.view(1, q.shape[0], query_states.shape[1], query_states.shape[3]).transpose(1, 2)

    def attention(self, layer_idx: int, query_states: torch.Tensor, scale: float) -> torch.Tensor:
        """Run blocked flash attention for one decode step.

        ``query_states`` is ``[rows, heads, 1, head_dim]`` for the active rows;
        the return has the same shape. Graph replay derives the atom lengths on device and
        writes into a preallocated output, so the replay performs no host work.
        """
        num_rows, num_heads, q_len, head_dim = query_states.shape
        if q_len > 1 and self.use_flashinfer_prefill:
            return self._flashinfer_prefill_attention(layer_idx, query_states, scale)
        if self._graph_mode and q_len != 1:
            raise ValueError("graph replay paged attention expects one query token per row")
        if self._graph_mode:
            if self._graph_outputs is None:
                raise RuntimeError("graph-safe paged cache has no output buffers")
            if num_rows != self._graph_width:
                raise ValueError(f"graph replay row count {num_rows} does not match width {self._graph_width}")
            atoms = self._graph_atoms
            output = self._graph_outputs[layer_idx][:num_rows]
        else:
            atoms = self._build_atoms(q_len)
            output = torch.empty((num_rows * q_len, num_heads * head_dim),
                                 dtype=query_states.dtype,
                                 device=query_states.device)
        query = query_states.transpose(1, 2).reshape(num_rows * q_len, num_heads * head_dim)
        self._kernel(output, query, self._key_views[layer_idx], self._value_views[layer_idx], atoms, scale)
        return output.view(num_rows, q_len, num_heads, head_dim).transpose(1, 2)

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
        self._block_refs.clear()
        self._block_hash.clear()
        self._block_hash_by_id.clear()
        if self._graph_safe:
            self._allocate_all_blocks()
            self._zero_storage()
            self._init_graph_atom_structure()
            self._graph_mode = False

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
            self._release_blocks(released)
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
    # attention() returns [rows, query, heads, head_dim]; the original forward gets
    # [rows, query, hidden] from SDPA, so transpose before flattening the head layout.
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
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

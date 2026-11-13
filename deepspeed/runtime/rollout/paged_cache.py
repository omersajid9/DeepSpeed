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
        use_flashinfer_decode: bool = False,
        num_kv_layers: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("paged cache requires a positive batch size")
        if max_cache_len <= 0:
            raise ValueError("paged cache requires a positive capacity")

        text_config = getattr(config, "text_config", config)
        self.num_layers = int(getattr(text_config, "num_hidden_layers"))
        # Hybrid models carry KV only on their full-attention layers; the KV
        # pool is sized for those and the layer table maps HF layer indices to
        # slots (identity here, remapped by the hybrid subclass).
        self.num_kv_layers = self.num_layers if num_kv_layers is None else int(num_kv_layers)
        if not 1 <= self.num_kv_layers <= self.num_layers:
            raise ValueError(f"num_kv_layers {self.num_kv_layers} outside [1, {self.num_layers}]")
        self._layer_to_slot = list(range(self.num_kv_layers)) if self.num_kv_layers == self.num_layers \
            else [None] * self.num_layers
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
            cache_shape=(self.num_kv_layers, self.num_kv_heads, self.head_dim),
            cache_dtype=DtypeEnum(dtype),
        )
        memory_config = MemoryConfig(mode=AllocationMode.ALLOCATE, size=num_blocks)
        self.kv_cache = BlockedKVCache((cache_config, ), memory_config)
        self._key_views, self._value_views = [], []
        for layer_idx in range(self.num_kv_layers):
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
        self.use_flashinfer_decode = use_flashinfer_decode
        self._flashinfer_decode_wrapper = None
        self._flashinfer_decode_workspace = None
        self._fi_decode_indptr_host = None
        self._fi_decode_last_page_len_host = None
        self._fi_decode_q = None
        self._fi_step_planned = False

    @property
    def layers(self) -> list[_PagedLayer]:
        return [_PagedLayer(key, value) for key, value in zip(self._key_views, self._value_views)]

    def _kv_slot(self, layer_idx: int) -> int:
        """Map an HF layer index to this cache's KV slot.

        Pure-attention models map identity; hybrid caches raise for their
        linear-attention layers, which keep no KV."""
        slot = self._layer_to_slot[layer_idx]
        if slot is None:
            raise ValueError(f"layer {layer_idx} is a linear-attention layer and holds no paged KV")
        return slot

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
        self._fi_step_planned = False
        self._step_q_len = 1

    def set_active_rows(self, rows) -> None:
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        if rows.dim() != 1:
            raise ValueError("active rows must be a 1-D tensor")
        self._active_rows = rows
        self._step_prepared = False
        self._step_atoms = None
        self._fi_step_planned = False

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
            self._graph_outputs = torch.empty(
                (self.num_kv_layers, self.max_batch_size, self.num_q_heads * self.head_dim),
                dtype=self.dtype,
                device=self.device)
        if self.use_flashinfer_decode:
            self._init_flashinfer_decode_graph()

    def set_eager_mode(self) -> None:
        """Return to eager writes/attention after a decode graph has run."""
        self._graph_mode = False
        self._graph_width = 0
        self._graph_atoms = None
        self._step_prepared = False
        self._step_atoms = None
        self._step_q_len = 1
        self._fi_step_planned = False

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
        # atoms. Restore the decode atom structure before every capture/replay. With
        # FlashInfer decode the replayed decode steps never read the atom table (the
        # multi-token paths rebuild their own atoms), so the rebuild is skipped.
        if not (self.use_flashinfer_decode and self._graph_safe):
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

    def _init_flashinfer_decode_graph(self) -> None:
        """Reserve the fixed buffers the FlashInfer decode plan reads at replay time.

        FlashInfer's cudagraph mode copies the page-table metadata passed to ``plan``
        into these fixed device buffers before every replay, so the captured kernels
        see each step's table. The table is ragged: a row claims exactly the blocks
        holding its valid tokens, because the paged kernel reads whole pages. The
        index buffer is separate from the live block table, whose rows stay in place
        while the ragged prefixes are concatenated here.
        """
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        if self._flashinfer_decode_wrapper is not None:
            return
        self._flashinfer_decode_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        max_pages = self.max_batch_size * self.max_blocks_per_row
        self._fi_decode_indptr_host = torch.zeros(self.max_batch_size + 1, dtype=torch.int32, pin_memory=True)
        self._fi_decode_last_page_len_host = torch.ones(self.max_batch_size, dtype=torch.int32, pin_memory=True)
        self._fi_decode_indices_host = torch.zeros(max_pages, dtype=torch.int32, pin_memory=True)
        self._flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self._flashinfer_decode_workspace,
            "NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            backend="fa2",
            paged_kv_indptr_buffer=torch.zeros(self.max_batch_size + 1, dtype=torch.int32, device=self.device),
            paged_kv_indices_buffer=torch.zeros(max_pages, dtype=torch.int32, device=self.device),
            paged_kv_last_page_len_buffer=torch.ones(self.max_batch_size, dtype=torch.int32, device=self.device))
        # Padded rows attend through this query; their outputs are never read, so any
        # finite values keep the fixed-batch kernel busy without touching live state.
        self._fi_decode_q = torch.zeros(self.max_batch_size,
                                        self.num_q_heads,
                                        self.head_dim,
                                        dtype=self.dtype,
                                        device=self.device)

    def _get_eager_flashinfer_decode_wrapper(self):
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        if self._flashinfer_decode_wrapper is None:
            workspace = self._flashinfer_workspace
            if workspace is None:
                workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
                self._flashinfer_workspace = workspace
            self._flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace,
                                                                                 "NHD",
                                                                                 use_cuda_graph=False,
                                                                                 use_tensor_cores=True,
                                                                                 backend="fa2")
        return self._flashinfer_decode_wrapper

    @staticmethod
    def _last_page_len(position: int, block_size: int) -> int:
        total = int(position) + 1
        full = (total + block_size - 1) // block_size
        return total - (full - 1) * block_size

    def plan_flashinfer_decode(self, positions_by_row) -> None:
        """Plan the FlashInfer decode step from host write positions.

        ``positions_by_row`` maps the rows that decode this step to the column of the
        token just written. The page table is ragged: a row claims exactly the blocks
        holding its valid tokens, because the paged kernel reads whole pages and would
        fault on blocks the row does not own. Rows that do not decode claim one valid
        block so the fixed-batch kernel has finite work for them; their outputs are
        never read. The graph-safe path plans from a small host copy of the block
        table and only issues pinned host-to-device copies before the replay.
        """
        block_size = self.block_size
        if self._graph_mode:
            if self._flashinfer_decode_wrapper is None:
                raise RuntimeError("FlashInfer decode plan needs a graph-safe cache")
            table_host = self.block_ids.cpu()
            indptr_host = self._fi_decode_indptr_host
            last_page_len_host = self._fi_decode_last_page_len_host
            indices_host = self._fi_decode_indices_host
            offset = 0
            for row in range(self.max_batch_size):
                position = positions_by_row.get(row)
                if position is None:
                    indptr_host[row + 1] = offset + 1
                    last_page_len_host[row] = 1
                    indices_host[offset] = table_host[row, 0]
                    offset += 1
                    continue
                total = int(position) + 1
                pages = (total + block_size - 1) // block_size
                indptr_host[row + 1] = offset + pages
                last_page_len_host[row] = self._last_page_len(total - 1, block_size)
                indices_host[offset:offset + pages].copy_(table_host[row, :pages])
                offset += pages
            self._flashinfer_decode_wrapper.plan(
                indptr_host,
                indices_host[:offset],
                last_page_len_host,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
                pos_encoding_mode="NONE",
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                sm_scale=self._scale,
            )
        else:
            rows = sorted(positions_by_row)
            table_host = self.block_ids[torch.tensor(rows, dtype=torch.long, device=self.device)].cpu()
            totals = [int(positions_by_row[row]) + 1 for row in rows]
            pages_by_row = [(total + block_size - 1) // block_size for total in totals]
            total_pages = sum(pages_by_row)
            indptr_host = torch.zeros(len(rows) + 1, dtype=torch.int32)
            last_page_len_host = torch.empty(len(rows), dtype=torch.int32)
            indices_host = torch.empty(total_pages, dtype=torch.int32)
            offset = 0
            for local, row in enumerate(rows):
                pages = pages_by_row[local]
                indptr_host[local + 1] = offset + pages
                last_page_len_host[local] = self._last_page_len(totals[local] - 1, block_size)
                indices_host[offset:offset + pages] = table_host[local, :pages]
                offset += pages
            self._get_eager_flashinfer_decode_wrapper().plan(indptr_host,
                                                             indices_host,
                                                             last_page_len_host,
                                                             self.num_q_heads,
                                                             self.num_kv_heads,
                                                             self.head_dim,
                                                             self.block_size,
                                                             pos_encoding_mode="NONE",
                                                             q_data_type=self.dtype,
                                                             kv_data_type=self.dtype,
                                                             sm_scale=self._scale)
        self._fi_step_planned = True

    def _flashinfer_decode_attention(self, layer_idx: int, query_states: torch.Tensor, scale: float) -> torch.Tensor:
        """Decode one token per row through FlashInfer's split-KV paged backend."""
        slot = self._kv_slot(layer_idx)
        if not self._fi_step_planned:
            # No host plan was issued this step; derive one from the device state.
            rows = self._active_rows.tolist()
            self.plan_flashinfer_decode(dict(zip(rows, self._write_positions[rows].tolist())))
        num_rows, num_heads, _, head_dim = query_states.shape
        if self._graph_mode:
            wrapper = self._flashinfer_decode_wrapper
            self._fi_decode_q[:num_rows].copy_(query_states.squeeze(2))
            query = self._fi_decode_q
            output = self._graph_outputs[slot].view(self.max_batch_size, num_heads, head_dim)
        else:
            wrapper = self._get_eager_flashinfer_decode_wrapper()
            query = query_states.squeeze(2)
            output = torch.empty((num_rows, num_heads, head_dim), dtype=query_states.dtype, device=query_states.device)
        paged_kv = (self._key_views[slot], self._value_views[slot])
        wrapper.run(query, paged_kv, out=output)
        return output[:num_rows].view(num_rows, 1, num_heads * head_dim)

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
        slot = self._kv_slot(layer_idx)
        if not self._graph_safe:
            self._free_row(target_row)
        self._ensure_blocks(target_row, prompt_len)
        positions = torch.arange(prompt_len, device=self.device)
        slots = self._slots(torch.tensor([target_row], device=self.device), positions[:1], prompt_len)[0]
        key_view = self._key_views[slot].view(-1, self.num_kv_heads, self.head_dim)
        value_view = self._value_views[slot].view(-1, self.num_kv_heads, self.head_dim)
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
        slot = self._kv_slot(layer_idx)
        rows = self._active_rows
        q_len = key_states.shape[2]
        if not self._graph_mode:
            self._prepare_step(q_len)
        elif q_len != 1:
            raise ValueError("graph replay paged writes expect one token per row")
        positions = self._write_positions[rows]
        slots = self._slots(rows, positions, q_len)
        key_view = self._key_views[slot].view(-1, self.num_kv_heads, self.head_dim)
        value_view = self._value_views[slot].view(-1, self.num_kv_heads, self.head_dim)
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
        slot = self._kv_slot(layer_idx)
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
        output = wrapper.run(q, (self._key_views[slot], self._value_views[slot]))
        return output.view(1, q.shape[0], query_states.shape[1] * query_states.shape[3])

    def attention(self, layer_idx: int, query_states: torch.Tensor, scale: float) -> torch.Tensor:
        """Run paged attention for the active rows.

        ``query_states`` is ``[rows, heads, q_len, head_dim]``; the return is
        ``[rows, q_len, heads * head_dim]`` head-major and contiguous, so the
        attention shim can feed ``o_proj`` without a copy. Graph replay derives the
        atom lengths on device and writes into a preallocated output, so the replay
        performs no host work.
        """
        num_rows, num_heads, q_len, head_dim = query_states.shape
        slot = self._kv_slot(layer_idx)
        if q_len > 1 and self.use_flashinfer_prefill:
            return self._flashinfer_prefill_attention(layer_idx, query_states, scale)
        if q_len == 1 and self.use_flashinfer_decode:
            return self._flashinfer_decode_attention(layer_idx, query_states, scale)
        if self._graph_mode and q_len != 1:
            raise ValueError("graph replay paged attention expects one query token per row")
        if self._graph_mode:
            if self._graph_outputs is None:
                raise RuntimeError("graph-safe paged cache has no output buffers")
            if num_rows != self._graph_width:
                raise ValueError(f"graph replay row count {num_rows} does not match width {self._graph_width}")
            atoms = self._graph_atoms
            output = self._graph_outputs[slot][:num_rows]
        else:
            atoms = self._build_atoms(q_len)
            output = torch.empty((num_rows * q_len, num_heads * head_dim),
                                 dtype=query_states.dtype,
                                 device=query_states.device)
        query = query_states.transpose(1, 2).reshape(num_rows * q_len, num_heads * head_dim)
        self._kernel(output, query, self._key_views[slot], self._value_views[slot], atoms, scale)
        return output.view(num_rows, q_len, num_heads * head_dim)

    def materialize(self, layer_idx: int, row: int, length: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather a row's blocks into dense ``[heads, length, head_dim]`` K/V for tests."""
        if length is None:
            length = int(self.row_len[row])
        slot = self._kv_slot(layer_idx)
        positions = torch.arange(length, device=self.device)
        slots = self._slots(torch.tensor([row], device=self.device), positions[:1], length)[0]
        keys = self._key_views[slot].view(-1, self.num_kv_heads, self.head_dim)[slots]
        values = self._value_views[slot].view(-1, self.num_kv_heads, self.head_dim)[slots]
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


class _HybridGDNLayer:
    """Per-layer GDN state surface a hybrid paged cache exposes in ``layers``.

    The model's GDN forward (and the segki fused GDN block) reads
    ``record_past``, ``conv_states[0]`` and ``recurrent_states[0]`` off the
    object it finds at ``cache.layers[layer_idx]``. The cache keeps those
    entries as views into the per-row state pools, so the model's in-place
    decode updates land on the pool rows directly."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.record_past = False
        self.conv_states: dict = {0: None}
        self.recurrent_states: dict = {0: None}

    @property
    def is_initialized(self) -> bool:
        # The pools are preallocated with the cache; nothing to lazy-init.
        return True


class HybridPagedAttentionCache(PagedAttentionCache):
    """Paged KV for full-attention layers plus per-row GDN state pools.

    Hybrid models carry two cache types with opposite growth profiles: KV
    scales with sequence length and is block-table addressed; GDN conv/
    recurrent state is fixed-size per row and cannot be paged (there is no
    temporal token structure to evict). This cache keeps the two in
    physically distinct pools behind one cache object, the way an asymmetric
    page table would: the KV pool is sized for the full-attention layers only,
    and every GDN layer owns row-aligned pools that the model's GDN forward
    updates in place through the HF cache protocol (get/update conv state,
    get/update recurrent state, is_recording_past).

    Row semantics: decode forwards run on the contiguous row prefix
    ``[0, active_width)``, which the CUDA-graph buckets and the eager
    compaction both guarantee (the in-place conv update needs a slice view of
    the pool, so arbitrary row sets are rejected); a paged prefill targets
    exactly one row, selected by ``set_active_rows`` while recording is on.
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
        use_flashinfer_decode: bool = False,
    ) -> None:
        text_config = getattr(config, "text_config", config)
        layer_types = list(getattr(text_config, "layer_types", None) or [])
        gdn_idx = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
        if not gdn_idx:
            raise ValueError("HybridPagedAttentionCache requires a config with linear-attention layers")
        kv_idx = [i for i, t in enumerate(layer_types) if t != "linear_attention"]
        super().__init__(config,
                         batch_size,
                         max_cache_len,
                         device,
                         dtype,
                         block_size=block_size,
                         use_flashinfer_prefill=use_flashinfer_prefill,
                         use_flashinfer_decode=use_flashinfer_decode,
                         num_kv_layers=len(kv_idx))
        slot_of = [None] * self.num_layers
        for slot, layer_idx in enumerate(kv_idx):
            slot_of[layer_idx] = slot
        self._layer_to_slot = slot_of
        self._gdn_layer_idx = gdn_idx
        self._gdn_slot = {layer_idx: i for i, layer_idx in enumerate(gdn_idx)}

        num_v_heads = int(getattr(text_config, "linear_num_value_heads"))
        num_k_heads = int(getattr(text_config, "linear_num_key_heads"))
        head_k = int(getattr(text_config, "linear_key_head_dim"))
        head_v = int(getattr(text_config, "linear_value_head_dim"))
        self._gdn_kernel = int(getattr(text_config, "linear_conv_kernel_dim"))
        conv_dim = 2 * num_k_heads * head_k + num_v_heads * head_v
        # The cached conv state holds the last kernel-sized window; the
        # per-token decode kernel reads its final kernel-1 entries from it.
        self._pool_conv = [
            torch.zeros((batch_size, conv_dim, self._gdn_kernel), dtype=dtype, device=self.device) for _ in gdn_idx
        ]
        self._pool_rec = [
            torch.zeros((batch_size, num_v_heads, head_k, head_v), dtype=dtype, device=self.device) for _ in gdn_idx
        ]
        for pool in self._pool_conv + self._pool_rec:
            # Decode kernels update these in place under graph capture; the
            # addresses must stay fixed for the captured graph's lifetime.
            torch._dynamo.mark_static_address(pool)
        self._gdn_objs = [_HybridGDNLayer(layer_idx) for layer_idx in gdn_idx]
        self._gdn_layer_obj = [None] * self.num_layers
        for layer_idx, obj in zip(gdn_idx, self._gdn_objs):
            self._gdn_layer_obj[layer_idx] = obj

        self._record_past = False
        self._gdn_width = 0
        self._gdn_prefill_row: Optional[int] = None
        # Rows whose prefill has started: chunk-1 conv state reads return None
        # for a row that has no state yet, matching the HF cache's
        # has_previous_state semantics.
        self._gdn_rows_with_state: set = set()
        self._gdn_has_state = False
        self._rebuild_gdn_views()

    # ------------------------------------------------------------------
    # GDN pool row selection
    # ------------------------------------------------------------------

    @property
    def has_gdn_layers(self) -> bool:
        return True

    def set_recording(self, flag: bool) -> None:
        """Toggle GDN state recording around the paged prefill forwards."""
        self._record_past = bool(flag)
        self._rebuild_gdn_views()

    def set_active_width(self, width: int) -> None:
        """Select the GDN pool rows a decode forward runs on: the prefix [0, width)."""
        self._gdn_width = int(width)
        self._rebuild_gdn_views()

    def set_active_rows(self, rows) -> None:
        super().set_active_rows(rows)
        flat = list(rows) if isinstance(rows, (list, tuple)) else rows.flatten().tolist()
        if self._record_past:
            # Paged prefill runs one request at a time; the GDN pools update
            # that row's state in place.
            if len(flat) != 1:
                raise ValueError("hybrid paged prefill targets exactly one row")
            self._gdn_prefill_row = int(flat[0])
        else:
            if flat != list(range(len(flat))):
                raise ValueError("hybrid paged decode needs the active rows as the contiguous prefix [0, width); "
                                 "compact the cache after retires")
            self._gdn_width = len(flat)
        self._rebuild_gdn_views()

    def set_graph_mode(self, width: int) -> None:
        super().set_graph_mode(width)
        self._gdn_width = width
        self._rebuild_gdn_views()

    def _gdn_span(self) -> tuple:
        if self._record_past:
            if self._gdn_prefill_row is None:
                return (0, 0)
            return (self._gdn_prefill_row, self._gdn_prefill_row + 1)
        return (0, self._gdn_width)

    def _rebuild_gdn_views(self) -> None:
        start, stop = self._gdn_span()
        for obj, conv, rec in zip(self._gdn_objs, self._pool_conv, self._pool_rec):
            obj.conv_states[0] = conv[start:stop]
            obj.recurrent_states[0] = rec[start:stop]
            obj.record_past = self._record_past

    def begin_prefill_row(self, row: int) -> None:
        """Reset one row's GDN state before a new prompt prefills into it.

        The row's KV blocks are rewritten from column zero, so its conv and
        recurrent state must start empty too; the prefill chunks rebuild it
        through the update_* protocol.
        """
        for conv, rec in zip(self._pool_conv, self._pool_rec):
            conv[row].zero_()
            rec[row].zero_()
        self._gdn_rows_with_state.discard(row)

    # ------------------------------------------------------------------
    # HF GDN cache protocol (the model's GDN forward talks to this cache
    # through these entries; the static-cache pool does the same)
    # ------------------------------------------------------------------

    def get_conv_state(self, layer_idx: int, state_idx: int = 0):
        if layer_idx not in self._gdn_slot:
            raise ValueError(f"layer {layer_idx} holds no GDN state")
        pool = self._pool_conv[self._gdn_slot[layer_idx]]
        if not self._record_past:
            return pool[:self._gdn_width]
        row = self._gdn_prefill_row
        if row is None or row not in self._gdn_rows_with_state:
            return None
        return pool[row:row + 1]

    def get_recurrent_state(self, layer_idx: int, state_idx: int = 0):
        if layer_idx not in self._gdn_slot:
            raise ValueError(f"layer {layer_idx} holds no GDN state")
        pool = self._pool_rec[self._gdn_slot[layer_idx]]
        if not self._record_past:
            return pool[:self._gdn_width]
        if self._gdn_prefill_row is None:
            return None
        return pool[self._gdn_prefill_row:self._gdn_prefill_row + 1]

    def update_conv_state(self, conv_state: torch.Tensor, layer_idx: int = 0, **kwargs) -> torch.Tensor:
        """Store the row's new conv state and return the left-extended input.

        The GDN forward runs the causal conv over the returned tensor and
        slices away the context prefix, so this mirrors HF's
        update_conv_state: fresh rows return the input as-is, rows with a
        state return state-plus-input.
        """
        if layer_idx not in self._gdn_slot:
            raise ValueError(f"layer {layer_idx} holds no GDN state")
        pool = self._pool_conv[self._gdn_slot[layer_idx]]
        kernel = self._gdn_kernel
        if not self._record_past:
            # One-token forwards that lost their state take the chunked
            # branch; the state still lands on the active prefix rows.
            if self._gdn_width:
                full = torch.cat([pool[:self._gdn_width], conv_state], dim=-1)
                pool[:self._gdn_width].copy_(full[..., -kernel:])
            else:
                full = conv_state
        else:
            row = self._gdn_prefill_row
            if row in self._gdn_rows_with_state:
                full = torch.cat([pool[row:row + 1], conv_state], dim=-1)
            else:
                full = conv_state
            # The pool row holds exactly the last kernel window; a prompt
            # shorter than the kernel is left-padded with zeros, which is the
            # causal conv's boundary context.
            tail = full[..., -kernel:]
            if tail.shape[-1] < kernel:
                tail = torch.nn.functional.pad(tail, (kernel - tail.shape[-1], 0))
            pool[row:row + 1].copy_(tail)
            self._gdn_rows_with_state.add(row)
        self._gdn_has_state = True
        return full

    def update_recurrent_state(self, recurrent_state: torch.Tensor, layer_idx: int = 0, state_idx: int = 0):
        if layer_idx not in self._gdn_slot:
            raise ValueError(f"layer {layer_idx} holds no GDN state")
        pool = self._pool_rec[self._gdn_slot[layer_idx]]
        if not self._record_past:
            pool[:self._gdn_width].copy_(recurrent_state)
            view = pool[:self._gdn_width]
        else:
            pool[self._gdn_prefill_row:self._gdn_prefill_row + 1].copy_(recurrent_state)
            self._gdn_rows_with_state.add(self._gdn_prefill_row)
            view = pool[self._gdn_prefill_row:self._gdn_prefill_row + 1]
        self._gdn_has_state = True
        return view

    def is_recording_past(self, layer_idx: int = 0, state_idx: int = 0) -> bool:
        return self._record_past

    def has_previous_state(self, layer_idx: int = 0, state_idx: int = 0) -> bool:
        # Python-level read, captured as a constant by graph capture: by the
        # time a decode graph is captured every row has been prefilled, and
        # the flag only ever turns back on.
        return self._gdn_has_state

    # ------------------------------------------------------------------
    # Row lifecycle: keep the pools aligned with the KV rows
    # ------------------------------------------------------------------

    @property
    def layers(self) -> list:
        layers = [None] * self.num_layers
        for layer_idx, slot in enumerate(self._layer_to_slot):
            if slot is not None:
                layers[layer_idx] = _PagedLayer(self._key_views[slot], self._value_views[slot])
        for layer_idx in self._gdn_layer_idx:
            layers[layer_idx] = self._gdn_layer_obj[layer_idx]
        return layers

    @property
    def memory_bytes(self) -> int:
        total = super().memory_bytes
        for pool in self._pool_conv + self._pool_rec:
            total += pool.nbytes
        return total

    def compact(self, active_indices: torch.Tensor) -> None:
        """Compact the KV rows and move the GDN pool rows with them."""
        super().compact(active_indices)
        keep = active_indices.to(device=self.device, dtype=torch.long)
        count = keep.numel()
        for pool in self._pool_conv + self._pool_rec:
            if count:
                pool[:count].copy_(pool.index_select(0, keep).clone())
            if count < pool.shape[0]:
                pool[count:].zero_()
        self._gdn_rows_with_state = {new for new, old in enumerate(keep.tolist()) if old in self._gdn_rows_with_state}
        self._gdn_width = min(self._gdn_width, count)
        self._rebuild_gdn_views()

    def reset(self) -> None:
        super().reset()
        for pool in self._pool_conv + self._pool_rec:
            pool.zero_()
        self._gdn_rows_with_state = set()
        self._gdn_has_state = False
        self._record_past = False
        self._gdn_prefill_row = None
        self._rebuild_gdn_views()

    # Prefix reuse would skip prompt tokens whose GDN state is not stored in
    # any block, and trimming would drop KV tokens the GDN state still
    # summarizes: both are state/KV misalignment, so fail closed.
    def prepare_prefix(self, hashes, target_row: int) -> int:
        raise ValueError("paged prefix reuse does not support hybrid models: blocks carry no GDN state")

    def register_prefix(self, hashes, target_row: int, count: int) -> None:
        raise ValueError("paged prefix reuse does not support hybrid models: blocks carry no GDN state")

    def trim_left(self, count: int) -> None:
        raise ValueError("cache trimming does not support hybrid models: GDN state cannot drop prefix tokens")


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

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    # Same math as transformers' apply_rotary_pos_emb without the rotate_half
    # cat allocation: rotate_half(x) = (-x2, x1), so x*cos + rotate_half(x)*sin
    # is (x1*cos1 - x2*sin1, x2*cos2 + x1*sin2) element-wise.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    half = self.head_dim // 2
    q1, q2 = query_states[..., :half], query_states[..., half:]
    k1, k2 = key_states[..., :half], key_states[..., half:]
    c1, c2 = cos[..., :half], cos[..., half:]
    s1, s2 = sin[..., :half], sin[..., half:]
    query_states = torch.empty_like(query_states)
    key_states = torch.empty_like(key_states)
    query_states[..., :half] = q1 * c1 - q2 * s1
    query_states[..., half:] = q2 * c2 + q1 * s2
    key_states[..., :half] = k1 * c1 - k2 * s1
    key_states[..., half:] = k2 * c2 + k1 * s2

    past_key_values.write(self.layer_idx, key_states, value_states)
    attn_output = past_key_values.attention(self.layer_idx, query_states, self.scaling)
    # attention() returns [rows, query, heads * head_dim] head-major and contiguous,
    # the same layout o_proj expects, so no flatten/copy is needed.
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def _apply_partial_rotary(query_states: torch.Tensor, key_states: torch.Tensor, cos: torch.Tensor,
                          sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE that rotates only the first ``cos.shape[-1]`` head dims.

    Mirrors transformers' apply_rotary_pos_emb for models with
    partial_rotary_factor < 1: the remainder of the head passes through
    unrotated. The full-width table takes the allocation-free fast path.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    head_dim = query_states.shape[-1]
    if rotary_dim < head_dim:
        query_states, q_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
        key_states, k_pass = key_states[..., :rotary_dim], key_states[..., rotary_dim:]
    else:
        q_pass = k_pass = None
    half = rotary_dim // 2
    q1, q2 = query_states[..., :half], query_states[..., half:]
    k1, k2 = key_states[..., :half], key_states[..., half:]
    c1, c2 = cos[..., :half], cos[..., half:]
    s1, s2 = sin[..., :half], sin[..., half:]
    query_out = torch.empty_like(query_states)
    key_out = torch.empty_like(key_states)
    query_out[..., :half] = q1 * c1 - q2 * s1
    query_out[..., half:] = q2 * c2 + q1 * s2
    key_out[..., :half] = k1 * c1 - k2 * s1
    key_out[..., half:] = k2 * c2 + k1 * s2
    if q_pass is None:
        return query_out, key_out
    return torch.cat([query_out, q_pass], dim=-1), torch.cat([key_out, k_pass], dim=-1)


def _paged_qwen3_5_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):
    """Paged forward for Qwen3_5Attention: q_proj carries a sigmoid attention
    gate alongside the query, and the model rotates only a partial head width
    (its RoPE tables are narrower than head_dim), so both are replayed here."""
    original_forward = self._paged_original_forward
    if not isinstance(past_key_values, PagedAttentionCache):
        return original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = _apply_partial_rotary(query_states, key_states, cos, sin)

    past_key_values.write(self.layer_idx, key_states, value_states)
    attn_output = past_key_values.attention(self.layer_idx, query_states, self.scaling)
    # attention() returns [rows, query, heads * head_dim] head-major, the layout
    # the gate and o_proj expect.
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def install_paged_attention(module) -> list:
    """Patch Qwen3/Qwen3.5 attention modules to use paged attention when handed a paged cache."""
    forwards = {"Qwen3Attention": _paged_qwen3_forward, "Qwen3_5Attention": _paged_qwen3_5_forward}
    patched = []
    for submodule in module.modules():
        forward = forwards.get(type(submodule).__name__)
        if forward is None or getattr(submodule, "_paged_original_forward", None) is not None:
            continue
        submodule._paged_original_forward = submodule.forward
        submodule.forward = MethodType(forward, submodule)
        patched.append(submodule)
    if not patched:
        raise ValueError("paged cache requires Qwen3Attention or Qwen3_5Attention modules")
    return patched

# Copyright (c) DeepSpeed Team
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""CUDA-graph-compatible static KV cache for hybrid engine rollout.

Derived from HuggingFace transformers ``StaticCache`` / ``StaticLayer``, but
with a critical difference: the write position is supplied externally via a
shared tensor instead of an internal ``cumulative_length`` counter.

Why this matters
----------------
Transformers' ``StaticLayer.update()`` maintains its own ``cumulative_length``
tensor that advances on every call.  During CUDA graph capture the captured
forward "freezes" this counter at whatever value it had at capture time.
On replay the counter does *not* advance, so subsequent KV writes go to the
wrong positions and the model silently produces incorrect logits.

Our ``DeepSpeedStaticCache`` instead reads the write position from a shared
tensor (``write_position``) that the caller updates in-place before each graph
replay.  Because ``write_position`` is a real tensor at a fixed address, CUDA
graph replays read the current value each time.

The caller (HybridEngineRollout) must call ``cache.set_write_position(pos)``
before each replay. ``pos`` may be a scalar (the existing fixed-batch behavior)
or one position per cache row for continuous batching. Per-row positions cover
both decode (one token per row) and chunked prefill (a chunk per row, written at
``[position_i, position_i + chunk)``), so rows can sit at different columns of
their own prompts.
"""

import torch

from deepspeed.accelerator import get_accelerator


class DeepSpeedStaticLayer:
    """A single layer's static KV cache whose write position is externally set.

    Parameters
    ----------
    max_cache_len : int
        Maximum number of tokens the cache can hold (last dim size).
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int):
        self.max_cache_len = max_cache_len
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.is_initialized = False
        self._write_position: torch.Tensor | None = None

    def set_write_position(self, pos: torch.Tensor):
        if not isinstance(pos, torch.Tensor) or pos.dim() not in (0, 1):
            raise ValueError("write position must be a scalar or a 1-D tensor")
        if pos.dtype not in (torch.int32, torch.int64):
            raise ValueError("write position must use an integer dtype")
        self._write_position = pos

    def lazy_initialization(self,
                            key_states: torch.Tensor,
                            value_states: torch.Tensor,
                            max_batch_size: int | None = None) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        num_heads = key_states.shape[1]
        if max_batch_size is None:
            max_batch_size = key_states.shape[0]
        self.max_batch_size = max_batch_size
        self.num_heads = num_heads
        self.k_head_dim = key_states.shape[-1]
        self.v_head_dim = value_states.shape[-1]

        self.keys = torch.zeros(
            (max_batch_size, num_heads, self.max_cache_len, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (max_batch_size, num_heads, self.max_cache_len, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        torch._dynamo.mark_static_address(self.keys)
        torch._dynamo.mark_static_address(self.values)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        kv_length = key_states.shape[-2]

        if self._write_position is not None and self._write_position.dim() == 1:
            if self._write_position.numel() < key_states.shape[0]:
                raise ValueError("per-row write positions must cover the active batch size")
            cache_position = self._write_position[:key_states.shape[0]].to(self.device)
            is_capturing = get_accelerator().is_current_stream_capturing()
            if not is_capturing and ((cache_position < 0).any() or
                                     (cache_position + kv_length - 1 >= self.max_cache_len).any()):
                raise ValueError("per-row write positions must be within the cache bounds")
            active_batch_size = key_states.shape[0]
            row_indices = torch.arange(active_batch_size, device=self.device)
            if kv_length == 1:
                self.keys[row_indices, :, cache_position, :] = key_states[:, :, 0, :]
                self.values[row_indices, :, cache_position, :] = value_states[:, :, 0, :]
            else:
                # A chunk per row: row i writes columns [position_i, position_i + kv_length), so
                # one cache can hold rows mid-way through different parts of their prompt.
                row_indices = row_indices[:, None]
                chunk_columns = cache_position[:, None] + torch.arange(kv_length, device=self.device)[None, :]
                self.keys[row_indices, :, chunk_columns] = key_states.permute(0, 2, 1, 3)
                self.values[row_indices, :, chunk_columns] = value_states.permute(0, 2, 1, 3)
            return self.keys[:active_batch_size], self.values[:active_batch_size]

        if self._write_position is not None:
            cache_position = torch.arange(kv_length, device=self.device) + self._write_position
        else:
            cache_position = torch.arange(kv_length, device=self.device)

        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states

        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # Mirror HF StaticLayer exactly: static caches answer with the full
        # buffer width, which matches the full-width K/V that update()
        # returns (position-based sizes are the DynamicLayer formula).
        return self.max_cache_len, 0

    def get_query_offset(self) -> int | torch.Tensor:
        # Mask construction precedes the current KV write, unlike get_seq_length().
        if self._write_position is None:
            return 0
        return self._write_position

    def get_seq_length(self) -> int | torch.Tensor:
        # Transformers >= 5.20 cache protocol (#8852): the length counts the
        # slot the current query is writing, so it is one past the write
        # position; position/mask math must use get_query_offset() instead
        # (it returns the write position itself). Our decode paths pass
        # cache_position/position_ids explicitly, so this value never feeds
        # RoPE. Kept as a tensor expression: this is read inside captured
        # regions and must not sync.
        if not self.is_initialized:
            return 0
        if self._write_position is not None:
            return self._write_position + 1
        return 0

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def memory_bytes(self) -> int:
        # Layers are allocated lazily from the first prefill, so an unused layer holds no memory yet.
        if not self.is_initialized:
            return 0
        return self.keys.nbytes + self.values.nbytes

    def reset(self) -> None:
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()

    def _compact_rows(self, active_indices: torch.Tensor, count: int | None = None) -> None:
        """Compact only the row tensors; used by the multi-layer cache."""
        if count is None:
            count = active_indices.numel()
        identity = torch.arange(count, device=active_indices.device, dtype=active_indices.dtype)
        if count and not torch.equal(active_indices, identity):
            keys = self.keys.index_select(0, active_indices).clone()
            values = self.values.index_select(0, active_indices).clone()
            self.keys[:count].copy_(keys)
            self.values[:count].copy_(values)
        # Clear retired rows so their stale KV cannot be observed after refill.
        if count < self.max_batch_size:
            self.keys[count:].zero_()
            self.values[count:].zero_()

    def _trim_left(self, count: int) -> None:
        """Shift the cache contents left while preserving tensor addresses."""
        if count <= 0:
            return
        if count >= self.max_cache_len:
            self.reset()
            return
        self.keys[:, :, :-count].copy_(self.keys[:, :, count:].clone())
        self.values[:, :, :-count].copy_(self.values[:, :, count:].clone())
        self.keys[:, :, -count:].zero_()
        self.values[:, :, -count:].zero_()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.is_initialized:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))
            if self._write_position is not None and self._write_position.dim() == 1:
                self._write_position = self._write_position.index_select(0, beam_idx.to(self._write_position.device))


class DSStaticGDNSlot:
    """Cache slot for hybrid models' linear-attention layers.

    Two ownership modes:

    *bound* (non-continuous paths): the slot adopts the prefill cache's HF
    ``LinearAttentionLayer`` by reference. HF's buffers are cudagraph-safe by
    construction: static shapes, ``mark_static_address`` buffers, and in-place
    ``copy_`` state updates. Duplicating that state management would only
    drift, so the slot forwards the HF API verbatim - zero copies, and
    decode-time updates land in the very buffers the prefill filled.

    *pooled* (continuous batching): the slot owns one full-batch pool of
    conv/recurrent state so every cache row carries its own state. Decode
    forwards read and write the pool's active prefix, prefills copy each row's
    state out of the prefill cache, and ``compact``/``reset`` keep the pool
    rows aligned with the KV rows."""

    is_compileable = True
    supports_early_init = False

    def __init__(self, max_cache_len: int):
        self._slot = None
        self.max_cache_len = max_cache_len
        self._write_position: torch.Tensor | None = None
        # Pooled (continuous-batching) ownership; populated by adopt_gdn_pool.
        self._owns_pool = False
        self._pool_batch_size = 0
        self._active_width = 0

    def bind(self, hf_slot) -> None:
        """Adopt a prefill cache's linear-attention slot (by reference)."""
        self._slot = hf_slot

    def __getattr__(self, name):
        # Forward the HF slot API (conv_states, recurrent_states,
        # has_previous_state, update_*, ...) so the model's GDN modules see
        # the exact object semantics they were written against.
        slot = object.__getattribute__(self, "_slot")
        if slot is None:
            raise AttributeError(f"DSStaticGDNSlot not bound: {name}")
        return getattr(slot, name)

    def adopt_gdn_pool(self, hf_slot, max_batch_size: int) -> None:
        """Allocate the full-batch conv/recurrent state pool this row owns.

        Shapes and dtype come from the prefill cache's slot after its first
        prefill; the pool starts zeroed and no row is considered to hold
        state until a prefill copies its state in. Idempotent.
        """
        if self._owns_pool:
            return
        conv_state = hf_slot.conv_states[0]
        if conv_state is None:
            raise ValueError("cannot pool a GDN layer whose prefill cache holds no conv state")
        rec_state = hf_slot.recurrent_states[0]
        device, dtype = conv_state.device, conv_state.dtype
        self._pool_conv = torch.zeros((max_batch_size, *conv_state.shape[1:]), dtype=dtype, device=device)
        self._pool_rec = (torch.zeros(
            (max_batch_size,
             *rec_state.shape[1:]), dtype=rec_state.dtype, device=rec_state.device) if rec_state is not None else None)
        torch._dynamo.mark_static_address(self._pool_conv)
        if self._pool_rec is not None:
            torch._dynamo.mark_static_address(self._pool_rec)
        self._owns_pool = True
        self._pool_batch_size = max_batch_size
        self._active_width = 0
        self.has_previous_state = {0: False}
        # Pooled state is always the live decode state: the model's GDN layers
        # must take the fused per-token update path, never the record-the-past
        # chunked-prefill branch, so the layer-level flag stays False.
        self.record_past = False

    def copy_state_from(self, hf_slot, source_row: int, target_row: int) -> None:
        """Move one row's prefilled conv/recurrent state into the pool."""
        if not self._owns_pool:
            raise ValueError("GDN state pools must be adopted before copying rows")
        self._pool_conv[target_row].copy_(hf_slot.conv_states[0][source_row])
        if self._pool_rec is not None:
            self._pool_rec[target_row].copy_(hf_slot.recurrent_states[0][source_row])
        self.has_previous_state[0] = True

    def compact_rows(self, active_indices: torch.Tensor) -> None:
        """Move the pool rows the way the KV rows are compacted."""
        if not self._owns_pool:
            return
        count = active_indices.numel()
        identity = torch.arange(count, device=active_indices.device, dtype=active_indices.dtype)
        if count and not torch.equal(active_indices, identity):
            for states in (self._pool_conv, self._pool_rec):
                if states is not None:
                    states[:count].copy_(states.index_select(0, active_indices).clone())
        if count < self._pool_batch_size:
            for states in (self._pool_conv, self._pool_rec):
                if states is not None:
                    states[count:].zero_()

    def set_active_width(self, width: int) -> None:
        """Decode forwards read and write pool rows [0, width)."""
        self._active_width = width

    def active_width(self) -> int:
        return self._active_width

    @property
    def is_initialized(self):
        # The capture warmup probe skips already-allocated buffers; a pooled
        # slot counts as initialized once its pool exists, a bound slot keeps
        # the HF slot's own view.
        if self._owns_pool:
            return True
        slot = self._slot
        return bool(getattr(slot, "is_initialized", False)) if slot is not None else False

    @property
    def conv_states(self):
        # Pooled mode serves the active prefix so the model's in-place
        # decode updates (conv_state.copy_) land on the active pool rows
        # only; bound mode forwards the HF slot's own dict.
        if self._owns_pool:
            return {0: self._pool_conv[:self._active_width]}
        slot = object.__getattribute__(self, "_slot")
        if slot is None:
            raise AttributeError("DSStaticGDNSlot not bound: conv_states")
        return slot.conv_states

    @property
    def recurrent_states(self):
        if self._owns_pool:
            return {0: self._pool_rec[:self._active_width] if self._pool_rec is not None else None}
        slot = object.__getattribute__(self, "_slot")
        if slot is None:
            raise AttributeError("DSStaticGDNSlot not bound: recurrent_states")
        return slot.recurrent_states

    def memory_bytes(self) -> int:
        if self._owns_pool:
            total = self._pool_conv.nbytes
            if self._pool_rec is not None:
                total += self._pool_rec.nbytes
            return total
        # Bound mode: the state tensors belong to the prefill cache's slot,
        # which already accounts for them.
        return 0

    def set_write_position(self, pos: torch.Tensor):
        self._write_position = pos  # tracked for symmetry; GDN state is position-free

    def get_seq_length(self) -> int:
        return 0

    def get_max_cache_shape(self) -> int:
        return 0

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return query_length, 0

    def reset(self) -> None:
        if self._owns_pool:
            self._pool_conv.zero_()
            if self._pool_rec is not None:
                self._pool_rec.zero_()
            self.has_previous_state[0] = False
            self._active_width = 0
        elif self._slot is not None:
            self._slot.reset()


class DeepSpeedStaticCache:
    # Mirrors HF StaticCache: static buffers answer mask builders with
    # full-width masks (is_compileable), which must match the full-width
    # K/V buffers update() returns; a position-truncated mask against a
    # full-width buffer crashes SDPA at decode.
    is_compileable = True
    """CUDA-graph-compatible static KV cache.

    Drop-in replacement for ``transformers.StaticCache`` in the graph-capture
    decode path of ``HybridEngineRollout``.  All layers share a single
    ``write_position`` tensor that the caller updates before each graph replay.

    Parameters
    ----------
    config : PreTrainedConfig
        HuggingFace model config (used to determine number of layers and head
        dimensions).
    batch_size : int
        Batch size for eager initialization.
    max_cache_len : int
        Maximum sequence length (prompt + generated tokens).
    device : torch.device | int | str | None
        Device for eager initialization.
    dtype : torch.dtype | None
        Dtype for eager initialization.
    """

    def __init__(
        self,
        config,
        batch_size: int = 1,
        max_cache_len: int = 4096,
        device=None,
        dtype=None,
    ):
        self.config = config
        text_config = getattr(config, "text_config", config)
        num_layers = getattr(text_config, "num_hidden_layers", 1)
        # Hybrid families (e.g. qwen3_5) mix full_attention and
        # linear_attention blocks behind one decoder-layer list; mirror the
        # split so KV slots and GDN pass-through slots line up by index.
        layer_types = getattr(text_config, "layer_types", None) or ["full_attention"] * num_layers
        self._layers = [
            DSStaticGDNSlot(max_cache_len) if t == "linear_attention" else DeepSpeedStaticLayer(max_cache_len)
            for t in layer_types
        ]
        self._max_cache_len = max_cache_len
        self._write_position: torch.Tensor | None = None
        self._has_gdn_layers = any(isinstance(layer, DSStaticGDNSlot) for layer in self._layers)

        if dtype is not None and device is not None and batch_size > 0:
            num_heads = getattr(text_config, "num_key_value_heads", getattr(text_config, "num_attention_heads", 1))
            # head_dim is not always hidden_size // num_attention_heads, so prefer the config value
            head_dim = getattr(text_config, "head_dim", None) or (getattr(text_config, "hidden_size", 1) //
                                                                  getattr(text_config, "num_attention_heads", 1))
            self.early_initialization(batch_size, num_heads, head_dim, dtype, device)

    @property
    def layers(self):
        return self._layers

    @property
    def has_gdn_layers(self) -> bool:
        """Continuous batching must scatter/compact GDN state rows and feed
        the model a per-type attention mask; the CB loops branch on this."""
        return self._has_gdn_layers

    def memory_bytes(self) -> int:
        """Return the bytes held by the layers' cache tensors."""
        return sum(layer.memory_bytes() for layer in self._layers)

    def set_write_position(self, pos: torch.Tensor):
        """Set the write position shared by all layers.

        Must be called before each graph replay. A scalar tensor preserves the
        fixed-batch path; a vector supplies one decode position per cache row.
        The tensor is stored by reference so subsequent in-place updates are
        immediately visible to all layers.
        """
        self._write_position = pos
        for layer in self._layers:
            layer.set_write_position(pos)

    def compact(self, active_indices: torch.Tensor) -> None:
        """Compact active rows after requests retire from a batch."""
        if not isinstance(active_indices, torch.Tensor) or active_indices.dim() != 1:
            raise ValueError("active_indices must be a 1-D tensor")
        # Hybrid caches mix KV layers with GDN state pools; the pools keep
        # their rows aligned with the KV rows, so both compact together.
        kv_layers = [layer for layer in self._layers if not isinstance(layer, DSStaticGDNSlot)]
        gdn_layers = [layer for layer in self._layers if isinstance(layer, DSStaticGDNSlot) and layer._owns_pool]
        if not kv_layers and not gdn_layers:
            return
        max_batch_size = kv_layers[0].max_batch_size if kv_layers else gdn_layers[0]._pool_batch_size
        device = kv_layers[0].keys.device if kv_layers else gdn_layers[0]._pool_conv.device
        active_indices = active_indices.to(device=device, dtype=torch.long)
        if active_indices.numel() > max_batch_size:
            raise ValueError("active_indices exceeds the cache batch size")
        if active_indices.numel() and ((active_indices < 0).any() or (active_indices >= max_batch_size).any()):
            raise ValueError("active_indices contains an out-of-range row")
        if active_indices.unique().numel() != active_indices.numel():
            raise ValueError("active_indices must not contain duplicates")
        for layer in kv_layers:
            layer._compact_rows(active_indices)
        for layer in gdn_layers:
            layer.compact_rows(active_indices)
        if self._write_position is not None and self._write_position.dim() == 1:
            positions = self._write_position
            if positions.numel() != max_batch_size:
                raise ValueError("per-row write positions must match the cache batch size")
            position_indices = active_indices.to(positions.device)
            count = active_indices.numel()
            compacted = positions.index_select(0, position_indices).clone() if count else positions[:0]
            positions[:count].copy_(compacted)
            if count < positions.numel():
                positions[count:].fill_(-1)
        # The physical capacity stays fixed; only the active prefix shrinks.
        # Hybrid linear-attention state is read through the active prefix.
        self._active_batch_size = active_indices.numel()

    def trim_left(self, count: int) -> None:
        """Shift all cache layers left to reclaim an unused prefix span."""
        if not isinstance(count, int) or count < 0:
            raise ValueError("trim count must be a non-negative integer")
        if count == 0:
            return
        for layer in self._layers:
            # GDN state carries no positions, so trimming KV columns leaves
            # it untouched.
            if isinstance(layer, DSStaticGDNSlot):
                continue
            layer._trim_left(count)

    def set_active_width(self, width: int) -> None:
        """Select the GDN pool rows the next decode forward reads and writes.

        Continuous batching calls this with the active row count (eager path)
        or the captured bucket width (graph path) before each decode forward.
        No-op for caches without linear-attention layers.
        """
        for layer in self._layers:
            if isinstance(layer, DSStaticGDNSlot):
                layer.set_active_width(width)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self._layers):
            raise IndexError(f"layer_idx {layer_idx} out of range (cache has {len(self._layers)} layers)")
        layer = self._layers[layer_idx]
        if isinstance(layer, DSStaticGDNSlot):
            raise TypeError("linear-attention slots manage conv/recurrent state, not key/value pairs")
        return layer.update(key_states, value_states, *args, **kwargs)

    def early_initialization(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device,
    ):
        for layer in self._layers:
            if isinstance(layer, DSStaticGDNSlot):
                continue  # bound to the prefill cache's HF slot later
            fake_k = torch.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype, device=device)
            fake_v = torch.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype, device=device)
            layer.lazy_initialization(fake_k, fake_v)

    def _first_attention_idx(self) -> int | None:
        # HF convention (cache_utils): alternating caches answer container
        # length queries with attention-slot semantics; a linear-attention
        # slot never tracks sequence length.
        for idx, layer in enumerate(self._layers):
            if not isinstance(layer, DSStaticGDNSlot):
                return idx
        return None

    def get_seq_length(self, layer_idx: int = 0) -> int | torch.Tensor:
        if layer_idx >= len(self._layers):
            return 0
        if isinstance(self._layers[layer_idx], DSStaticGDNSlot):
            first = self._first_attention_idx()
            if first is None:
                return 0
            layer_idx = first
        length = self._layers[layer_idx].get_seq_length()
        if isinstance(length, torch.Tensor) and length.dim() == 1:
            # Transformer decoder implementations use one scalar past length
            # to build the causal mask. Continuous batching keeps rows aligned
            # to a shared physical position, so the largest row is sufficient.
            return length.max()
        return length

    def update_recurrent_state(self, recurrent_state, layer_idx: int = 0, state_idx: int = 0):
        layer = self._layers[layer_idx]
        if isinstance(layer, DSStaticGDNSlot) and layer._owns_pool:
            states = layer._pool_rec
            if states is not None:
                states[:layer.active_width()].copy_(recurrent_state)
            layer.has_previous_state[0] = True
            return recurrent_state
        return layer.update_recurrent_state(recurrent_state, state_idx=state_idx)

    def update_conv_state(self, conv_state, layer_idx: int = 0, **kwargs):
        layer = self._layers[layer_idx]
        if isinstance(layer, DSStaticGDNSlot) and layer._owns_pool:
            states = layer._pool_conv
            states[:layer.active_width()].copy_(conv_state[..., -states.shape[-1]:])
            layer.has_previous_state[0] = True
            return states[:layer.active_width()]
        return layer.update_conv_state(conv_state, state_idx=0, **kwargs)

    def get_conv_state(self, layer_idx: int, state_idx: int = 0):
        """HF >= 5.19 GDN cache protocol bridge: the native (non-segKI)
        forward fetches conv/recurrent state through these top-level
        entries; pre-5.19 builds poke layer attributes directly, which the
        slot's attribute forwarding already serves. Pooled slots answer with
        the active prefix of their state pool, a view whose in-place decode
        updates land on the active pool rows only."""
        layer = self._layers[layer_idx]
        if isinstance(layer, DSStaticGDNSlot) and layer._owns_pool:
            if not layer.has_previous_state[state_idx]:
                return None
            return layer._pool_conv[:layer.active_width()]
        return layer.conv_states[state_idx]

    def get_recurrent_state(self, layer_idx: int, state_idx: int = 0):
        layer = self._layers[layer_idx]
        if isinstance(layer, DSStaticGDNSlot) and layer._owns_pool:
            if not layer.has_previous_state[state_idx]:
                return None
            states = layer._pool_rec
            return None if states is None else states[:layer.active_width()]
        return layer.recurrent_states[state_idx]

    def is_recording_past(self, layer_idx: int, state_idx: int = 0) -> bool:
        layer = self._layers[layer_idx]
        if isinstance(layer, DSStaticGDNSlot) and layer._owns_pool:
            # Pooled state is always the live decode state; recording the
            # past would push the model's GDN layers onto the chunked-prefill
            # branch instead of the fused decode kernels.
            return False
        return layer.is_recording_past(state_idx) if hasattr(layer, "is_recording_past") else False

    def has_previous_state(self, layer_idx: int = 0, state_idx: int = 0) -> bool:
        # transformers cache protocol: GDN blocks ask the container whether a
        # previous recurrent state exists before reading it.
        if layer_idx >= len(self._layers):
            return False
        layer = self._layers[layer_idx]
        if isinstance(layer, DSStaticGDNSlot):
            return layer.has_previous_state[state_idx]
        return self.get_seq_length(layer_idx) > 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._layers):
            return self._max_cache_len
        return self._layers[layer_idx].get_max_cache_shape()

    def get_query_offset(self, layer_idx: int = 0) -> int | torch.Tensor:
        if layer_idx >= len(self._layers):
            return 0
        offset = self._layers[layer_idx].get_query_offset()
        if isinstance(offset, torch.Tensor) and offset.dim() == 1:
            # As with get_seq_length(), per-row masks handle validity below this shared frontier.
            return offset.max()
        return offset

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        # HF convention: mask builders sample shared masks with the default
        # layer_idx, which on hybrid models may land on a linear-attention
        # slot; redirect such queries to the first attention slot so the
        # shared mask gets attention-sized spans.
        if layer_idx < len(self._layers) and isinstance(self._layers[layer_idx], DSStaticGDNSlot):
            first = self._first_attention_idx()
            if first is None:
                return query_length, 0
            return self._layers[first].get_mask_sizes(query_length)
        if layer_idx >= len(self._layers):
            return query_length, 0
        return self._layers[layer_idx].get_mask_sizes(query_length)

    def reset(self):
        for layer in self._layers:
            layer.reset()

    def __len__(self):
        return len(self._layers)

# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine backed by DeepSpeed's hybrid engine.

Two generation paths:
  1. **model.generate()** (default): delegates to HuggingFace generate.
     Supports sampling (temperature, top_p) and greedy.
  2. **graph capture + DeepSpeedStaticCache**: greedy and sampling
     (temperature, top_k, top_p). Pre-allocates a StaticCache, captures the
     decode forward pass into a graph, and replays it for each decode
     step.  Eliminates kernel launch overhead.
  3. **continuous batching (experimental)**: an opt-in bounded path
     selected through ``SamplingConfig.continuous_batch_size``; greedy or
     sampled like the graph path, with or without graph capture. The captured
     path dispatches each step to the smallest captured batch width from a
     ladder, so a rollout never replays a wider graph than the rows it is running.
     With chunked prefill it also captures the prompt's chunk window, so a
     staged prompt costs one narrow prefill replay per step.
"""

import itertools
import math
import time
from copy import copy
from dataclasses import dataclass, field, replace
from functools import lru_cache
from inspect import signature
from typing import Dict, Optional, Tuple

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.continuous_batching import (ContinuousBatchRequest, ContinuousBatchScheduler,
                                                           plan_prefill_buckets)
from deepspeed.utils import logger


def _cache_bytes(cache) -> int:
    """Bytes held by a cache's key/value tensors, or zero for a cache that is not allocated."""
    if cache is None:
        return 0
    layers = getattr(cache, "layers", None)
    if layers is None:
        return 0
    total = 0
    for layer in layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            continue
        total += keys.numel() * keys.element_size() + values.numel() * values.element_size()
    return total


class _ForwardProfiler:
    """Measure top-level model forwards without synchronizing each call."""

    def __init__(self, accelerator):
        self.event_type = accelerator.Event
        self.use_host_timer = accelerator.use_host_timers()
        # Host intervals are only reliable when there is no asynchronous device
        # work, as on CPU. MPS exposes events but cannot time them, so its host
        # intervals would measure enqueue latency rather than model execution.
        self.is_available = not self.use_host_timer or self.event_type is None
        self.active_start = None
        self.measurements = []

    def register(self, module):
        if not self.is_available:
            return ()
        pre_handle = module.register_forward_pre_hook(self._start_forward)
        post_handle = module.register_forward_hook(self._end_forward)
        return pre_handle, post_handle

    def get_stage_times(self):
        if not self.measurements:
            return None, None, 0

        durations = [self._elapsed_time(start, end) for start, end in self.measurements]
        return durations[0], sum(durations[1:]), len(durations) - 1

    def _start_forward(self, _module, _args):
        self.active_start = self._record_time()

    def _end_forward(self, _module, _args, _output):
        end = self._record_time()
        self.measurements.append((self.active_start, end))
        self.active_start = None

    def _record_time(self):
        if self.use_host_timer:
            return time.perf_counter()
        event = self.event_type(enable_timing=True)
        event.record()
        return event

    def _elapsed_time(self, start, end):
        if self.use_host_timer:
            return (end - start) * 1000.0
        return start.elapsed_time(end)


class GlobalWorkRemaining:
    """Rendezvous a local work flag over a distributed process group."""

    def __init__(self, process_group=None):
        self.process_group = process_group

    def __call__(self, local_work_remaining):
        work_remaining = torch.as_tensor(local_work_remaining, dtype=torch.int32).clone()
        if dist.is_initialized() and dist.get_world_size(group=self.process_group) > 1:
            dist.all_reduce(work_remaining, op=dist.ReduceOp.MAX, group=self.process_group)
        return bool(work_remaining.item())


class _GlobalEosStoppingCriteria:
    """Stop all local sequences only after every rank has observed EOS."""

    def __init__(self, response_start, eos_token_id, work_remaining):
        self.response_start = response_start
        self.eos_token_ids = torch.as_tensor(eos_token_id).flatten()
        self.work_remaining = work_remaining

    def __call__(self, input_ids, scores, **kwargs):  # noqa: ARG002
        response_ids = input_ids[:, self.response_start:]
        eos_token_ids = self.eos_token_ids.to(device=input_ids.device, dtype=input_ids.dtype)
        sequence_has_eos = (response_ids.unsqueeze(-1) == eos_token_ids).any(dim=-1).any(dim=-1)
        local_work_remaining = (~sequence_has_eos).any().to(dtype=torch.int32)
        global_work_remaining = self.work_remaining(local_work_remaining)
        # This is a rank-global decision, so a scalar preserves the same
        # behavior for every local sequence and remains compatible with older
        # Transformers releases that apply Python ``any`` to criterion results.
        return not global_work_remaining


@dataclass
class HybridEngineRolloutConfig:
    """Configuration for HybridEngineRollout.

    ``align_decode_fronts`` is opt-in. When false, continuous batching keeps
    the existing equal-width padded layout. Requests may still have different
    effective lengths in their attention masks, but their physical decode
    fronts are not right-aligned or compacted. When true, the rollout uses
    effective prompt lengths to align physical decode fronts.

    Cache trimming is opt-in because it copies the live cache columns after
    each reclaim. Without it, the rollout trims only when the cache would
    otherwise exhaust and a dead prefix can be reclaimed.

    ``graph_capture_batch_sizes`` controls the captured-batch ladder of the
    continuous graph path. The loop dispatches each step to the smallest
    captured batch that fits the rows doing work, so a request arriving on its
    own replays a one-row graph instead of the full batch, and a step that
    stages one prompt captures prefill chunks at one row instead of padding up
    to the whole batch. Defaults to the powers of two up to ``max_batch_size``
    plus ``max_batch_size`` itself. Capture is lazy: a bucket is captured the
    first time the loop needs it, and a bucket whose capture fails runs that
    forward eagerly instead.

    Chunked prefill is opt-in and covers every greedy generation path: both
    continuous-batching paths, the captured-graph static path, and the plain
    ``model.generate()`` path (which is handed a cache that already holds the
    prompt). When enabled, a prompt whose effective length exceeds
    ``prefill_chunk_size`` is staged in a private cache and its prefill advances by at
    most ``prefill_chunk_size`` tokens per forward, so rows that are already decoding
    keep making progress while a long prompt fills. Staging is decided on real tokens,
    so a batch padded to a wide common width does not stage prompts that are actually
    short, and a staged request spends no generation budget until its prompt is
    complete. Sampling rollouts prefill in a single forward and log a warning. The
    cost is one staging cache per prompt being filled, sized to that prompt's
    effective length.
    """
    use_graph_capture: bool = False
    enable_profiling: bool = False
    use_shared_prefill: bool = False
    align_decode_fronts: bool = False
    # Apply segment-KI kernel injection to the engine's module at rollout
    # construction. Fuses comm-free projection+activation segments into
    # single GEMM+custom-kernel calls (MLP gate|up and GDN input projections).
    enable_cache_trimming: bool = False
    continuous_cache_capacity: Optional[int] = None
    adaptive_prefill: bool = False
    prefill_fixed_cost_ms: float = 25.0
    prefill_token_cost_ms: float = 0.1
    prefill_attention_cost_ms: float = 0.01
    prefill_kv_cost_ms: float = 0.04
    continuous_decode_cost_ms: float = 10.0
    prefill_max_tokens: Optional[int] = 65536
    # PR-local switches; kept after every upstream field so the legacy
    # positional constructor contract (#8786's test) keeps holding
    use_segki: bool = False
    graph_capture_batch_sizes: Optional[Tuple[int, ...]] = None
    enable_chunked_prefill: bool = False
    prefill_chunk_size: int = 512


@dataclass
class _DecodeBucket:
    """Captured decode forward for one batch width.

    The buffers are views of the rollout's master buffers, so the per-row state the
    scheduler writes (tokens, positions, validity, write positions) is visible to every
    bucket that covers those rows. ``logits`` is the tensor the captured forward returns
    and the graph overwrites in place on every replay.
    """

    width: int
    input_ids: torch.Tensor
    causal_attention: torch.Tensor
    position_ids: torch.Tensor
    write_positions: torch.Tensor
    graph: object = None
    logits: Optional[torch.Tensor] = None
    disabled: bool = False


@dataclass
class _PrefillBucket:
    """Captured chunked-prefill forward for one batch width.

    One row per staged prompt, and the chunk width is ``prefill_chunk_size`` for every row
    and every step: the only shape that varies between steps is the number of staged
    prompts, which is what the ladder absorbs. Rows past the staged count are parked
    behind an all-masked attention row, exactly like parked decode rows.
    """

    width: int
    input_ids: torch.Tensor
    causal_attention: torch.Tensor
    position_ids: torch.Tensor
    write_positions: torch.Tensor
    graph: object = None
    logits: Optional[torch.Tensor] = None
    disabled: bool = False


@dataclass
class _StagedPrompt:
    """A prompt filled one chunk per step inside the shared staging cache.

    ``slot`` is the decode-cache row the prompt will land in, ``row`` is its own row in
    the staging cache, and ``start`` is the first column of its next chunk window. A row
    belongs to exactly one prompt while it is staged, so a staged prompt never reads
    another prompt's columns.
    """

    slot: int
    row: Optional[int]
    prompt_ids: torch.Tensor
    prompt_attention: torch.Tensor
    position_ids: torch.Tensor
    start: int = 0

    @property
    def prompt_len(self):
        """Padded prompt width, which every request in a continuous batch shares."""
        return self.prompt_ids.shape[1]


@dataclass
class _PrefillChunk:
    """State of a prompt whose prefill is split across scheduler steps.

    ``cache`` holds the prefix computed by the chunks so far; the next chunk's
    attention reads it, so the two only ever move forward together. ``offset`` is
    the first prompt column the next chunk writes, which is also the number of
    columns already staged.
    """

    slot: int
    cache: object
    prompt_ids: torch.Tensor
    prompt_attention: torch.Tensor
    position_ids: torch.Tensor
    offset: int = 0

    @property
    def prompt_len(self):
        return self.prompt_ids.shape[1]

    @property
    def remaining(self):
        return self.prompt_len - self.offset

    def staging_bytes(self):
        """Bytes held by the staging cache, mirroring the main cache accounting.

        Transformers allocates a static layer's key/value tensors on first use, so a
        cache that has not run a chunk yet reports zero here.
        """
        return _cache_bytes(self.cache)


@dataclass
class _ContinuousGraphState:
    """Captured graphs and static buffers shared by rollouts with the same shapes."""

    signature: tuple
    ladder: Tuple[int, ...]
    cache: object
    staging_cache: object
    static_input: torch.Tensor
    static_attention: torch.Tensor
    static_causal_attention: torch.Tensor
    static_position_ids: torch.Tensor
    static_write_positions: torch.Tensor
    static_cache_position: torch.Tensor
    decode: Dict[int, _DecodeBucket] = field(default_factory=dict)
    prefill: Dict[int, _PrefillBucket] = field(default_factory=dict)
    captures: int = 0


class HybridEngineRollout(RolloutEngine):
    """Rollout engine using DeepSpeed hybrid engine.

    Args:
        engine: DeepSpeed engine wrapping the model.
        tokenizer: HuggingFace tokenizer (must have pad_token_id or eos_token_id).
        cfg: Optional HybridEngineRolloutConfig.
    """

    def __init__(self, engine, tokenizer, cfg=None):
        self.engine = engine
        self.tokenizer = tokenizer
        self.use_graph_capture = getattr(cfg, 'use_graph_capture', False) if cfg else False
        self.enable_profiling = getattr(cfg, 'enable_profiling', False) if cfg else False
        self.use_shared_prefill = getattr(cfg, 'use_shared_prefill', False) if cfg else False
        self.align_decode_fronts = getattr(cfg, 'align_decode_fronts', False) if cfg else False
        self.adaptive_prefill = getattr(cfg, 'adaptive_prefill', False) if cfg else False
        for name, default in (("prefill_fixed_cost_ms", 25.0), ("prefill_token_cost_ms", 0.1),
                              ("prefill_attention_cost_ms", 0.01), ("prefill_kv_cost_ms",
                                                                    0.04), ("continuous_decode_cost_ms", 10.0)):
            value = getattr(cfg, name, default) if cfg else default
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            setattr(self, name, value)
        self.prefill_max_tokens = getattr(cfg, 'prefill_max_tokens', 65536) if cfg else 65536
        if self.prefill_max_tokens is not None:
            if not isinstance(self.prefill_max_tokens, int) or self.prefill_max_tokens <= 0:
                raise ValueError("prefill_max_tokens must be a positive integer or None")
        self.enable_cache_trimming = getattr(cfg, 'enable_cache_trimming', False) if cfg else False
        self.continuous_cache_capacity = getattr(cfg, 'continuous_cache_capacity', None) if cfg else None
        self._last_profile = None
        self._continuous_graph_state = None
        self._last_continuous_stats = None
        self._generation_defaults = None
        self._chunked_prefill_warned = False

        self.graph_capture_batch_sizes = getattr(cfg, 'graph_capture_batch_sizes', None) if cfg else None
        self.enable_chunked_prefill = getattr(cfg, 'enable_chunked_prefill', False) if cfg else False
        self.prefill_chunk_size = getattr(cfg, 'prefill_chunk_size', 0) if cfg else 0
        self.use_segki = cfg is not None and getattr(cfg, 'use_segki', False)
        if self.use_segki:
            self._segki_report = self._apply_segki()
        else:
            self._segki_report = None

    def refresh_weights(self) -> None:
        """Handshake for in-place weight updates (arena / reshard syncs).

        Drops the injected forwards' derived-weight caches. The weights
        themselves are read live, so only the derived tensors go stale.
        A retained continuous-batching graph captures those tensors'
        addresses, so it is released here too (re-captured on the next
        rollout) -- replays would otherwise read freed memory. Interim
        cost: one re-capture per sync; remove when refresh refills the
        derived tensors in place and graphs survive by construction.

        PROVISIONAL contract — see segki_refresh; the final weight-sync
        handshake is still under design.
        """
        from deepspeed.module_inject.segment_ki import segki_refresh
        if self._continuous_graph_state is not None:
            get_accelerator().synchronize()
            self._continuous_graph_state = None
        # Resolve through the engine like every other module access; unit
        # tests hand us SimpleNamespace(module=...) engines.
        module = getattr(self.engine, "module", None) or getattr(self, "module", None)
        if module is None:
            return
        segki_refresh(module)

    def _apply_segki(self):
        """Apply segment-KI kernel injection to the wrapped module.

        Returns the injection report dict, or None if the module is not
        supported (unsupported architectures pass through silently).
        """
        try:
            from deepspeed.module_inject.segment_ki import apply_segment_ki
        except ImportError:
            return None
        report = apply_segment_ki(self.engine.module)
        if isinstance(report, dict) and report.get('segments_replaced', 0) == 0 \
                and report.get('segments_found', 0) == 0:
            # No segments found: model architecture not supported, not an error
            return report
        return report

    @torch.no_grad()
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        if sampling.continuous_batch_size is not None:
            return self._generate_continuous(request, sampling, sampling.continuous_batch_size)
        return self._generate_batched(request, sampling)

    def _chunked_prefill_enabled(self, sampling) -> bool:
        """Whether this rollout stages long prompts for ``sampling``.

        Chunked prefill currently covers greedy generation only: a sampled response has
        to draw its first token from the same distribution as the rest of its response,
        and the staged path's completing chunk does not sample yet. A sampling rollout
        falls back to a single-forward prefill, like every rollout before chunking.
        """
        if not self.enable_chunked_prefill:
            return False
        if sampling.temperature > 0.0:
            if not self._chunked_prefill_warned:
                logger.warning("Rollout chunked prefill currently covers greedy generation only; "
                               "sampling rollouts prefill the prompt in a single forward.")
                self._chunked_prefill_warned = True
            return False
        return True

    def _generation_kwargs(self, sampling, attention_mask, pad_token_id, module=None):
        zero3_sync = self._zero3_work_remaining()
        stopping_criteria = None
        if zero3_sync is not None and self.tokenizer.eos_token_id is not None:
            stopping_criteria = [
                _GlobalEosStoppingCriteria(attention_mask.shape[1], self.tokenizer.eos_token_id, zero3_sync)
            ]
        kwargs = {
            "attention_mask": attention_mask,
            "max_new_tokens": sampling.max_new_tokens,
            "eos_token_id": None if zero3_sync is not None else self.tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria,
            # The group-aware criterion already synchronizes ZeRO-3 parameter gathers.
            "synced_gpus": False if zero3_sync is not None else None,
            **self._hf_sampling_kwargs(sampling),
            "pad_token_id": pad_token_id,
        }
        if module is not None:
            prepare = getattr(module, "_prepare_generation_config", None)
            if callable(prepare) and "use_model_defaults" in signature(prepare).parameters:
                kwargs["use_model_defaults"] = False
        return kwargs

    def _generate_batched(self, request, sampling, mask_padding=True):
        device = request.prompt_ids.device
        B = request.prompt_ids.shape[0]
        n = sampling.n_samples_per_prompt
        total = B * n
        prompt_len = request.prompt_ids.shape[1]
        max_new_tokens = sampling.max_new_tokens
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("The tokenizer must define pad_token_id or eos_token_id")

        module = self.engine.module

        if self.enable_profiling:
            accelerator = get_accelerator()
            accelerator.synchronize()
            profile_start = time.perf_counter()

        # Expand prompts for n samples per prompt
        if n > 1:
            prompt_ids = request.prompt_ids.repeat_interleave(n, dim=0)
            prompt_attn = request.prompt_attention_mask.repeat_interleave(n, dim=0)
        else:
            prompt_ids = request.prompt_ids
            prompt_attn = request.prompt_attention_mask

        # The injected decode-attention kernel attends every cached slot up
        # to write_pos unconditionally, so padded prompts must ride the SDPA
        # path (which masks pad positions natively). One host sync per
        # generate call; the decode loop itself stays sync-free.
        module._ki_prompt_padded = bool((prompt_attn == 0).any().item())

        if self.enable_profiling:
            accelerator.synchronize()
            expansion_end = time.perf_counter()

        # Master's graph-capture gating (#8980 + #8939): the local variable
        # is the config value narrowed by the accelerator capability check.
        # No is_greedy gate — #8939 added sampling to the graph path; the
        # token selector runs outside the captured graph (~0.9% decode cost).
        use_graph_capture = self.use_graph_capture
        if use_graph_capture and not get_accelerator().is_graph_capture_supported():
            # Same opt-in as the HybridEngine graph cache and the continuous
            # rollout path: falling back keeps backends that only expose a
            # graph API on the eager path.
            logger.warning_once("HybridEngineRollout: graph capture is not supported on the current accelerator; "
                                "falling back to eager generation.")
            use_graph_capture = False

        forward_profiler = None
        forward_profile_handles = []
        if self.enable_profiling and not use_graph_capture:
            forward_profiler = _ForwardProfiler(accelerator)
            forward_profile_handles = forward_profiler.register(module)

        shared_prefill_handles = []
        try:
            if self.use_shared_prefill and n > 1:
                if self.use_graph_capture:
                    raise RuntimeError("Shared prefill does not support graph capture")
                self.engine.prepare_shared_prefill(B, n, prompt_len)
                shared_prefill_handles = self._register_shared_prefill_hooks(module, B, n)
            # Padded prompts fall back to module.generate ONLY when our
            # kernels cannot honor the mask: hybrid prefill feeds GDN a
            # None mask (pad positions corrupt conv-state init), and the
            # injected segKI decode kernels attend every cached slot
            # unconditionally. Standard graph users keep master's padded
            # behavior — the mask flows through prefill and the decode
            # static_attn buffer.
            # mock modules in tests carry no config — default to standard
            _cfg = getattr(module, "config", None)
            layer_types = getattr(getattr(_cfg, "text_config", _cfg), "layer_types", None) or []
            needs_pad_fallback = module._ki_prompt_padded and ("linear_attention" in layer_types or self.use_segki)
            if use_graph_capture and not needs_pad_fallback:
                output_ids = self._generate_graph(prompt_ids, prompt_attn, max_new_tokens, pad_token_id, module,
                                                  device, sampling)
            elif (not needs_pad_fallback and self._chunked_prefill_enabled(sampling)
                  and prompt_len > self.prefill_chunk_size
                  and getattr(module, "_supports_cache_class", None) is not False):
                output_ids = self._generate_from_staged_prefill(prompt_ids, prompt_attn, max_new_tokens, pad_token_id,
                                                                module, device)
            else:
                generate_kwargs = self._generation_kwargs(sampling, prompt_attn, pad_token_id,
                                                          module if self.adaptive_prefill else None)
                output_ids = module.generate(prompt_ids, **generate_kwargs)
        finally:
            for handle in shared_prefill_handles:
                handle.remove()
            for handle in forward_profile_handles:
                handle.remove()

        if self.enable_profiling:
            accelerator.synchronize()
            generation_end = time.perf_counter()

        # ZeRO-3 generation ignores EOS in the decode loop so every rank runs the same
        # number of parameter-gather collectives. Restore the usual generation semantics
        # before returning: retain the first EOS in each response and replace every later
        # token with padding.
        output_ids, response_attn = self._pad_after_eos(
            output_ids,
            response_start=prompt_len,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            mask_padding=mask_padding,
        )

        # Build attention mask: pad positions (both left padding from prompt
        # and right padding from EOS / shorter sequences) are 0.
        response_start = prompt_len
        attention_mask = (output_ids != pad_token_id).long()
        attention_mask[:, response_start:] = response_attn
        for i in range(total):
            prompt_valid = request.prompt_attention_mask[i // n if B > 1 else 0]
            attention_mask[i, :prompt_len] = prompt_valid

        rollout_batch = RolloutBatch(
            input_ids=output_ids,
            attention_mask=attention_mask,
            response_start_idx=torch.full((total, ), response_start, dtype=torch.long, device=device),
        )

        if self.enable_profiling:
            accelerator.synchronize()
            post_processing_end = time.perf_counter()
            prompt_expansion_ms = (expansion_end - profile_start) * 1000.0
            generation_ms = (generation_end - expansion_end) * 1000.0
            post_processing_ms = (post_processing_end - generation_end) * 1000.0
            total_ms = (post_processing_end - profile_start) * 1000.0
            prefill_forward_ms = None
            decode_forward_ms = None
            num_decode_forwards = 0
            if forward_profiler is not None:
                prefill_forward_ms, decode_forward_ms, num_decode_forwards = forward_profiler.get_stage_times()
            generation_overhead_ms = generation_ms
            if prefill_forward_ms is not None:
                generation_overhead_ms -= prefill_forward_ms + decode_forward_ms
                generation_overhead_ms = max(0.0, generation_overhead_ms)
            response_length = int(output_ids.shape[1] - prompt_len)
            num_generated_tokens = int(output_ids.shape[0] * response_length)
            tokens_per_second = 0.0
            if total_ms > 0.0:
                tokens_per_second = num_generated_tokens / (total_ms / 1000.0)
            self._last_profile = {
                "prompt_expansion_ms": prompt_expansion_ms,
                "generation_ms": generation_ms,
                "prefill_forward_ms": prefill_forward_ms,
                "decode_forward_ms": decode_forward_ms,
                "generation_overhead_ms": generation_overhead_ms,
                "num_decode_forwards": num_decode_forwards,
                "post_processing_ms": post_processing_ms,
                "total_ms": total_ms,
                "num_generated_tokens": num_generated_tokens,
                "tokens_per_second": tokens_per_second,
                "batch_size": B,
                "num_samples_per_prompt": n,
                "prompt_length": prompt_len,
                "response_length": response_length,
            }

        return rollout_batch

    @torch.no_grad()
    def _generate_continuous(self, request, sampling, max_batch_size):
        """Generate independent greedy requests in an experimental continuous batch.

        Continuous batching currently requires one prompt row per request and
        greedy decoding. Completed rows retire immediately and pending prompts
        prefill into the released rows before the next decode step.
        """
        select_next_tokens = self._token_selector(sampling)
        profile = self._start_continuous_profile() if self.enable_profiling else None
        original_request = request
        batch_size = request.prompt_ids.shape[0]
        effective_lengths = self._validate_continuous_inputs(request, sampling, max_batch_size)
        use_graph_capture = self.use_graph_capture
        if use_graph_capture and not get_accelerator().is_graph_capture_supported():
            # The eager continuous path is a full fallback, so keep it instead
            # of aborting when the backend has not opted into graph capture.
            logger.warning_once("HybridEngineRollout: graph capture is not supported on the current accelerator; "
                                "falling back to eager continuous batching.")
            use_graph_capture = False
        module = self.engine.module
        prepared_generation_config = None
        if self.adaptive_prefill and dist.is_initialized() and dist.get_world_size() > 1:
            raise ValueError("adaptive prefill currently supports single-process rollouts only")
        align_decode_fronts = self.adaptive_prefill or self.align_decode_fronts
        if self.adaptive_prefill:
            generation_config = getattr(module, "generation_config", None)
            if generation_config is not None:
                from transformers import GenerationConfig, GenerationMixin, PretrainedConfig
                # HF can refresh untouched defaults from legacy model settings during generate().
                prepare_config = getattr(module, "_prepare_generation_config", None)
                if callable(prepare_config):
                    native_prepare = getattr(prepare_config, "__func__",
                                             None) is GenerationMixin._prepare_generation_config
                    needs_prepare = not native_prepare or getattr(generation_config, "cache_implementation",
                                                                  None) == "hybrid"
                    if getattr(generation_config, "_from_model_config", False):
                        legacy_settings = getattr(module.config, "_get_non_default_generation_parameters", None)
                        needs_prepare = needs_prepare or not callable(legacy_settings) or bool(legacy_settings())
                    if needs_prepare:
                        pad_token_id = self.tokenizer.pad_token_id
                        if pad_token_id is None:
                            pad_token_id = self.tokenizer.eos_token_id
                        overrides = self._generation_kwargs(sampling, request.prompt_attention_mask, pad_token_id,
                                                            module)
                        # generate() consumes these arguments before preparing its generation config.
                        overrides.pop("stopping_criteria")
                        overrides.pop("synced_gpus")
                        request_mask = overrides["attention_mask"]
                        mask_snapshot = request_mask.clone()
                        generation_config, model_kwargs = prepare_config(None, **overrides)
                        # generate() would use a replaced, edited or dropped mask, while CB routes plan
                        # from the request mask.
                        prepared_mask = model_kwargs.pop("attention_mask", None)
                        mask_replaced = prepared_mask is not request_mask
                        mask_edited = not mask_replaced and not torch.equal(request_mask, mask_snapshot)
                        if model_kwargs or mask_replaced or mask_edited:
                            raise ValueError("adaptive prefill does not support custom generation model kwargs")
                        # CB routes read these settings from the request, while generate() reruns the preparer.
                        for name, value in overrides.items():
                            if hasattr(generation_config, name) and getattr(generation_config, name) != value:
                                raise ValueError(f"adaptive prefill does not support a generation preparer that "
                                                 f"changes {name}")
                    prepared_generation_config = generation_config
                if self._generation_defaults is None:
                    defaults = GenerationConfig.from_model_config(PretrainedConfig())
                    get_defaults = getattr(defaults, "_get_default_generation_params", None)
                    if callable(get_defaults):
                        for name, value in get_defaults().items():
                            if getattr(defaults, name) is None:
                                setattr(defaults, name, value)
                    self._generation_defaults = defaults
                defaults = self._generation_defaults
                supported_settings = {
                    "max_length", "max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "bos_token_id",
                    "eos_token_id", "pad_token_id", "repetition_penalty", "transformers_version"
                }
                # CB implements repetition penalty; other HF processors can change tokens across routes.
                uses_global_defaults = hasattr(defaults, "_get_default_generation_params")
                for name, value in vars(generation_config).items():
                    if name.startswith("_") or name in supported_settings:
                        continue
                    if uses_global_defaults and value is None:
                        value = getattr(defaults, name, None)
                    if not hasattr(defaults, name) or value != getattr(defaults, name):
                        raise ValueError(f"adaptive prefill does not support non-default generation setting: {name}")
        if align_decode_fronts:
            prompt_lengths = dict(enumerate(map(int, effective_lengths)))
        else:
            prompt_lengths = dict.fromkeys(range(batch_size), request.prompt_ids.shape[1])
        prompt_len = max(prompt_lengths.values())
        if self.adaptive_prefill and self.prefill_max_tokens is not None and prompt_len > self.prefill_max_tokens:
            raise ValueError("a prompt exceeds the prefill token limit; increase the configured limit")
        if use_graph_capture:
            requests = tuple(
                RolloutRequest(request.prompt_ids[index:index + 1], request.prompt_attention_mask[index:index + 1])
                for index in range(batch_size))
            return self._generate_continuous_graph(original_request,
                                                   requests,
                                                   sampling,
                                                   max_batch_size,
                                                   prompt_lengths=prompt_lengths,
                                                   generation_config=prepared_generation_config,
                                                   profile=profile)

        max_positions = getattr(module.config, "max_position_embeddings", None)
        if max_positions is not None:
            if any(length + sampling.max_new_tokens > max_positions for length in prompt_lengths.values()):
                raise ValueError("continuous batching request exceeds the model maximum position embeddings")
        if self.continuous_cache_capacity is not None:
            if self.continuous_cache_capacity < prompt_len:
                raise ValueError("continuous_cache_capacity must be at least the padded prompt length")
            max_cache_len = self.continuous_cache_capacity
        else:
            max_cache_len = (prompt_len +
                             sampling.max_new_tokens if align_decode_fronts else self._estimate_continuous_cache_len(
                                 prompt_len, [sampling.max_new_tokens] * batch_size, max_batch_size))
        if max_positions is not None and max_cache_len > max_positions:
            raise ValueError("continuous batching cache exceeds the model maximum position embeddings")
        if getattr(module, "_supports_cache_class", None) is False:
            raise ValueError("continuous batching requires a model with cache-class support; use the default "
                             "generate() path or upgrade transformers")

        use_batched = False
        if self.adaptive_prefill:
            full_width = request.prompt_ids.shape[1]
            full_tokens_fit = self.prefill_max_tokens is None or request.prompt_ids.numel() <= self.prefill_max_tokens
            full_positions_fit = max_positions is None or full_width + sampling.max_new_tokens <= max_positions
            if (batch_size <= max_batch_size and full_tokens_fit and full_positions_fit
                    and self.continuous_cache_capacity is None and not self.enable_cache_trimming):
                prefill_cost = self._prefill_cost_function(module)
                batched_cost = prefill_cost(batch_size, full_width, include_kv=False)
                decode_cost = self.continuous_decode_cost_ms * (sampling.max_new_tokens - 1)
                # Every partition pays at least one Forward plus each request's unpadded work.
                # If ordinary generation already beats that bound, there is no reason to run DP.
                if min(prompt_lengths.values()) == prompt_len:
                    lower_bound = prefill_cost(batch_size, prompt_len)
                else:
                    lower_bound = self.prefill_fixed_cost_ms + sum(
                        prefill_cost(1, length) - self.prefill_fixed_cost_ms for length in prompt_lengths.values())
                use_batched = batched_cost <= lower_bound + decode_cost
                if not use_batched:
                    _, bucket_cost = plan_prefill_buckets(list(prompt_lengths.values()), prefill_cost,
                                                          self.prefill_max_tokens)
                    use_batched = batched_cost <= bucket_cost + decode_cost
        if use_batched:
            # A bounded rollout with no pending rows can use the existing batch path.
            output = self._generate_batched(request, replace(sampling, continuous_batch_size=None), mask_padding=False)
            self._last_continuous_stats = None
            if self.enable_profiling:
                total_ms = (self._profile_start(profile) - profile["start"]) * 1000
                profile = self._last_profile
                routing_ms = total_ms - profile["total_ms"]
                profile["generation_ms"] += routing_ms
                profile["generation_overhead_ms"] += routing_ms
                profile["total_ms"] = total_ms
                profile["generation_strategy"] = "batched"
                generated_tokens = int(output.attention_mask[:, request.prompt_ids.shape[1]:].sum().item())
                profile["num_generated_tokens"] = generated_tokens
                total_ms = profile["total_ms"]
                profile["tokens_per_second"] = generated_tokens * 1000 / total_ms if total_ms > 0 else 0.0
                profile["active_batch_size"] = batch_size
                profile["continuous_batch_size"] = max_batch_size
            return output

        if self.enable_profiling and profile is None:
            profile = self._start_continuous_profile()
        profile_accelerator = profile["accelerator"] if profile is not None else None
        profile_start = profile["start"] if profile is not None else None

        from transformers import StaticCache
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        requests = tuple(
            RolloutRequest(request.prompt_ids[index:index + 1], request.prompt_attention_mask[index:index + 1])
            for index in range(batch_size))
        device = requests[0].prompt_ids.device
        model_dtype = next(module.parameters()).dtype
        select_next_tokens = self._token_selector(sampling)
        repetition_penalty = self._repetition_penalty(module)

        scheduler_start = self._profile_start(profile)
        scheduler = ContinuousBatchScheduler(max_batch_size, sampling.max_new_tokens)
        request_by_id = {}
        responses = {}
        request_order = (sorted(range(len(requests)), key=lambda request_id: -prompt_lengths[request_id])
                         if align_decode_fronts else range(len(requests)))
        for request_id in request_order:
            request = requests[request_id]
            scheduler.submit(ContinuousBatchRequest(request_id))
            request_by_id[request_id] = request
            responses[request_id] = []
        self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)

        cache_start = self._profile_start(profile)
        # Layers are sized by the first prefill because AutoTP shards KV heads per rank.
        cache = DeepSpeedStaticCache(module.config, max_cache_len=max_cache_len)
        write_positions = torch.full((max_batch_size, ), -1, dtype=torch.long, device=device)
        cache.set_write_position(write_positions)
        attention_mask = torch.zeros((max_batch_size, max_cache_len), dtype=torch.long, device=device)
        stats = {
            "cache_capacity": max_cache_len,
            "peak_cache_length": prompt_len,
            "trim_count": 0,
            "trimmed_columns": 0,
            "trim_latency_ms": 0.0,
            "decode_steps": 0,
        }
        next_tokens = {}
        logical_positions = dict(prompt_lengths)
        cache_position = prompt_len
        span_starts = [0] * max_batch_size
        prefill_chunks = {}
        chunk_stats = {"chunks": 0, "peak_staging_bytes": 0}
        chunk_size = self.prefill_chunk_size if self._chunked_prefill_enabled(sampling) else 0

        self._profile_end(profile, "cache_management_overhead_ms", cache_start)
        scheduler_start = self._profile_start(profile)
        update = scheduler.schedule()
        self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)

        while update.active:
            if profile is not None:
                profile["active_batch_sizes"].append(len(update.active))
            cache_start = self._profile_start(profile)
            keep_slots = torch.tensor(update.keep_slots, dtype=torch.long, device=device)
            survivor_count = keep_slots.numel()
            if survivor_count:
                if update.retired:
                    cache.compact(keep_slots)
                    # Compaction moves surviving rows down, so a staged prompt's recorded slot
                    # has to move with its data or it would keep writing to another row.
                    if prefill_chunks:
                        row_map = {old: new for new, old in enumerate(keep_slots.tolist())}
                        for chunk in prefill_chunks.values():
                            chunk.slot = row_map[chunk.slot]
                    identity = torch.arange(survivor_count, dtype=torch.long, device=device)
                    if not torch.equal(keep_slots, identity):
                        survivor_attention = attention_mask.index_select(0, keep_slots).clone()
                        attention_mask[:survivor_count].copy_(survivor_attention)
                    attention_mask[survivor_count:].zero_()
                    if align_decode_fronts:
                        span_starts[:survivor_count] = [span_starts[index] for index in update.keep_slots]
                        span_starts[survivor_count:] = [0] * (max_batch_size - survivor_count)

                dead_prefix = 0
                if self.enable_cache_trimming:
                    if align_decode_fronts:
                        dead_prefix = min(span_starts[:survivor_count])
                    else:
                        trim_threshold = max(1, prompt_len)
                        dead_prefix = self._continuous_dead_prefix(attention_mask, survivor_count)
                        if dead_prefix < trim_threshold and cache_position < max_cache_len - trim_threshold:
                            dead_prefix = 0
                elif cache_position >= max_cache_len:
                    if align_decode_fronts:
                        dead_prefix = min(span_starts[:survivor_count])
                    else:
                        dead_prefix = self._continuous_dead_prefix(attention_mask, survivor_count)
                if dead_prefix:
                    if update.admitted:
                        longest_admitted = max(prompt_lengths[request.request_id] for request in update.admitted)
                        dead_prefix = min(dead_prefix, max(0, cache_position - longest_admitted))
                    trim_start = None
                    if profile_accelerator is not None:
                        profile_accelerator.synchronize()
                        trim_start = time.perf_counter()
                    moved_bytes = sum((layer.keys[:, :, dead_prefix:, :].numel() * layer.keys.element_size() +
                                       layer.values[:, :, dead_prefix:, :].numel() * layer.values.element_size())
                                      for layer in cache.layers)
                    moved_bytes += attention_mask[:, dead_prefix:].numel() * attention_mask.element_size()
                    cache.trim_left(dead_prefix)
                    remaining_columns = max_cache_len - dead_prefix
                    if remaining_columns:
                        attention_mask[:, :remaining_columns].copy_(attention_mask[:, dead_prefix:].clone())
                    attention_mask[:, remaining_columns:].zero_()
                    write_positions[:survivor_count].sub_(dead_prefix)
                    cache_position -= dead_prefix
                    stats["trim_count"] += 1
                    stats["trimmed_columns"] += dead_prefix
                    stats["trim_bytes_moved"] = stats.get("trim_bytes_moved", 0) + moved_bytes
                    if profile_accelerator is not None:
                        profile_accelerator.synchronize()
                        stats["trim_latency_ms"] += (time.perf_counter() - trim_start) * 1000.0
                    if align_decode_fronts:
                        span_starts[:survivor_count] = [start - dead_prefix for start in span_starts[:survivor_count]]
            else:
                cache.reset()
                write_positions.fill_(-1)
                attention_mask.zero_()
                cache_position = prompt_len
                span_starts = [0] * max_batch_size

            self._profile_end(profile, "cache_management_overhead_ms", cache_start)
            if self.align_decode_fronts and survivor_count and cache_position >= max_cache_len:
                raise self._continuous_cache_exhaustion_error(max_cache_len)
            admitted_tokens = self._continuous_prefill(
                module,
                StaticCache,
                cache,
                update,
                request_by_id,
                attention_mask,
                write_positions,
                cache_position,
                prompt_lengths,
                model_dtype,
                device,
                sampling,
                align_decode_fronts,
                profile,
                generation_config=prepared_generation_config,
                prefill_chunks=prefill_chunks,
                chunk_size=chunk_size,
            )
            published_ids = set()
            if prefill_chunks:
                published = self._run_chunked_prefill(module, prefill_chunks, chunk_size, device, cache,
                                                      attention_mask, write_positions, chunk_stats, profile)
                admitted_tokens.update(published)
                # A prompt that completed in this step starts decoding next step, exactly
                # like a freshly admitted row: this step's token is the first response token.
                published_ids = set(published)
            if align_decode_fronts:
                for admitted, target_row in zip(update.admitted, update.admitted_slots):
                    span_starts[target_row] = cache_position - prompt_lengths[admitted.request_id]

            decoded_tokens = {}
            active_ids = update.active_ids
            # Only rows that were already generating decode: this step's admissions and rows
            # still staging their prompt hold a slot but own no decode column yet. Each row
            # writes to its own column, so a row that joins the batch later does not push the
            # others forward.
            admitted_ids = set(request.request_id for request in update.admitted)
            decodable_ids = tuple(request_id for request_id in active_ids if request_id not in admitted_ids
                                  and request_id not in prefill_chunks and request_id not in published_ids)
            if decodable_ids:
                active_count = len(active_ids)
                decodable_rows = [row for row, request_id in enumerate(active_ids) if request_id in set(decodable_ids)]
                staged_rows = [row for row, request_id in enumerate(active_ids) if request_id in prefill_chunks]
                dummy_token = torch.zeros(1, 1, dtype=torch.long, device=device)
                decode_input = torch.cat([next_tokens.get(request_id, dummy_token) for request_id in active_ids],
                                         dim=0)
                row_positions = write_positions[:active_count]
                if align_decode_fronts:
                    # The aligned layout packs every survivor behind one front and re-derives
                    # their positions from it each step; chunked prefill does not combine with it.
                    row_positions[:survivor_count].fill_(cache_position)
                    if cache_position >= max_cache_len:
                        raise self._continuous_cache_exhaustion_error(max_cache_len)
                    attention_mask[:survivor_count, cache_position] = 1
                    decode_cache_position = torch.tensor([cache_position], dtype=torch.long, device=device)
                else:
                    # Every row advances its own column, so a row that joins the batch later
                    # neither pushes the others forward nor inherits their position.
                    if staged_rows:
                        row_positions[staged_rows] = max_cache_len - 1
                    latest_position = row_positions[decodable_rows].max()
                    if int(latest_position) >= max_cache_len:
                        raise self._continuous_cache_exhaustion_error(max_cache_len)
                    decode_cache_position = latest_position.reshape(1)
                if align_decode_fronts:
                    position_ids = torch.tensor([logical_positions[request_id] for request_id in active_ids],
                                                dtype=torch.long,
                                                device=device).unsqueeze(1)
                else:
                    # A row's next token sits one column past the valid columns it already
                    # holds, so the count is the token's position; the row's own pending
                    # column is revealed after the count, like the shared-front path does.
                    position_ids = attention_mask[:active_count].sum(dim=1, keepdim=True)
                    attention_mask[decodable_rows, row_positions[decodable_rows]] = 1
                decode_start = self._profile_start(profile)
                output = self._call_model(
                    module,
                    decode_input,
                    attention_mask=attention_mask[:active_count],
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=decode_cache_position,
                    position_ids=position_ids,
                )
                self._profile_end(profile, "decode_forward_ms", decode_start, count="num_decode_forwards")
                decoded = self._continuous_next_tokens(
                    output.logits[decodable_rows, -1, :],
                    decodable_ids,
                    request_by_id,
                    responses,
                    module,
                    select_next_tokens,
                    generation_config=prepared_generation_config,
                )
                stats["decode_steps"] += 1
                decoded_tokens = dict(zip(decodable_ids, decoded.split(1, dim=0)))
                if not self.align_decode_fronts:
                    row_positions[decodable_rows] += 1
                for request_id in decodable_ids:
                    logical_positions[request_id] += 1

            next_tokens = decoded_tokens | admitted_tokens
            finished_ids = []
            for request_id in update.active_ids:
                token = next_tokens.get(request_id)
                if token is None:
                    # Still staging its prompt in chunks: nothing to append or retire yet.
                    continue
                responses[request_id].append(token)
                if self._is_eos(token):
                    finished_ids.append(request_id)

            if decodable_ids:
                cache_position += 1
            stats["peak_cache_length"] = max(stats["peak_cache_length"], cache_position)
            scheduler_start = self._profile_start(profile)
            update = scheduler.advance(finished_ids, prefilling_ids=tuple(prefill_chunks))
            self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)

        stats["cache_memory_bytes"] = _cache_bytes(cache) + attention_mask.nbytes + write_positions.nbytes
        generation_end = self._profile_start(profile)
        post_processing_start = generation_end
        output = self._build_continuous_batch(original_request, responses)
        post_processing_end = self._profile_end(profile, None, post_processing_start)
        if profile_accelerator is not None:
            profile_accelerator.synchronize()
            total_ms = (time.perf_counter() - profile_start) * 1000.0
            generated_tokens = sum(len(response) for response in responses.values())
            stats["end_to_end_ms"] = total_ms
            stats["tokens_per_second"] = generated_tokens / (total_ms / 1000.0) if total_ms > 0.0 else 0.0
        else:
            stats["end_to_end_ms"] = None
            stats["tokens_per_second"] = None
        stats.setdefault("trim_bytes_moved", 0)
        stats["trim_frequency"] = (stats["trim_count"] / stats["decode_steps"] if stats["decode_steps"] else 0.0)
        stats["chunked_prefill_chunks"] = chunk_stats["chunks"]
        stats["chunked_prefill_staging_bytes"] = chunk_stats["peak_staging_bytes"]
        self._last_continuous_stats = stats
        if profile is not None:
            self._finish_continuous_profile(profile, original_request, responses, max_batch_size, prompt_len,
                                            generation_end, post_processing_end)
            self._last_profile["generation_strategy"] = ("bucketed" if self.adaptive_prefill else
                                                         "aligned" if align_decode_fronts else "equal_width")
        return output

    @torch.no_grad()
    def _generate_continuous_graph(self, original_request, requests, sampling, max_batch_size, *, prompt_lengths,
                                   generation_config, profile):
        """Run continuous decoding with a fixed-capacity graph.

        Scheduler slots stay attached to physical cache rows for their whole
        lifetime. Retired rows are masked until a new prompt is prefetched into
        them, so the captured decode always receives the same batch shape and
        static tensor addresses.
        """
        module = self.engine.module
        prompt_len = requests[0].prompt_ids.shape[1]
        self._validate_continuous_graph_options(prompt_len, sampling)
        if getattr(module, "_supports_cache_class", None) is False:
            raise ValueError("continuous batching requires a model with cache-class support; use the default "
                             "generate() path or upgrade transformers")
        if getattr(module.config, "_attn_implementation", None) == "flash_attention_2":
            raise ValueError("continuous graph capture does not support flash_attention_2")
        device = requests[0].prompt_ids.device
        if get_accelerator().create_graph() is None:
            raise ValueError("continuous graph capture is not supported on the current accelerator")

        from transformers import StaticCache
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        if self.enable_profiling and profile is None:
            profile = self._start_continuous_profile()
        profile_accelerator = profile["accelerator"] if profile is not None else None
        profile_start = profile["start"] if profile is not None else None
        minimum_cache_len = prompt_len + sampling.max_new_tokens
        max_cache_len = (self.continuous_cache_capacity
                         if self.continuous_cache_capacity is not None else minimum_cache_len)
        max_positions = getattr(module.config, "max_position_embeddings", None)
        if max_positions is not None and max_cache_len > max_positions:
            raise ValueError("continuous batching cache exceeds the model maximum position embeddings")
        model_dtype = next(module.parameters()).dtype
        select_next_tokens = self._token_selector(sampling)
        repetition_penalty = self._repetition_penalty(module)
        scheduler = ContinuousBatchScheduler(max_batch_size, sampling.max_new_tokens)
        request_by_id = {}
        responses = {}
        for request_id, request in enumerate(requests):
            scheduler.submit(ContinuousBatchRequest(request_id))
            request_by_id[request_id] = request
            responses[request_id] = []

        chunk_size = self.prefill_chunk_size if self._chunked_prefill_enabled(sampling) else 0
        state = self._get_continuous_graph_state(module, max_batch_size, max_cache_len, device, model_dtype,
                                                 DeepSpeedStaticCache)
        if chunk_size > 0:
            # The staging cache holds one padded prompt per row, so it is sized to this
            # batch's padded prompt width. The graph state itself is reused across prompt
            # widths that fit the decode capacity, so the staging cache and its captured
            # prefill buckets are rebuilt when the width changes.
            staging_width = state.staging_cache.get_max_cache_shape() if state.staging_cache is not None else None
            if staging_width != prompt_len:
                state.staging_cache = self._create_static_cache(DeepSpeedStaticCache, module.config, max_batch_size,
                                                                prompt_len, device, model_dtype)
                state.prefill.clear()
        cache = state.cache
        static_input = state.static_input
        static_attention = state.static_attention
        static_causal_attention = state.static_causal_attention
        static_position_ids = state.static_position_ids
        static_write_positions = state.static_write_positions
        staging_cache = state.staging_cache
        staging_bytes = _cache_bytes(staging_cache)
        captures_before = state.captures
        stats = {
            "cache_capacity": max_cache_len,
            "peak_cache_length": prompt_len,
            "cache_memory_bytes": _cache_bytes(cache) + static_attention.nbytes + static_write_positions.nbytes,
            "trim_count": 0,
            "trimmed_columns": 0,
            "trim_latency_ms": 0.0,
            "trim_bytes_moved": 0,
            "decode_steps": 0,
            "graph_capture_ladder": list(state.ladder),
            "graph_captures": 0,
            "decode_bucket_steps": 0,
            "decode_padded_rows": 0,
            "decode_bucket_widths": {},
            "prefill_bucket_steps": 0,
            "prefill_padded_rows": 0,
            "prefill_bucket_widths": {},
            "chunked_prefill_chunks": 0,
            "chunked_prefill_staging_bytes": staging_bytes,
        }

        slot_by_request = {}
        write_positions = {}
        next_tokens = {}
        prefill_chunks = {}
        chunk_stats = {"chunks": 0, "peak_staging_bytes": 0}
        update = scheduler.schedule()
        while update.active:
            if profile is not None:
                profile["active_batch_sizes"].append(len(update.active))

            for request_id in update.retired:
                static_attention[slot_by_request[request_id]].zero_()
                slot_by_request.pop(request_id, None)
                write_positions.pop(request_id, None)
                next_tokens.pop(request_id, None)
                # A staged prompt that retires drops out of its wave; its staging row stays
                # parked behind an all-masked row until the wave ends.
                prefill_chunks.pop(request_id, None)

            free_slots = [slot for slot in range(max_batch_size) if slot not in slot_by_request.values()]
            admitted_ids = tuple(request.request_id for request in update.admitted)
            admitted_slots = []
            for request_id in admitted_ids:
                slot = free_slots.pop(0)
                slot_by_request[request_id] = slot
                write_positions[request_id] = prompt_len
                admitted_slots.append(slot)

            if admitted_ids:
                # A full-width admission that fits the token limit is already the optimal single bucket.
                bucket_prefill = self.adaptive_prefill and (
                    any(prompt_lengths[i] != prompt_len for i in admitted_ids) or
                    (self.prefill_max_tokens is not None and len(admitted_ids) * prompt_len > self.prefill_max_tokens))
                admitted_tokens = self._continuous_prefill(
                    module,
                    StaticCache,
                    cache,
                    update,
                    request_by_id,
                    static_attention,
                    static_write_positions,
                    prompt_len,
                    prompt_lengths,
                    model_dtype,
                    device,
                    sampling,
                    profile=profile,
                    align_decode_fronts=bucket_prefill,
                    target_slots=tuple(admitted_slots),
                    generation_config=generation_config,
                    prefill_chunks=prefill_chunks,
                    chunk_size=chunk_size,
                    stage_prompts=self._stage_staged_prompt,
                )
                next_tokens.update(admitted_tokens)

            published_ids = set()
            if prefill_chunks:
                chunked_tokens = self._run_staged_prefill(module,
                                                          prefill_chunks,
                                                          chunk_size,
                                                          prompt_len,
                                                          cache,
                                                          static_attention,
                                                          static_write_positions,
                                                          chunk_stats,
                                                          profile,
                                                          request_write_positions=write_positions)
                next_tokens.update(chunked_tokens)
                # A prompt that completed in this step starts decoding next step, exactly
                # like a freshly admitted row: this step's token is the first response token
                # and must not be overwritten by a decode of the same row.
                published_ids = set(chunked_tokens)

            decode_ids = tuple(request_id for request_id in update.active_ids if request_id not in admitted_ids
                               and request_id not in prefill_chunks and request_id not in published_ids)
            static_input.zero_()
            static_position_ids.zero_()
            # Captured replay always executes every physical row. Keep non-decoding
            # rows away from their prefetched prompt KV until they become active.
            static_write_positions.fill_(max_cache_len - 1)
            for request_id in decode_ids:
                slot = slot_by_request[request_id]
                position = write_positions[request_id]
                static_input[slot:slot + 1].copy_(next_tokens[request_id])
                static_position_ids[slot, 0] = static_attention[slot, :position].sum()
                static_attention[slot, position] = 1
                static_write_positions[slot] = position

            decoded_tokens = {}
            if decode_ids:
                self._update_static_causal_attention(static_attention, static_causal_attention)
                bucket = self._decode_bucket(max(slot_by_request[request_id] for request_id in decode_ids) + 1)
                self._capture_bucket_if_needed(module, cache, bucket, state.static_cache_position,
                                               bucket.write_positions)
                decode_start = self._profile_start(profile)
                decode_logits_all = self._run_bucket(module, cache, bucket, state.static_cache_position,
                                                     bucket.write_positions)
                self._profile_end(profile, "decode_forward_ms", decode_start, count="num_decode_forwards")
                decode_logits = torch.cat(
                    [decode_logits_all[slot_by_request[request_id], -1, :].unsqueeze(0) for request_id in decode_ids],
                    dim=0)
                decoded = self._continuous_next_tokens(
                    decode_logits,
                    decode_ids,
                    request_by_id,
                    responses,
                    module,
                    select_next_tokens,
                    generation_config=generation_config,
                )
                for request_id, token in zip(decode_ids, decoded.split(1, dim=0)):
                    decoded_tokens[request_id] = token
                    write_positions[request_id] += 1
                stats["decode_steps"] += 1
                stats["decode_bucket_steps"] += 1
                stats["decode_padded_rows"] += bucket.width - len(decode_ids)
                stats["decode_bucket_widths"][bucket.width] = stats["decode_bucket_widths"].get(bucket.width, 0) + 1
                next_tokens.update(decoded_tokens)

            finished_ids = []
            for request_id in update.active_ids:
                token = next_tokens.get(request_id)
                if token is None:
                    # This row is still prefilling its prompt in chunks: it holds a slot
                    # but has nothing to append or retire in this step.
                    continue
                responses[request_id].append(token)
                if self._is_eos(token):
                    finished_ids.append(request_id)

            scheduler_start = self._profile_start(profile)
            update = scheduler.advance(finished_ids, prefilling_ids=tuple(prefill_chunks))
            self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)
            if write_positions:
                stats["peak_cache_length"] = max(stats["peak_cache_length"], max(write_positions.values()))

        generation_end = self._profile_start(profile)
        output = self._build_continuous_batch(original_request, responses)
        post_processing_end = self._profile_end(profile, None, generation_end)
        if profile_accelerator is not None:
            profile_accelerator.synchronize()
            total_ms = (time.perf_counter() - profile_start) * 1000.0
            generated_tokens = sum(len(response) for response in responses.values())
            stats["end_to_end_ms"] = total_ms
            stats["tokens_per_second"] = generated_tokens / (total_ms / 1000.0) if total_ms > 0.0 else 0.0
        else:
            stats["end_to_end_ms"] = None
            stats["tokens_per_second"] = None
        stats["trim_frequency"] = 0.0
        stats["graph_captures"] = state.captures - captures_before
        stats["chunked_prefill_chunks"] = chunk_stats["chunks"]
        if staging_cache is None:
            stats["chunked_prefill_staging_bytes"] = chunk_stats["peak_staging_bytes"]
        else:
            stats["prefill_bucket_steps"] = chunk_stats.get("graph_steps", 0)
            stats["prefill_padded_rows"] = chunk_stats.get("padded_rows", 0)
            stats["prefill_bucket_widths"] = dict(chunk_stats.get("bucket_widths", {}))
        self._last_continuous_stats = stats
        if profile is not None:
            self._finish_continuous_profile(profile, original_request, responses, max_batch_size, prompt_len,
                                            generation_end, post_processing_end)
            if self.adaptive_prefill:
                self._last_profile["generation_strategy"] = "bucketed_graph"
        return output

    def _validate_continuous_graph_options(self, prompt_len, sampling):
        if self.align_decode_fronts and not self.adaptive_prefill:
            raise ValueError("continuous graph capture does not support aligned decode fronts")
        if self.enable_cache_trimming:
            raise ValueError("continuous graph capture does not support cache trimming")
        if self.enable_chunked_prefill and self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive when chunked prefill is enabled")
        if (self.continuous_cache_capacity is not None
                and self.continuous_cache_capacity < prompt_len + sampling.max_new_tokens):
            raise ValueError("continuous_cache_capacity must fit the prompt and all generated tokens")
        self._graph_capture_ladder(sampling.continuous_batch_size, self.graph_capture_batch_sizes)

    @staticmethod
    def _graph_capture_ladder(max_batch_size, configured=None):
        """Return the batch widths the graph path captures, smallest first.

        The loop buckets every step up to the nearest width in the ladder, so a sparse
        ladder means fewer captures and more padded rows, and a dense one means the
        opposite. ``max_batch_size`` is always included: a full physical batch has to be
        replayable for the step that drains every slot.
        """
        widths = {max_batch_size}
        if configured:
            for width in configured:
                if not isinstance(width, int) or width <= 0:
                    raise ValueError("graph_capture_batch_sizes must be positive integers")
                if width > max_batch_size:
                    raise ValueError("graph_capture_batch_sizes cannot exceed continuous_batch_size")
                widths.add(width)
        else:
            width = 1
            while width < max_batch_size:
                widths.add(width)
                width *= 2
        return tuple(sorted(widths))

    @staticmethod
    def _bucket_width(ladder, required):
        """Smallest captured width that covers ``required`` rows."""
        for width in ladder:
            if width >= required:
                return width
        raise ValueError(f"captured batch ladder {ladder} does not cover {required} rows")

    def _get_continuous_graph_state(self, module, max_batch_size, max_cache_len, device, model_dtype, cache_type):
        ladder = self._graph_capture_ladder(max_batch_size, self.graph_capture_batch_sizes)
        signature = self._continuous_graph_signature(module, max_batch_size, max_cache_len, device, model_dtype,
                                                     ladder)
        state = self._continuous_graph_state
        if state is None or state.signature != signature:
            # Free the stale cache and graph pool before allocating replacements.
            self._continuous_graph_state = None
            state = None
            # Capture sizes the layers from the model's local KV tensors.
            cache = cache_type(module.config, max_cache_len=max_cache_len)
            # One staging cache serves every staged prompt and every captured prefill
            # width: rows are staged prompts and columns are the padded prompt width,
            # which is the same for every request in a continuous batch. It is sized to
            # the padded prompt width, so the caller that knows the width builds it on
            # the state; the decode graph is reused across widths that fit the capacity.
            staging_cache = None
            static_input = torch.zeros((max_batch_size, 1), dtype=torch.long, device=device)
            static_attention = torch.zeros((max_batch_size, max_cache_len), dtype=torch.long, device=device)
            static_causal_attention = torch.zeros((max_batch_size, 1, 1, max_cache_len),
                                                  dtype=model_dtype,
                                                  device=device)
            static_position_ids = torch.zeros((max_batch_size, 1), dtype=torch.long, device=device)
            static_write_positions = torch.zeros(max_batch_size, dtype=torch.long, device=device)
            cache.set_write_position(static_write_positions)
            # Transformers uses cache_position to build the causal mask. Per-row
            # validity is represented by static_attention, so a fixed upper bound
            # lets one captured graph serve requests at different decode lengths.
            static_cache_position = torch.tensor([max_cache_len - 1], dtype=torch.long, device=device)
            state = _ContinuousGraphState(
                signature,
                ladder,
                cache,
                staging_cache,
                static_input,
                static_attention,
                static_causal_attention,
                static_position_ids,
                static_write_positions,
                static_cache_position,
            )
            self._continuous_graph_state = state

        state.cache.reset()
        state.static_input.zero_()
        state.static_attention.zero_()
        state.static_causal_attention.zero_()
        state.static_position_ids.zero_()
        state.static_write_positions.zero_()
        return state

    def _decode_bucket(self, highest_row):
        """Return the decode bucket covering the rows that are decoding, allocating it if new.

        Buckets are captured by the caller once its buffers hold the step it is about to
        run, so a capture is never asked to warm up on stale rows.
        """
        state = self._continuous_graph_state
        width = self._bucket_width(state.ladder, highest_row)
        bucket = state.decode.get(width)
        if bucket is None:
            bucket = _DecodeBucket(width, state.static_input[:width], state.static_causal_attention[:width],
                                   state.static_position_ids[:width], state.static_write_positions[:width])
            state.decode[width] = bucket
        return bucket

    def _prefill_bucket(self, width):
        """Return the chunked-prefill bucket for a staged row count, allocating it if new."""
        state = self._continuous_graph_state
        bucket = state.prefill.get(width)
        if bucket is None:
            device = state.static_input.device
            bucket = _PrefillBucket(
                width,
                torch.zeros((width, self.prefill_chunk_size), dtype=torch.long, device=device),
                torch.zeros((width, 1, self.prefill_chunk_size, state.staging_cache.get_max_cache_shape()),
                            dtype=state.static_causal_attention.dtype,
                            device=device),
                torch.zeros((width, self.prefill_chunk_size), dtype=torch.long, device=device),
                torch.zeros(width, dtype=torch.long, device=device),
            )
            state.prefill[width] = bucket
        return bucket

    def _capture_bucket_if_needed(self, module, cache, bucket, cache_position, write_position):
        """Capture a bucket the first time the loop needs it."""
        if bucket.graph is None and not bucket.disabled:
            self._capture_bucket(module, cache, bucket, cache_position, write_position)
        return bucket

    def _capture_bucket(self, module, cache, bucket, cache_position, write_position):
        """Capture one bucket's forward, falling back to eager forwards if capture fails.

        The capture runs the model for real during its warmup, with the buffers the next
        replay will use, so it must be called only once the caller has filled them.
        """
        try:
            cache.set_write_position(write_position)
            bucket.graph, bucket.logits = self._capture_continuous_graph(module, cache, bucket.input_ids,
                                                                         bucket.causal_attention, write_position,
                                                                         cache_position, bucket.position_ids,
                                                                         self._last_logits_kwargs(module))
            self._continuous_graph_state.captures += 1
        except Exception as error:  # a graph is an optimization: a failed capture must not fail the rollout
            bucket.graph = None
            bucket.disabled = True
            logger.warning(
                "Rollout graph capture failed for batch width %s (%s); this bucket will run eagerly "
                "for the rest of the rollout.", bucket.width, error)

    def _run_bucket(self, module, cache, bucket, cache_position, write_position):
        """Replay a bucket's captured forward, or run the same buffers eagerly without a graph."""
        if bucket.graph is not None:
            get_accelerator().replay_graph(bucket.graph)
            return bucket.logits
        # The eager fallback reads the same static buffers, so it stays token-identical
        # to a replay; only the launch overhead differs.
        cache.set_write_position(write_position)
        output = self._call_model(
            module,
            bucket.input_ids,
            attention_mask=bucket.causal_attention,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            position_ids=bucket.position_ids,
            **self._last_logits_kwargs(module),
        )
        return output.logits

    @staticmethod
    def _continuous_graph_signature(module, max_batch_size, max_cache_len, device, model_dtype, ladder):
        # Replay reads parameters and buffers through the address and layout fixed
        # at capture, so in-place value updates need no recapture, but a moved or
        # re-laid-out tensor does. Prefill runs eagerly, so the prompt width only
        # matters through max_cache_len.
        tensor_signature = tuple((tensor.data_ptr(), tensor.device, tensor.dtype, tuple(tensor.shape), tensor.stride())
                                 for tensor in itertools.chain(module.parameters(), module.buffers()))
        device_index = device.index
        if device_index is None and device.type == get_accelerator().device_name():
            # torch.device("cuda") and torch.device("cuda", 0) name the same device.
            device_index = get_accelerator().current_device()
        return (id(module), device.type, device_index, model_dtype, max_batch_size, max_cache_len, ladder,
                tensor_signature)

    @staticmethod
    def _update_static_causal_attention(static_attention, static_causal_attention):
        static_causal_attention.copy_(static_attention[:, None, None, :])
        static_causal_attention.neg_().add_(1).mul_(torch.finfo(static_causal_attention.dtype).min)

    @staticmethod
    def _capture_continuous_graph(module,
                                  cache,
                                  static_input,
                                  static_attention,
                                  static_write_positions,
                                  static_cache_position,
                                  static_position_ids,
                                  logits_kwargs=None):
        """Capture one fixed-capacity decode forward and return its graph and static logits."""
        logits_kwargs = logits_kwargs or {}
        accelerator = get_accelerator()
        saved_pre = dict(module._forward_pre_hooks)
        saved_post = dict(module._forward_hooks)
        module._forward_pre_hooks.clear()
        module._forward_hooks.clear()
        try:
            if not all(layer.is_initialized for layer in cache.layers):
                # Allocate uninitialized KV layers on the current stream, not the warmup stream.
                module(
                    static_input,
                    attention_mask=static_attention,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=static_cache_position,
                    position_ids=static_position_ids,
                )
            warmup_stream = accelerator.Stream()
            warmup_stream.wait_stream(accelerator.current_stream())
            with accelerator.stream(warmup_stream):
                for _ in range(3):
                    module(
                        static_input,
                        attention_mask=static_attention,
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=static_cache_position,
                        position_ids=static_position_ids,
                        **logits_kwargs,
                    )
            accelerator.current_stream().wait_stream(warmup_stream)
            graph = accelerator.create_graph()
            with accelerator.capture_to_graph(graph):
                output = module(
                    static_input,
                    attention_mask=static_attention,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=static_cache_position,
                    position_ids=static_position_ids,
                    **logits_kwargs,
                )
        finally:
            module._forward_pre_hooks.update(saved_pre)
            module._forward_hooks.update(saved_post)
        return graph, output.logits

    @staticmethod
    def _continuous_cache_exhaustion_error(max_cache_len):
        return ValueError(
            f"continuous batching cache capacity ({max_cache_len}) exhausted before active requests retired; "
            "increase HybridEngineRolloutConfig.continuous_cache_capacity or enable cache trimming with "
            "HybridEngineRolloutConfig(enable_cache_trimming=True)")

    @staticmethod
    def _estimate_continuous_cache_len(prompt_len, max_new_tokens, max_batch_size):
        """Estimate the largest cache span before the scheduler becomes empty."""
        pending = list(max_new_tokens)
        active = []
        max_cache_len = prompt_len
        busy_span = 0

        while active or pending:
            if not active:
                active = pending[:max_batch_size]
                pending = pending[max_batch_size:]
                busy_span = 0

            busy_span += 1
            max_cache_len = max(max_cache_len, prompt_len + busy_span)
            survivors = [remaining - 1 for remaining in active if remaining > 1]
            free_slots = max_batch_size - len(survivors)
            admitted = pending[:free_slots]
            pending = pending[free_slots:]

            if survivors:
                active = survivors + admitted
                continue

            if admitted:
                active = admitted
                # All previous rows retired, so the implementation resets the cache
                # before prefilling the newly admitted requests.
                busy_span = 0
            else:
                active = []

        return max_cache_len

    def _validate_continuous_inputs(self, request, sampling, max_batch_size):
        if request.prompt_ids.shape[0] == 0:
            raise ValueError("continuous batching requires at least one request")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.use_shared_prefill:
            raise ValueError("continuous batching does not support shared prompt prefill")
        if self.enable_chunked_prefill and self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive when chunked prefill is enabled")
        if self.enable_chunked_prefill and (self.align_decode_fronts or self.adaptive_prefill):
            raise ValueError("chunked prefill does not support aligned decode fronts")
        # Hybrid models: GDN layers keep per-row conv/recurrent state that
        # the continuous-batching cache does not yet scatter or compact, so
        # a run would crash on an unbound pass-through slot mid-decode. Fail
        # with a clear message up front instead.
        config = getattr(self.engine.module, "config", None)
        text_config = getattr(config, "text_config", config)
        if "linear_attention" in (getattr(text_config, "layer_types", None) or []):
            raise ValueError("continuous batching does not yet support hybrid models with linear-attention layers")

        if sampling.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.adaptive_prefill and sampling.temperature > 0:
            raise ValueError("adaptive prefill currently supports greedy decoding only")
        if sampling.n_samples_per_prompt != 1:
            raise ValueError("continuous batching currently supports one sample per prompt")
        # Rollout masks contain 0/1; validate and read all lengths in one device-to-host transfer.
        # Count like the boolean token selection; a low-precision sum would round long prompts.
        lengths = request.prompt_attention_mask.count_nonzero(dim=1).tolist()
        if any(length <= 0 for length in lengths):
            raise ValueError("continuous batching requires at least one prompt token per request")
        return lengths

    @staticmethod
    def _continuous_dead_prefix(attention_mask, active_count):
        if active_count == 0:
            return 0
        occupied = attention_mask[:active_count].any(dim=0)
        if not occupied.any():
            return 0
        return int(occupied.to(dtype=torch.int32).argmax().item())

    def _continuous_prefill(self,
                            module,
                            static_cache_type,
                            cache,
                            update,
                            request_by_id,
                            attention_mask,
                            write_positions,
                            cache_position,
                            prompt_lengths,
                            model_dtype,
                            device,
                            sampling,
                            align_decode_fronts=False,
                            profile=None,
                            target_slots=None,
                            generation_config=None,
                            prefill_chunks=None,
                            chunk_size=0,
                            stage_prompts=None):
        if not update.admitted:
            return {}

        if target_slots is None:
            target_slots = update.admitted_slots
        if len(target_slots) != len(update.admitted):
            raise ValueError("continuous prefill needs one target slot per admitted request")
        if not align_decode_fronts:
            admitted_ids = tuple(request.request_id for request in update.admitted)
            if prefill_chunks is not None and chunk_size > 0:
                stager = stage_prompts if stage_prompts is not None else self._stage_long_prompts
                admitted_ids, target_slots = stager(module, admitted_ids, target_slots, request_by_id, prefill_chunks,
                                                    chunk_size, static_cache_type, model_dtype, device)
                if not admitted_ids:
                    return {}
            prompt_ids = torch.cat([request_by_id[request_id].prompt_ids for request_id in admitted_ids], dim=0)
            prompt_attention = torch.cat(
                [request_by_id[request_id].prompt_attention_mask for request_id in admitted_ids], dim=0)
            prompt_len = prompt_ids.shape[1]
            prefill_start = self._profile_start(profile)
            prefill_tokens, prefill_cache = self._continuous_prefill_cache(
                module,
                static_cache_type,
                prompt_ids,
                prompt_attention,
                model_dtype,
                device,
                sampling,
                generation_config=generation_config,
            )
            self._profile_end(profile, "prefill_forward_ms", prefill_start, count="num_prefill_forwards")
            cache_copy_start = self._profile_start(profile)
            if chunk_size > 0:
                # With chunked prefill every row lives behind its own column zero: a row
                # admitted mid-run must not be placed relative to a front that staged rows
                # have already pushed past the room it needs for its response.
                cache_start = 0
            else:
                cache_start = cache_position - prompt_len
                if cache_start < 0:
                    raise ValueError("continuous batching prompt does not fit behind the decode front")
            for layer_idx in range(self._cache_layer_count(prefill_cache)):
                prefill_keys, prefill_values = self._cache_layer_tensors(prefill_cache, layer_idx)
                target_layer = self._continuous_cache_layer(cache, layer_idx, prefill_keys, prefill_values,
                                                            attention_mask.shape[0])
                for source_row, target_row in enumerate(target_slots):
                    target_layer.keys[target_row, :,
                                      cache_start:cache_start + prompt_len].copy_(prefill_keys[source_row])
                    target_layer.values[target_row, :,
                                        cache_start:cache_start + prompt_len].copy_(prefill_values[source_row])
            for source_row, target_row in enumerate(target_slots):
                attention_mask[target_row, cache_start:cache_start + prompt_len].copy_(prompt_attention[source_row])
                write_positions[target_row] = cache_start + prompt_len
            self._profile_end(profile, "cache_management_overhead_ms", cache_copy_start)
            return dict(zip(admitted_ids, prefill_tokens.split(1, dim=0)))

        admitted_tokens = {}
        admitted = list(zip(update.admitted, target_slots))
        if self.adaptive_prefill:
            lengths = [prompt_lengths[item.request_id] for item, _ in admitted]
            buckets, _ = plan_prefill_buckets(lengths, self._prefill_cost_function(module), self.prefill_max_tokens)
        else:
            buckets = tuple((row, ) for row in range(len(admitted)))
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        for bucket in buckets:
            width = max(prompt_lengths[admitted[row][0].request_id] for row in bucket)
            prompt_ids = torch.full((len(bucket), width), pad_token_id, dtype=torch.long, device=device)
            prompt_attention = torch.zeros_like(prompt_ids)
            prompt_requests = {}
            for source_row, row in enumerate(bucket):
                item, _ = admitted[row]
                request = request_by_id[item.request_id]
                length = prompt_lengths[item.request_id]
                prompt_ids[source_row, -length:] = request.prompt_ids[0, request.prompt_attention_mask[0].bool()]
                prompt_attention[source_row, -length:] = 1
                prompt_requests[source_row] = request
            prefill_start = self._profile_start(profile)
            prefill_tokens, prefill_cache = self._continuous_prefill_cache(module,
                                                                           static_cache_type,
                                                                           prompt_ids,
                                                                           prompt_attention,
                                                                           model_dtype,
                                                                           device,
                                                                           sampling,
                                                                           prompt_requests=prompt_requests,
                                                                           generation_config=generation_config)
            self._profile_end(profile, "prefill_forward_ms", prefill_start, count="num_prefill_forwards")
            cache_copy_start = self._profile_start(profile)
            for source_row, row in enumerate(bucket):
                item, target_row = admitted[row]
                length = prompt_lengths[item.request_id]
                cache_start = cache_position - length
                if cache_start < 0:
                    raise ValueError("continuous batching prompt does not fit behind the decode front")
                admitted_tokens[item.request_id] = prefill_tokens[source_row:source_row + 1]
                for layer_idx in range(self._cache_layer_count(prefill_cache)):
                    keys, values = self._cache_layer_tensors(prefill_cache, layer_idx)
                    target_layer = self._continuous_cache_layer(cache, layer_idx, keys, values,
                                                                attention_mask.shape[0])
                    target_layer.keys[target_row, :, cache_start:cache_position].copy_(keys[source_row, :, -length:])
                    target_layer.values[target_row, :, cache_start:cache_position].copy_(values[source_row, :,
                                                                                                -length:])
                attention_mask[target_row, cache_start:cache_position].fill_(1)
                write_positions[target_row] = cache_position
            self._profile_end(profile, "cache_management_overhead_ms", cache_copy_start)
        return admitted_tokens

    @staticmethod
    def _continuous_cache_layer(cache, layer_idx, prefill_keys, prefill_values, max_batch_size):
        layer = cache.layers[layer_idx]
        if not layer.is_initialized:
            layer.lazy_initialization(prefill_keys, prefill_values, max_batch_size=max_batch_size)
        return layer

    def _prefill_cost_function(self, module):
        """Read model geometry once; DP candidates use only scalar arithmetic."""
        config = getattr(module.config, "text_config", module.config)
        layers = getattr(config, "num_hidden_layers", 1)
        heads = getattr(config, "num_attention_heads", 1)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(config, "hidden_size", heads) // heads
        kv_heads = getattr(config, "num_key_value_heads", None) or heads
        attention_cost = self.prefill_attention_cost_ms * 4 * layers * heads * head_dim / 1e9
        element_size = next(module.parameters()).element_size()
        kv_cost = self.prefill_kv_cost_ms * 2 * layers * kv_heads * head_dim * element_size / (1024 * 1024)

        def cost(count, width, include_kv=True):
            tokens = count * width
            return self.prefill_fixed_cost_ms + tokens * (self.prefill_token_cost_ms + attention_cost * width +
                                                          (kv_cost if include_kv else 0))

        return cost

    # ------------------------------------------------------------------
    # Chunked prefill (eager continuous path)
    # ------------------------------------------------------------------

    def _stage_long_prompts(self, module, admitted_ids, target_slots, request_by_id, prefill_chunks, chunk_size,
                            static_cache_type, model_dtype, device):
        """Divert prompts that do not fit one chunk out of this step's prefill batch.

        A staged prompt keeps its slot and is advanced by ``_run_chunked_prefill`` on later
        steps, so the decode batch keeps running while a long prompt fills. The staged window
        only ever covers real tokens, so the staged length is the prompt's effective length:
        a batch padded to a wide common width does not stage prompts that are actually short.

        Returns the ids and slots that still take the one-shot prefill path.
        """
        batched_ids, batched_slots = [], []
        for request_id, target_slot in zip(admitted_ids, target_slots):
            request = request_by_id[request_id]
            valid = request.prompt_attention_mask[0].bool()
            prompt_ids = request.prompt_ids[:, valid]
            prompt_len = prompt_ids.shape[1]
            if prompt_len <= chunk_size:
                batched_ids.append(request_id)
                batched_slots.append(target_slot)
                continue
            prompt_attention = torch.ones_like(prompt_ids)
            staging_cache = self._create_prefill_cache(static_cache_type, module.config, prompt_ids.shape[0],
                                                       prompt_len, device, model_dtype)
            prefill_chunks[request_id] = _PrefillChunk(slot=target_slot,
                                                       cache=staging_cache,
                                                       prompt_ids=prompt_ids,
                                                       prompt_attention=prompt_attention,
                                                       position_ids=self._prefill_position_ids(prompt_attention))
        return tuple(batched_ids), tuple(batched_slots)

    def _advance_prefill_chunk(self, module, chunk, width, device, profile=None):
        """Prefill the next chunk of a staged prompt.

        Returns the first response token once the prompt is complete, else ``None``. The
        window ends at the chunk boundary but the attention mask covers the whole staged
        prefix, which is what lets a later chunk read the KV written by earlier ones.
        """
        end = min(chunk.prompt_len, chunk.offset + width)
        validity = torch.zeros_like(chunk.prompt_attention)
        validity[:, :end] = chunk.prompt_attention[:, :end]
        forward_start = self._profile_start(profile)
        output = self._call_model(
            module,
            chunk.prompt_ids[:, chunk.offset:end],
            attention_mask=validity,
            past_key_values=chunk.cache,
            use_cache=True,
            cache_position=torch.arange(chunk.offset, end, device=device),
            position_ids=chunk.position_ids[:, chunk.offset:end],
            **self._last_logits_kwargs(module),
        )
        self._profile_end(profile, "prefill_forward_ms", forward_start, count="num_prefill_forwards")
        chunk.offset = end
        if end < chunk.prompt_len:
            return None
        # The chunk that completes the prompt also predicts the first response token.
        return output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    def _publish_prefill_chunk(self, chunk, cache, attention_mask, write_positions):
        """Move a completed staged prompt into its row of the decode cache.

        The prompt lands at column zero of its own row: the cache is row-major, so a row's
        columns are its own and do not compete with the decode front of the rows around it.
        That keeps the cache capacity a function of the prompt and the response budget only,
        never of how long the prompt spent staging.
        """
        prompt_end = chunk.prompt_len
        for layer_idx in range(self._cache_layer_count(chunk.cache)):
            staged_keys, staged_values = self._cache_layer_tensors(chunk.cache, layer_idx)
            target_layer = cache.layers[layer_idx]
            if not target_layer.is_initialized:
                # The decode cache sizes its layers lazily from the first KV it sees; a batch whose
                # admitted rows are all staged never runs a one-shot prefill, so initialize here.
                target_layer.lazy_initialization(staged_keys[:1],
                                                 staged_values[:1],
                                                 max_batch_size=attention_mask.shape[0])
            target_layer.keys[chunk.slot, :, :prompt_end].copy_(staged_keys[0])
            target_layer.values[chunk.slot, :, :prompt_end].copy_(staged_values[0])
        attention_mask[chunk.slot, :prompt_end].copy_(chunk.prompt_attention[0])
        write_positions[chunk.slot] = prompt_end

    def _run_chunked_prefill(self,
                             module,
                             prefill_chunks,
                             chunk_size,
                             device,
                             cache,
                             attention_mask,
                             write_positions,
                             chunk_stats,
                             profile=None):
        """Advance the staged prompts by at most ``chunk_size`` tokens in total.

        The budget is spent in admission order, so the number of extra forwards in a step
        is bounded no matter how many prompts are waiting. Returns the first response
        token of every prompt that completed in this step, keyed by request id.
        """
        budget = chunk_size
        completed = {}
        for request_id in list(prefill_chunks):
            if budget <= 0:
                break
            chunk = prefill_chunks[request_id]
            width = min(chunk.remaining, budget)
            first_token = self._advance_prefill_chunk(module, chunk, width, device, profile)
            budget -= width
            chunk_stats["chunks"] += 1
            chunk_stats["peak_staging_bytes"] = max(chunk_stats["peak_staging_bytes"], chunk.staging_bytes())
            if first_token is None:
                continue
            self._publish_prefill_chunk(chunk, cache, attention_mask, write_positions)
            completed[request_id] = first_token
            del prefill_chunks[request_id]
        return completed

    # ------------------------------------------------------------------
    # Captured chunked prefill (continuous graph path)
    # ------------------------------------------------------------------

    def _stage_staged_prompt(self, module, admitted_ids, target_slots, request_by_id, prefill_chunks, chunk_size,
                             static_cache_type, model_dtype, device):
        """Divert prompts that do not fit one chunk into the shared staging cache.

        The signature matches ``_stage_long_prompts`` so the continuous graph loop can hand
        it to the shared admission path, but the staging itself is different: rows live in
        one cache that every captured prefill width can address, and the staged row is the
        whole padded prompt rather than its real tokens. A continuous batch shares one
        padded width, so staging the padded row keeps the chunk grid constant across
        prompts, which is what lets one captured chunk graph serve all of them.

        Staged prompts advance in waves: every row of a wave writes the same chunk window,
        so the wave is a single forward with a single scalar write position. A prompt
        admitted while a wave runs waits for the next one, holding its decode slot and
        spending no generation budget, exactly like a prompt that is mid-wave.
        """
        state = self._continuous_graph_state
        if state.staging_cache is None:
            raise ValueError("chunked prefill capture requires a staging cache")
        batched_ids, batched_slots = [], []
        pending = []
        for request_id, target_slot in zip(admitted_ids, target_slots):
            request = request_by_id[request_id]
            if int(request.prompt_attention_mask.sum().item()) <= chunk_size:
                batched_ids.append(request_id)
                batched_slots.append(target_slot)
                continue
            pending.append((request_id, target_slot, request))
        wave_running = any(chunk.row is not None for chunk in prefill_chunks.values())
        for request_id, target_slot, request in pending:
            prefill_chunks[request_id] = _StagedPrompt(slot=target_slot,
                                                       row=None,
                                                       prompt_ids=request.prompt_ids,
                                                       prompt_attention=request.prompt_attention_mask,
                                                       position_ids=self._prefill_position_ids(
                                                           request.prompt_attention_mask))
        if pending and not wave_running:
            for row, chunk in enumerate(prefill_chunks.values()):
                chunk.row = row
        return tuple(batched_ids), tuple(batched_slots)

    def _run_staged_prefill(self,
                            module,
                            prefill_chunks,
                            chunk_size,
                            prompt_len,
                            cache,
                            attention_mask,
                            write_positions,
                            chunk_stats,
                            profile=None,
                            request_write_positions=None):
        """Advance the running wave by one chunk window and publish the prompts that finish.

        Every row of a wave shares one window, so the whole wave is one forward at
        ``[rows, chunk_size]`` padded up to the nearest captured bucket, replayed from a
        graph when the bucket is warm. Prompts still waiting for a wave are skipped.
        """
        state = self._continuous_graph_state
        wave = [(request_id, chunk) for request_id, chunk in prefill_chunks.items() if chunk.row is not None]
        if not wave:
            return {}
        starts = {chunk.start for _, chunk in wave}
        if len(starts) != 1:
            raise ValueError(f"staged prompts of one wave must share a chunk window, saw {sorted(starts)}")
        # The last window slides back to end at the prompt instead of starting where the
        # chunk grid would: it keeps the chunk width constant without padding fake tokens
        # into the cache, and it re-writes columns the previous window already filled with
        # the same values.
        window = min(wave[0][1].start, prompt_len - chunk_size)
        end = window + chunk_size
        width = self._bucket_width(state.ladder, len(wave))
        bucket = self._prefill_bucket(width)
        bucket.input_ids.zero_()
        bucket.position_ids.zero_()
        for row_index, (_, chunk) in enumerate(wave):
            bucket.input_ids[row_index].copy_(chunk.prompt_ids[0, window:end])
            bucket.position_ids[row_index].copy_(chunk.position_ids[0, window:end])
        self._fill_staged_attention(bucket, wave, window)
        # The bucket is filled before it is captured: capture warms up on the buffers, and
        # those warmups write the same KV the following replay writes.
        self._capture_bucket_if_needed(module, state.staging_cache, bucket, state.static_cache_position,
                                       bucket.write_positions)
        forward_start = self._profile_start(profile)
        logits = self._run_bucket(module, state.staging_cache, bucket, state.static_cache_position,
                                  bucket.write_positions)
        self._profile_end(profile, "prefill_forward_ms", forward_start, count="num_prefill_forwards")
        completion = end >= prompt_len
        completed = {}
        for row_index, (request_id, chunk) in enumerate(wave):
            chunk.start = end
            if not completion:
                continue
            # The chunk that completes the prompt also predicts the first response token.
            first_token = logits[row_index:row_index + 1, -1, :].argmax(dim=-1, keepdim=True)
            self._publish_staged_prompt(chunk, state, cache, attention_mask, write_positions)
            if request_write_positions is not None:
                request_write_positions[request_id] = prompt_len
            completed[request_id] = first_token
            del prefill_chunks[request_id]
        chunk_stats["chunks"] += len(wave)
        chunk_stats["graph_steps"] = chunk_stats.get("graph_steps", 0) + 1
        chunk_stats["padded_rows"] = chunk_stats.get("padded_rows", 0) + width - len(wave)
        widths = chunk_stats.setdefault("bucket_widths", {})
        widths[width] = widths.get(width, 0) + 1
        return completed

    def _fill_staged_attention(self, bucket, wave, window):
        """Write the chunk's additive mask and its write positions in place.

        A query at column ``window + j`` may attend to every staged key up to and including
        column ``window + j``, minus the prompt's own padding. Rows the bucket carries past
        the wave stay masked everywhere, so their logits are the ones the caller discards.
        A wave shares one window, so every row of it writes the same columns; the position
        stays a per-row tensor because the cache needs one that covers only the active rows.
        """
        min_dtype = torch.finfo(bucket.causal_attention.dtype).min
        if bucket.width > len(wave):
            bucket.causal_attention[len(wave):].fill_(min_dtype)
        chunk_size = bucket.input_ids.shape[1]
        keys = torch.arange(bucket.causal_attention.shape[-1], device=bucket.causal_attention.device)
        queries = window + torch.arange(chunk_size, device=keys.device)
        for row_index, (_, chunk) in enumerate(wave):
            valid = (keys[None, :] <= queries[:, None]) & chunk.prompt_attention[0].bool()[None, :]
            bucket.causal_attention[row_index, 0].copy_(torch.where(valid, 0.0, min_dtype))
        bucket.write_positions.fill_(window)

    def _publish_staged_prompt(self, chunk, state, cache, attention_mask, write_positions):
        """Move a finished staged prompt into its row of the decode cache.

        The prompt lands at column zero of its own row, exactly like a one-shot prefill, so
        the row's response starts at the padded prompt width and the cache capacity stays a
        function of the prompt and the response budget, never of how many chunks it took.
        """
        prompt_len = chunk.prompt_len
        for layer_idx in range(self._cache_layer_count(state.staging_cache)):
            staged_keys, staged_values = self._cache_layer_tensors(state.staging_cache, layer_idx)
            target_layer = cache.layers[layer_idx]
            if not target_layer.is_initialized:
                # The capture cache sizes its layers lazily; when every bucket falls back to
                # eager windows no captured warmup has allocated them yet.
                target_layer.lazy_initialization(staged_keys[chunk.row:chunk.row + 1],
                                                 staged_values[chunk.row:chunk.row + 1],
                                                 max_batch_size=attention_mask.shape[0])
            target_layer.keys[chunk.slot, :, :prompt_len].copy_(staged_keys[chunk.row, :, :prompt_len])
            target_layer.values[chunk.slot, :, :prompt_len].copy_(staged_values[chunk.row, :, :prompt_len])
        attention_mask[chunk.slot, :prompt_len].copy_(chunk.prompt_attention[0])
        write_positions[chunk.slot] = prompt_len

    def _chunked_prefill_cache(self,
                               module,
                               static_cache_type,
                               prompt_ids,
                               prompt_attention,
                               model_dtype,
                               device,
                               chunk_size,
                               max_cache_len=None,
                               profile=None):
        """Prefill a prompt batch in windows and return the completing window's logits and cache.

        This is the whole-prompt counterpart of ``_advance_prefill_chunk``: the static
        generation paths own no scheduler, so they consume the chunks in one pass and then
        hand the filled cache to their decode loop. Windows advance by at most
        ``chunk_size`` tokens, which bounds the largest forward a caller can be made to run.
        ``max_cache_len`` sizes the cache for callers that keep decoding into it.
        """
        prompt_len = prompt_ids.shape[1]
        if max_cache_len is None:
            max_cache_len = prompt_len
        position_ids = self._prefill_position_ids(prompt_attention)
        prefill_cache = self._create_prefill_cache(static_cache_type, module.config, prompt_ids.shape[0],
                                                   max_cache_len, device, model_dtype)
        output = None
        for start in range(0, prompt_len, chunk_size):
            end = min(prompt_len, start + chunk_size)
            # Validity covers the whole staged prefix, so a later chunk reads the KV that
            # the earlier ones wrote while only the current window is queried.
            validity = torch.zeros_like(prompt_attention)
            validity[:, :end] = prompt_attention[:, :end]
            forward_start = self._profile_start(profile)
            output = self._call_model(
                module,
                prompt_ids[:, start:end],
                attention_mask=validity,
                past_key_values=prefill_cache,
                use_cache=True,
                cache_position=torch.arange(start, end, device=device),
                position_ids=position_ids[:, start:end],
                **self._last_logits_kwargs(module),
            )
            self._profile_end(profile, "prefill_forward_ms", forward_start, count="num_prefill_forwards")
        # The completing window's logits belong to the caller: they carry the first response
        # token, and the caller's selector may have to reshape them first (repetition penalty).
        return output.logits[:, -1, :], prefill_cache

    def _start_continuous_profile(self):
        accelerator = get_accelerator()
        accelerator.synchronize()
        return {
            "accelerator": accelerator,
            "start": time.perf_counter(),
            "prefill_forward_ms": 0.0,
            "decode_forward_ms": 0.0,
            "scheduler_overhead_ms": 0.0,
            "cache_management_overhead_ms": 0.0,
            "num_prefill_forwards": 0,
            "num_decode_forwards": 0,
            "active_batch_sizes": [],
        }

    @staticmethod
    def _profile_start(profile):
        if profile is None:
            return None
        profile["accelerator"].synchronize()
        return time.perf_counter()

    @staticmethod
    def _profile_end(profile, field, start, count=None):
        if profile is None or start is None:
            return None
        profile["accelerator"].synchronize()
        end = time.perf_counter()
        if field is not None:
            profile[field] += (end - start) * 1000.0
        if count is not None:
            profile[count] += 1
        return end

    def _finish_continuous_profile(self, profile, request, responses, max_batch_size, prompt_len, generation_end,
                                   post_processing_end):
        total_ms = (post_processing_end - profile["start"]) * 1000.0
        generation_ms = (generation_end - profile["start"]) * 1000.0
        prefill_ms = profile["prefill_forward_ms"]
        decode_ms = profile["decode_forward_ms"]
        response_lengths = [len(response) for response in responses.values()]
        num_generated_tokens = sum(response_lengths)
        self._last_profile = {
            "prompt_expansion_ms": 0.0,
            "generation_ms": generation_ms,
            "prefill_forward_ms": prefill_ms,
            "decode_forward_ms": decode_ms,
            "generation_overhead_ms": max(0.0, generation_ms - prefill_ms - decode_ms),
            "num_prefill_forwards": profile["num_prefill_forwards"],
            "num_decode_forwards": profile["num_decode_forwards"],
            "scheduler_overhead_ms": profile["scheduler_overhead_ms"],
            "cache_management_overhead_ms": profile["cache_management_overhead_ms"],
            "post_processing_ms": (post_processing_end - generation_end) * 1000.0,
            "total_ms": total_ms,
            "num_generated_tokens": num_generated_tokens,
            "tokens_per_second": num_generated_tokens / (total_ms / 1000.0) if total_ms > 0.0 else 0.0,
            "batch_size": request.prompt_ids.shape[0],
            "num_samples_per_prompt": 1,
            "prompt_length": prompt_len,
            "response_length": max(response_lengths, default=0),
            "active_batch_size": max(profile["active_batch_sizes"], default=0),
            "continuous_batch_size": max_batch_size,
        }

    def _continuous_prefill_cache(self,
                                  module,
                                  static_cache_type,
                                  prompt_ids,
                                  prompt_attention,
                                  model_dtype,
                                  device,
                                  sampling,
                                  prompt_requests=None,
                                  generation_config=None):
        position_ids = self._prefill_position_ids(prompt_attention)
        if getattr(module, "_supports_cache_class", None) is None:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            prefill_output = module.generate(
                prompt_ids,
                attention_mask=prompt_attention,
                position_ids=position_ids,
                max_new_tokens=1,
                eos_token_id=None,
                pad_token_id=pad_token_id,
                return_dict_in_generate=True,
                **self._hf_sampling_kwargs(sampling),
                **({
                    "output_logits": True
                } if self.adaptive_prefill else {}),
            )
            prefill_cache = prefill_output.past_key_values
            if not self.adaptive_prefill:
                return prefill_output.sequences[:, -1:], prefill_cache
            logits = prefill_output.logits[-1]
        else:
            prompt_len = prompt_ids.shape[1]
            prefill_cache = self._create_prefill_cache(static_cache_type, module.config, prompt_ids.shape[0],
                                                       prompt_len, device, model_dtype)
            prefill_output = self._call_model(
                module,
                prompt_ids,
                attention_mask=prompt_attention,
                past_key_values=prefill_cache,
                use_cache=True,
                cache_position=torch.arange(prompt_len, device=device),
                position_ids=position_ids,
                **self._last_logits_kwargs(module),
            )
            logits = prefill_output.logits[:, -1, :]
        request_ids = range(prompt_ids.shape[0])
        if prompt_requests is None:
            prompt_requests = {
                row: RolloutRequest(prompt_ids[row:row + 1], prompt_attention[row:row + 1])
                for row in request_ids
            }
        # Trimming model input must not trim HF's repetition-penalty history.
        next_tokens = self._continuous_next_tokens(logits,
                                                   request_ids,
                                                   prompt_requests,
                                                   dict.fromkeys(request_ids, ()),
                                                   module,
                                                   self._token_selector(sampling),
                                                   generation_config=generation_config)
        return next_tokens, prefill_cache

    @staticmethod
    def _repetition_penalty(module, generation_config=None):
        if generation_config is None:
            generation_config = getattr(module, "generation_config", None)
        repetition_penalty = getattr(generation_config, "repetition_penalty", 1.0)
        return 1.0 if repetition_penalty is None else repetition_penalty

    @staticmethod
    @lru_cache(maxsize=1)
    def _repetition_penalty_ignores_padding():
        from transformers import RepetitionPenaltyLogitsProcessor

        # HF introduced per-row ignore lengths together with padding-aware generation.
        try:
            RepetitionPenaltyLogitsProcessor(1.0, prompt_ignore_length=torch.zeros(1, dtype=torch.long, device="cpu"))
        except (TypeError, ValueError):
            return False
        return True

    @classmethod
    def _apply_repetition_penalty(cls, logits, input_ids, repetition_penalty, prompt_attention_mask=None):
        """Penalize the tokens of ``input_ids`` in last-position ``logits``, without changing ``logits``.

        generate() applies the penalty in float32, before the sampling warpers, so it shapes the sampled distribution.
        """
        if repetition_penalty == 1.0:
            return logits
        logits = logits.float()
        if prompt_attention_mask is not None and cls._repetition_penalty_ignores_padding():
            from transformers import RepetitionPenaltyLogitsProcessor

            padding_lengths = prompt_attention_mask.ne(0).int().argmax(dim=1)
            return RepetitionPenaltyLogitsProcessor(repetition_penalty,
                                                    prompt_ignore_length=padding_lengths)(input_ids, logits)
        scores = logits.gather(1, input_ids)
        scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
        return logits.scatter(1, input_ids, scores)

    @classmethod
    def _continuous_next_tokens(cls,
                                logits,
                                request_ids,
                                request_by_id,
                                responses,
                                module,
                                select_next_tokens,
                                generation_config=None):
        repetition_penalty = cls._repetition_penalty(module, generation_config)
        if repetition_penalty != 1.0:
            logits = torch.cat([
                cls._apply_repetition_penalty(
                    logits[row:row + 1],
                    torch.cat((request_by_id[request_id].prompt_ids, *responses[request_id]), dim=1),
                    repetition_penalty,
                    request_by_id[request_id].prompt_attention_mask,
                ) for row, request_id in enumerate(request_ids)
            ],
                               dim=0)
        return select_next_tokens(logits)

    @staticmethod
    def _prefill_position_ids(attention_mask):
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        return position_ids.masked_fill(attention_mask == 0, 1)

    def _is_eos(self, token):
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            return False
        eos_ids = torch.as_tensor(eos_token_id, dtype=token.dtype, device=token.device).flatten()
        return bool((token == eos_ids).any().item())

    @staticmethod
    def _last_logits_kwargs(module):
        """Ask models that support it for last-position logits only, since prefill samples just one token."""
        # Without this the prefill materializes [batch, prompt_len, vocab] logits, the largest activation of the
        # rollout. Models that do not declare the argument may forward unknown kwargs to submodules, so skip them.
        if "logits_to_keep" in signature(module.forward).parameters:
            return {"logits_to_keep": 1}
        return {}

    @staticmethod
    def _call_model(module, input_ids, **kwargs):
        """Call models with only the cache arguments their HF version accepts."""
        parameters = signature(module.forward).parameters
        accepts_kwargs = any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values())
        if not accepts_kwargs:
            kwargs = {name: value for name, value in kwargs.items() if name in parameters}
        return module(input_ids, **kwargs)

    @classmethod
    def _create_prefill_cache(cls, static_cache_type, config, batch_size, max_cache_len, device, dtype):
        """Cache for a single prefill forward whose KV is then copied into the decode cache.

        Under autocast, keys and values can come out in different dtypes (rotary embeddings promote
        bf16 key projections to fp32), which HF StaticCache rejects. DeepSpeedStaticCache accepts them,
        but models every layer as full attention, so sliding-window and linear-attention models keep
        HF StaticCache.
        """
        text_config = getattr(config, "text_config", config)
        layer_types = getattr(text_config, "layer_types", None) or []
        full_attention_only = (getattr(text_config, "sliding_window", None) is None
                               and all(layer_type == "full_attention" for layer_type in layer_types))
        if not full_attention_only:
            return cls._create_static_cache(static_cache_type, config, batch_size, max_cache_len, device, dtype)

        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        # Sized lazily from the prefill keys, as AutoTP shards KV heads.
        return DeepSpeedStaticCache(config, max_cache_len=max_cache_len)

    @staticmethod
    def _create_static_cache(static_cache_type, config, batch_size, max_cache_len, device, dtype):
        """Construct StaticCache across Transformers' batch-size API variants."""
        cache_config = config
        if not hasattr(config, "num_key_value_heads") or config.num_key_value_heads is None:
            cache_config = copy(config)
            cache_config.num_key_value_heads = getattr(config, "num_attention_heads", 1)
        common_kwargs = {
            "config": cache_config,
            "max_cache_len": max_cache_len,
            "device": device,
            "dtype": dtype,
        }
        parameters = signature(static_cache_type).parameters
        if "batch_size" in parameters:
            common_kwargs["batch_size"] = batch_size
        elif "max_batch_size" in parameters:
            common_kwargs["max_batch_size"] = batch_size
        return static_cache_type(**common_kwargs)

    @staticmethod
    def _cache_layer_count(cache):
        if hasattr(cache, "layers"):
            return len(cache.layers)
        return len(cache.key_cache)

    @staticmethod
    def _cache_layer_tensors(cache, layer_idx):
        if hasattr(cache, "layers"):
            layer = cache.layers[layer_idx]
            return layer.keys, layer.values
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]

    def _build_continuous_batch(self, request, responses):
        response_ids = [torch.cat(responses[index], dim=1) for index in range(request.prompt_ids.shape[0])]
        max_response_len = max(response.shape[1] for response in response_ids)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("The tokenizer must define pad_token_id or eos_token_id")
        input_rows = []
        attention_rows = []
        for index, response in enumerate(response_ids):
            padding = max_response_len - response.shape[1]
            if padding:
                response_padding = torch.full((1, padding), pad_token_id, dtype=response.dtype, device=response.device)
                attention_padding = torch.zeros((1, padding),
                                                dtype=request.prompt_attention_mask.dtype,
                                                device=response.device)
                response = torch.cat((response, response_padding), dim=1)
            else:
                attention_padding = torch.empty((1, 0),
                                                dtype=request.prompt_attention_mask.dtype,
                                                device=response.device)
            input_rows.append(torch.cat((request.prompt_ids[index:index + 1], response), dim=1))
            response_attention = torch.ones((1, response_ids[index].shape[1]),
                                            dtype=request.prompt_attention_mask.dtype,
                                            device=response.device)
            attention_rows.append(
                torch.cat((request.prompt_attention_mask[index:index + 1], response_attention, attention_padding),
                          dim=1))

        input_ids = torch.cat(input_rows, dim=0)
        attention_mask = torch.cat(attention_rows, dim=0)
        response_start = request.prompt_ids.shape[1]
        return RolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_start_idx=torch.full((request.prompt_ids.shape[0], ),
                                          response_start,
                                          dtype=torch.long,
                                          device=input_ids.device),
        )

    def _zero3_work_remaining(self):
        """Return a rendezvous over the group used for ZeRO-3 parameter gathers."""
        is_zero3 = getattr(self.engine, "zero_optimization_partition_weights", None)
        if not callable(is_zero3) or not is_zero3():
            return None

        process_group = None
        if dist.is_initialized():
            optimizer = getattr(self.engine, "optimizer", None)
            process_group = getattr(optimizer, "dp_process_group", None)
            if process_group is None:
                raise RuntimeError("ZeRO-3 synchronized rollout requires optimizer.dp_process_group")
        return GlobalWorkRemaining(process_group)

    def get_last_profile(self):
        """Return the most recent profiling snapshot for this rollout instance."""
        return self._last_profile

    def get_last_continuous_stats(self):
        """Return cache and throughput statistics from the most recent continuous rollout."""
        return self._last_continuous_stats

    def _register_shared_prefill_hooks(self, module, batch_size, repeats):
        state = {"pending": True, "reduced": False}

        def reduce_prompt_batch(_module, args, kwargs):
            input_ids = kwargs.get("input_ids")
            if not state["pending"]:
                return args, kwargs
            if input_ids is None:
                raise RuntimeError("Shared prefill requires input_ids as a keyword argument")
            expected_batch_size = batch_size * repeats
            if input_ids.shape[0] != expected_batch_size:
                raise RuntimeError("Shared prefill input batch does not match the expanded rollout batch")
            if input_ids.shape[1] <= 1:
                raise RuntimeError("Shared prefill requires a prompt with more than one token")
            kwargs = dict(kwargs)
            kwargs["input_ids"] = input_ids[::repeats]
            for name in ("attention_mask", "position_ids", "token_type_ids"):
                value = kwargs.get(name)
                if isinstance(value, torch.Tensor) and value.shape[0] == expected_batch_size:
                    kwargs[name] = value[::repeats]
            state["reduced"] = True
            return args, kwargs

        def expand_prompt_output(_module, _args, _kwargs, output):
            if not state["pending"]:
                return output
            if not state["reduced"]:
                raise RuntimeError("Shared prefill did not reduce the prompt batch")
            state["pending"] = False
            output.past_key_values = self.engine.repeat_shared_prefill_cache(batch_size, repeats)
            output.logits = output.logits.repeat_interleave(repeats, dim=0)
            return output

        pre_handle = module.register_forward_pre_hook(reduce_prompt_batch, with_kwargs=True)
        post_handle = module.register_forward_hook(expand_prompt_output, with_kwargs=True)
        return pre_handle, post_handle

    @staticmethod
    def _pad_after_eos(output_ids, response_start, eos_token_id, pad_token_id, mask_padding=True):
        """Retain the first response EOS and pad every subsequent position."""
        response_ids = output_ids[:, response_start:]
        response_attn = (response_ids != pad_token_id) if mask_padding else torch.ones_like(response_ids,
                                                                                            dtype=torch.bool)

        if eos_token_id is None or response_ids.shape[1] == 0:
            return output_ids, response_attn.long()

        eos_ids = torch.as_tensor(eos_token_id, device=response_ids.device, dtype=response_ids.dtype).flatten()
        is_eos = (response_ids.unsqueeze(-1) == eos_ids).any(dim=-1)
        has_eos = is_eos.any(dim=-1)
        first_eos_idx = is_eos.long().argmax(dim=-1)
        positions = torch.arange(response_ids.shape[1], device=response_ids.device).unsqueeze(0)
        after_first_eos = has_eos.unsqueeze(1) & (positions > first_eos_idx.unsqueeze(1))
        first_eos = has_eos.unsqueeze(1) & (positions == first_eos_idx.unsqueeze(1))

        output_ids = output_ids.clone()
        output_ids[:, response_start:].masked_fill_(after_first_eos, pad_token_id)
        # EOS is a valid generated token even when pad_token_id == eos_token_id.
        response_attn = (response_attn | first_eos) & ~after_first_eos
        return output_ids, response_attn.long()

    # ------------------------------------------------------------------
    # Graph capture decode loop (greedy only)
    # ------------------------------------------------------------------

    @staticmethod
    def _token_selector(sampling):
        """Return a function that picks the next tokens ``[B, 1]`` from last-position logits ``[B, V]``.

        Tokens are picked after each forward or graph replay, outside any
        captured graph. Sampling applies Hugging Face's temperature, top-k and
        top-p warpers in the order ``generate()`` applies them, so every path
        draws from the same distribution as ``generate()``.
        """
        if sampling.temperature <= 0.0:
            return lambda logits: logits.argmax(dim=-1, keepdim=True)

        from transformers.generation.logits_process import (TemperatureLogitsWarper, TopKLogitsWarper,
                                                            TopPLogitsWarper)

        warpers = []
        if sampling.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(sampling.temperature))
        if sampling.top_k > 0:
            warpers.append(TopKLogitsWarper(top_k=sampling.top_k))
        if sampling.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=sampling.top_p))

        def sample(logits):
            # generate() also warps and samples in float32.
            scores = logits.float()
            for warper in warpers:
                # The warpers read only the scores, not the generated ids.
                scores = warper(None, scores)
            return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)

        return sample

    @staticmethod
    def _hf_sampling_kwargs(sampling):
        """Hugging Face ``generate()`` arguments for ``sampling``; ``temperature <= 0`` is greedy."""
        if sampling.temperature <= 0.0:
            return {"do_sample": False, "temperature": 1.0, "top_p": 1.0}
        return {
            "do_sample": True,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": max(sampling.top_k, 0),
        }

    def _generate_from_staged_prefill(self, prompt_ids, prompt_attn, max_new_tokens, pad_token_id, module, device):
        """Stage a long prompt in chunks, then let HuggingFace generate from the filled cache.

        This path hands the decode loop to ``module.generate``, so the only forward this
        layer can split is the prompt's. Passing a cache that already holds the prompt keeps
        the token stream identical to a single-forward prefill while bounding the largest
        forward the caller has to run. The first response token comes out of the staging
        chunks, so the continuation asks for one token less.
        """
        from transformers import StaticCache

        prompt_len = prompt_ids.shape[1]
        model_dtype = next(module.parameters()).dtype
        first_logits, cache = self._chunked_prefill_cache(module,
                                                          StaticCache,
                                                          prompt_ids,
                                                          prompt_attn,
                                                          model_dtype,
                                                          device,
                                                          self.prefill_chunk_size,
                                                          max_cache_len=prompt_len + max_new_tokens)
        repetition_penalty = self._repetition_penalty(module)
        if repetition_penalty != 1.0:
            # generate() would penalize the first response token like every later one.
            first_logits = self._apply_repetition_penalty(first_logits, prompt_ids, repetition_penalty, prompt_attn)
        first_token = first_logits.argmax(dim=-1, keepdim=True)
        if max_new_tokens == 1:
            return torch.cat([prompt_ids, first_token], dim=1)
        continuation_mask = torch.cat(
            [prompt_attn, torch.ones(prompt_ids.shape[0], 1, dtype=prompt_attn.dtype, device=device)], dim=1)
        continuation = module.generate(
            first_token,
            past_key_values=cache,
            attention_mask=continuation_mask,
            max_new_tokens=max_new_tokens - 1,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=pad_token_id,
        )
        return torch.cat([prompt_ids, continuation], dim=1)

    def _generate_graph(self, prompt_ids, prompt_attn, max_new_tokens, pad_token_id, module, device, sampling):
        """Decode with DeepSpeedStaticCache + graph capture, greedy or sampled per ``sampling``."""
        from transformers import StaticCache
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        if getattr(module.config, "_attn_implementation", None) == "flash_attention_2":
            raise ValueError("graph capture does not support flash_attention_2")

        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        max_len = prompt_len + max_new_tokens
        eos_token_id = self.tokenizer.eos_token_id
        model_dtype = next(module.parameters()).dtype
        select_next_tokens = self._token_selector(sampling)
        repetition_penalty = self._repetition_penalty(module)
        output_ids = [prompt_ids]

        def next_tokens(logits):
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, torch.cat(output_ids, dim=1), repetition_penalty,
                                                        prompt_attn)
            return select_next_tokens(logits)

        # --- Prefill with a static cache (correct attention semantics) ---
        # Hybrid models take a per-type mask dict (GDN must see None; a 2D
        # mask corrupts its conv-state init); standard models take a tensor.
        is_hybrid = "linear_attention" in (getattr(getattr(module.config, "text_config", module.config), "layer_types",
                                                   None) or [])
        prefill_mask = {"full_attention": None, "linear_attention": None} if is_hybrid else prompt_attn
        if (self._chunked_prefill_enabled(sampling) and not is_hybrid and prompt_len > self.prefill_chunk_size):
            # Stage the prompt and read the first response token off the completing window,
            # so the largest forward this path runs is one chunk wide. Capture starts after
            # prefill either way, so the decode graph itself is unaffected.
            first_logits, prefill_cache = self._chunked_prefill_cache(module, StaticCache, prompt_ids, prompt_attn,
                                                                      model_dtype, device, self.prefill_chunk_size)
            next_token = next_tokens(first_logits)
        else:
            prefill_cache = self._create_prefill_cache(StaticCache, module.config, batch_size, max_len, device,
                                                       model_dtype)
            prefill_out = module(
                prompt_ids,
                attention_mask=prefill_mask,
                past_key_values=prefill_cache,
                use_cache=True,
                # explicit position tensors: HF derives them from the cache
                # otherwise, which D2H-syncs and aborts stream capture on some
                # model/cache combinations (reproduced with LLaMA + StaticCache)
                cache_position=torch.arange(prompt_len, device=device),
                **self._last_logits_kwargs(module),
            )
            next_token = next_tokens(prefill_out.logits[:, -1, :])

        # --- Copy prefill KV into DeepSpeedStaticCache ---
        # per-row positions: the batched step and decode-attention kernels
        # advance one entry per sequence (a shared scalar would race)
        write_pos = torch.full((batch_size, ), prompt_len, dtype=torch.long, device=device)
        ds_cache = DeepSpeedStaticCache(module.config, max_cache_len=max_len)
        ds_cache.set_write_position(write_pos)
        # Trigger lazy init then copy real data. Hybrid models carry
        # linear-attention (GDN) slots alongside KV slots; the GDN slots are
        # bound by reference to the prefill cache's HF slot objects, whose
        # state management is already cudagraph-safe by construction.
        from deepspeed.utils.static_cache import DSStaticGDNSlot
        # Infer each layer's local KV shape from prefill before graph capture.
        for layer_idx in range(self._cache_layer_count(ds_cache)):
            ds_layer = ds_cache.layers[layer_idx]
            if isinstance(ds_layer, DSStaticGDNSlot):
                ds_layer.bind(prefill_cache.layers[layer_idx])
                continue
            hf_keys, hf_values = self._cache_layer_tensors(prefill_cache, layer_idx)
            ds_layer = ds_cache.layers[layer_idx]
            if not ds_layer.is_initialized:
                ds_layer.lazy_initialization(hf_keys, hf_values)
            ds_layer.keys[:, :, :prompt_len, :].copy_(hf_keys[:, :, :prompt_len, :])
            ds_layer.values[:, :, :prompt_len, :].copy_(hf_values[:, :, :prompt_len, :])

        output_ids.append(next_token)

        # --- Static buffers for graph capture ---
        static_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        # Full-width static causal mask for the full-attention layers; revealed
        # one position per decode step by in-place writes (graph-replay safe).
        # GDN layers must receive None: their recurrence ignores attention masks
        # and a non-None value corrupts the conv-state updates.
        static_attn = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        static_attn[:, :prompt_len] = prompt_attn.bool()
        static_attn[:, prompt_len] = True
        static_causal_attn = torch.zeros((batch_size, 1, 1, max_len), dtype=model_dtype, device=device)
        self._update_static_causal_attention(static_attn, static_causal_attn)

        # Decode-step mask argument: the per-type dict for hybrid models, the
        # plain 4-D tensor for standard full-attention ones.
        decode_mask = ({
            "full_attention": static_causal_attn,
            "linear_attention": None
        } if is_hybrid else static_causal_attn)

        # Persistent decode position tensors as VIEWS of write_pos: the
        # captured forwards read this memory, and the in-graph step kernel
        # advances write_pos itself — so replays see the new position with
        # no host-side mirror updates. (A clone here would freeze the
        # captured value and desync the b=1 full-step path.)
        static_cache_pos = write_pos[:1]
        static_pos_ids = write_pos.unsqueeze(1)

        full_token_buf = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        full_token_buf[:, :prompt_len] = prompt_ids
        full_token_buf[:, prompt_len] = next_token.view(-1)  # first generated token per row

        write_pos.fill_(prompt_len)

        # Replace SDPA in the full-attention layers with the custom
        # decode_attn kernel for the b=1 decode steps (prefill and any
        # non-graph path keep the original forward). Must happen before the
        # warmup forwards so the captured graph records the kernel.
        attn_patched = 0
        module._ki_fused_norm_patched = 0
        if self.use_segki:
            try:
                from deepspeed.module_inject.segment_ki import install_decode_attention, install_fused_norm, install_lm_head
                from deepspeed.ops.module_inject import get_fused_glu_op
                attn_op = get_fused_glu_op()
                # Full-length RoPE tables for the fused q-side path; positions
                # are indexed on-device at replay time, so the addresses stay
                # fixed for the captured graph.
                cos_t = sin_t = None
                rotary = getattr(getattr(module, "model", module), "rotary_emb", None)
                if rotary is not None:
                    with torch.no_grad():
                        positions = torch.arange(max_len, device=device)[None, :]
                        if hasattr(rotary, "mrope_section"):
                            # MRoPE rotary expects [3, batch, seq] positions; text-only
                            # rows use the same positions for all three sections
                            positions = positions[None].expand(3, 1, -1)
                        cos_t, sin_t = rotary(torch.zeros(1, max_len, dtype=model_dtype, device=device), positions)
                        cos_t = cos_t[0].float().contiguous()
                        sin_t = sin_t[0].float().contiguous()
                attn_patched = install_decode_attention(module, write_pos, attn_op, cos_table=cos_t,
                                                        sin_table=sin_t) if hasattr(attn_op, "decode_attn") else 0
                module._ki_fused_norm_patched = install_fused_norm(module, attn_op)
                install_lm_head(module, attn_op)
            except Exception as e:
                # The install chain shares one try: a failure in any part (rope
                # tables, decode attention, fused norm, lm head) silently drops
                # the whole segKI decode path back to the HF forward, which is
                # numerically correct but loses all kernel gains. Make that
                # visible instead of looking like normal slow decoding.
                logger.warning_once(f"segKI decode install failed; rollout falls back to the HF forward "
                                    f"({type(e).__name__}: {e}). Decode will run without the injected kernels.")
                attn_patched = 0
                module._ki_fused_norm_patched = 0

        # Snapshot the GDN states right after prefill: the warmup forwards
        # advance conv/recurrent states by extra steps, so they must be
        # restored before capture or every replay starts from corrupted state.
        gdn_snapshot = []
        for ds_layer in ds_cache.layers:
            if type(ds_layer).__name__ == "DSStaticGDNSlot":
                cs, rs = ds_layer.conv_states[0], ds_layer.recurrent_states[0]
                gdn_snapshot.append((cs, rs, cs.clone(), rs.clone()))

        def restore_gdn_states():
            for cs, rs, cs0, rs0 in gdn_snapshot:
                cs.copy_(cs0)
                rs.copy_(rs0)

        # Remove forward hooks (they synchronize — illegal during graph capture)
        saved_pre = dict(module._forward_pre_hooks)
        saved_post = dict(module._forward_hooks)
        module._forward_pre_hooks.clear()
        module._forward_hooks.clear()

        try:
            # Warmup on side stream
            static_token.copy_(next_token)
            s = get_accelerator().Stream()
            s.wait_stream(get_accelerator().current_stream())
            with get_accelerator().stream(s):
                for _ in range(3):
                    out = module(
                        static_token,
                        attention_mask=decode_mask,
                        past_key_values=ds_cache,
                        use_cache=True,
                        cache_position=static_cache_pos,
                        position_ids=static_pos_ids,
                    )
            get_accelerator().current_stream().wait_stream(s)
            restore_gdn_states()

            # Capture: full-step graph (forward + argmax + buffer updates).
            # The step-update kernel indexes token_buf via write_pos (a GPU
            # tensor the kernel itself advances), so no host-side step counter
            # is needed — the graph is fully self-contained for replay.
            graph_op = None
            step_kernel_mounted = False
            try:
                from deepspeed.ops.module_inject import get_fused_glu_op
                candidate = get_fused_glu_op()
                if hasattr(candidate, "decode_step_graph"):
                    graph_op = candidate
            except Exception:
                graph_op = None

            graph = get_accelerator().create_graph()
            with get_accelerator().capture_to_graph(graph):
                self._update_static_causal_attention(static_attn, static_causal_attn)
                out = module(
                    static_token,
                    attention_mask=decode_mask,
                    past_key_values=ds_cache,
                    use_cache=True,
                    cache_position=static_cache_pos,
                    position_ids=static_pos_ids,
                )
                static_logits = out.logits
                # decode_step_graph kernel is b=1 only (single-sequence argmax);
                # for b>1 the graph captures forward only and Python handles
                # argmax + buffer updates outside the graph. The kernel also
                # only accepts CUDA bf16 logits — fp16 models must keep the
                # PyTorch argmax path instead of failing during capture.
                is_greedy = sampling is None or sampling.temperature <= 0.0
                if (batch_size == 1 and graph_op is not None and is_greedy and repetition_penalty == 1.0
                        and static_logits.dtype is torch.bfloat16 and get_accelerator().on_accelerator(static_logits)):
                    graph_op.decode_step_graph(static_logits[:, -1, :].contiguous(), static_token.view(batch_size, 1),
                                               write_pos, static_attn, full_token_buf)
                    step_kernel_mounted = True

            # The capture run advanced the GDN conv/recurrent states (its
            # forward consumed static_token) and mutated write_pos whether
            # or not decode_step_graph was mounted; restore unconditionally
            # so the fallback graph (native op unavailable) does not start
            # from a state that already consumed the first decode token.
            restore_gdn_states()
            write_pos.fill_(prompt_len)
            static_attn[:, prompt_len + 1:].zero_()
            static_token.copy_(next_token)
        finally:
            module._forward_pre_hooks.update(saved_pre)
            module._forward_hooks.update(saved_post)

        # --- Decode loop: full-step graph (b=1) > graph+Python (b>1) > Python loop ---
        if graph is not None and batch_size > 1:
            # Graph forward + Python argmax for b>1: the graph eliminates kernel
            # launch overhead for the forward pass; argmax and buffer updates
            # stay in Python (PyTorch argmax is batch-native).  The Python
            # overhead (~50us/step) is a smaller fraction of the larger GPU
            # workload at higher batch.
            # 1 = check every step: EOS-heavy rollouts (RL responses end
            # early; max_new_tokens is only a cap) would otherwise replay up
            # to eos_check_every - 1 dead steps after the last row finishes,
            # which costs far more than the per-step sync saves (measured
            # +0.1-0.3% at b=32 without EOS vs up to ~20% with early EOS).
            eos_check_every = 1
            eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for step in range(max_new_tokens - 1):
                static_token.copy_(next_token)
                pos = prompt_len + step
                write_pos.fill_(pos)
                # Reveal only the current token's own slot: the pos+1 slot has
                # not been written yet (warmup/capture leftovers), and SDPA at
                # b>1 attends every slot the mask reveals.
                static_attn[:, pos] = True
                self._update_static_causal_attention(static_attn, static_causal_attn)
                get_accelerator().replay_graph(graph)
                next_token = next_tokens(static_logits[:, -1, :])
                output_ids.append(next_token)
                # eos_mask.all() is a device->host sync; running it every step
                # serializes the ~50us of Python glue against the replay and
                # idles the GPU between steps. Amortize it like the b=1 path:
                # check every eos_check_every steps and break with a bulk pad.
                # Tokens generated after a row's EOS (at most
                # eos_check_every - 1 of them) are padded away by
                # _pad_after_eos, so the output is identical.
                if eos_token_id is None:
                    continue
                eos_mask |= (next_token.view(batch_size) == eos_token_id)
                if step % eos_check_every == eos_check_every - 1 and bool(eos_mask.all()):
                    remaining = max_new_tokens - 2 - step
                    if remaining > 0:
                        output_ids.append(
                            torch.full((batch_size, remaining), pad_token_id, dtype=torch.long, device=device))
                    break
            return torch.cat(output_ids, dim=1)

        # Full-step graph requires the step kernel to actually be in the
        # graph: mounting is conditional (b=1, CUDA bf16 logits), and a
        # kernel-less graph replay would read a token buffer nobody wrote.
        if step_kernel_mounted:
            # This path avoids a device sync on every b=1 replay. EOS can be
            # late by up to 15 steps, which is padded away below; batched
            # graph+Python decode checks each step because one dead replay is
            # multiplied by every active row.
            eos_check_every = 16
            for step in range(max_new_tokens - 1):
                get_accelerator().replay_graph(graph)
                if eos_token_id is not None and step % eos_check_every == eos_check_every - 1:
                    if bool((static_token.view(-1) == eos_token_id).all().item()):
                        break
            # pad each row's tail after its own first EOS (response region only;
            # a prompt may legitimately contain the EOS id as content)
            if eos_token_id is not None:
                resp = full_token_buf[:, prompt_len:]
                is_eos = resp == eos_token_id
                first_eos = torch.where(is_eos.any(dim=1),
                                        is_eos.float().argmax(dim=1), torch.full_like(write_pos[0], resp.shape[1]))
                for r in (first_eos < resp.shape[1]).nonzero(as_tuple=True)[0].tolist():
                    full_token_buf[r, prompt_len + first_eos[r].item() + 1:] = pad_token_id
            gen_ids = full_token_buf[:, prompt_len:prompt_len + max_new_tokens]
            return torch.cat([prompt_ids, gen_ids], dim=1)

        eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for step in range(max_new_tokens - 1):
            if eos_mask.all():
                output_ids.append(torch.full((batch_size, 1), pad_token_id, dtype=torch.long, device=device))
                continue
            static_token.copy_(next_token)
            pos = prompt_len + step
            write_pos.fill_(pos)
            static_attn[:, pos] = True
            self._update_static_causal_attention(static_attn, static_causal_attn)
            get_accelerator().replay_graph(graph)
            next_token = next_tokens(static_logits[:, -1, :])
            output_ids.append(next_token)
            eos_mask |= (next_token.view(1) == eos_token_id)
        return torch.cat(output_ids, dim=1)

    def sync_weights(self, step: int) -> None:  # noqa: ARG002
        """Invalidate derived-weight caches after in-place weight updates.

        The colocated hybrid engine reads weights live, so there is
        nothing to push -- but the injected forwards cache tensors
        derived from the weights, and an in-place write (arena or reshard
        sync) leaves those stale. Every training-step entry goes through
        here, so each sync source invalidates uniformly instead of
        relying on callers to remember refresh_weights().
        """
        self.refresh_weights()

    def shutdown(self) -> None:
        """Release the captured continuous decode graph and its static cache.

        The rollout stays usable and captures again on the next graph rollout.
        Under tensor parallelism the graph holds captured collectives, so release
        it before destroying the process group.
        """
        if self._continuous_graph_state is None:
            return
        get_accelerator().synchronize()
        self._continuous_graph_state = None

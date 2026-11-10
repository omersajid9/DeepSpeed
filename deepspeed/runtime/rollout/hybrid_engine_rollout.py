# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine backed by DeepSpeed's hybrid engine.

Two generation paths:
  1. **model.generate()** (default): delegates to HuggingFace generate.
     Supports sampling (temperature, top_p) and greedy.
  2. **graph capture + DeepSpeedStaticCache**: only for greedy (temperature=0).
     Pre-allocates a StaticCache, captures the decode forward pass into a
     graph, and replays it for each decode step.  Eliminates kernel launch
     overhead.
  3. **continuous batching (experimental)**: an opt-in bounded greedy path
     selected through ``SamplingConfig.continuous_batch_size``.
"""

import time
from copy import copy
from dataclasses import dataclass
from inspect import signature
from typing import Optional

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.continuous_batching import ContinuousBatchRequest, ContinuousBatchScheduler
from deepspeed.utils import logger


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
    enable_cache_trimming: bool = False
    continuous_cache_capacity: Optional[int] = None
    enable_chunked_prefill: bool = False
    prefill_chunk_size: int = 512


@dataclass
class _ContinuousGraphState:
    signature: tuple
    cache: object
    static_input: torch.Tensor
    static_attention: torch.Tensor
    static_causal_attention: torch.Tensor
    static_position_ids: torch.Tensor
    static_write_positions: torch.Tensor
    static_cache_position: torch.Tensor
    graph: object
    static_logits: torch.Tensor


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
        layers = getattr(self.cache, "layers", None)
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
        self.enable_cache_trimming = getattr(cfg, 'enable_cache_trimming', False) if cfg else False
        self.continuous_cache_capacity = getattr(cfg, 'continuous_cache_capacity', None) if cfg else None
        self.enable_chunked_prefill = getattr(cfg, 'enable_chunked_prefill', False) if cfg else False
        self.prefill_chunk_size = getattr(cfg, 'prefill_chunk_size', 0) if cfg else 0
        self._last_profile = None
        self._continuous_graph_state = None
        self._last_continuous_stats = None
        self._chunked_prefill_warned = False

    @torch.no_grad()
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        if sampling.continuous_batch_size is not None:
            return self._generate_continuous(request, sampling, sampling.continuous_batch_size)

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

        if self.enable_profiling:
            accelerator.synchronize()
            expansion_end = time.perf_counter()

        is_greedy = sampling.temperature <= 0.0

        forward_profiler = None
        forward_profile_handles = []
        if self.enable_profiling and not (self.use_graph_capture and is_greedy):
            forward_profiler = _ForwardProfiler(accelerator)
            forward_profile_handles = forward_profiler.register(module)

        shared_prefill_handles = []
        try:
            if self.use_shared_prefill and n > 1:
                if self.use_graph_capture:
                    raise RuntimeError("Shared prefill does not support graph capture")
                self.engine.prepare_shared_prefill(B, n, prompt_len)
                shared_prefill_handles = self._register_shared_prefill_hooks(module, B, n)
            if self.use_graph_capture and is_greedy:
                output_ids = self._generate_graph(prompt_ids, prompt_attn, max_new_tokens, pad_token_id, module,
                                                  device)
            elif (self.enable_chunked_prefill and is_greedy and prompt_len > self.prefill_chunk_size
                  and getattr(module, "_supports_cache_class", None) is not False):
                output_ids = self._generate_from_staged_prefill(prompt_ids, prompt_attn, max_new_tokens, pad_token_id,
                                                                module, device)
            else:
                if self.enable_chunked_prefill and not is_greedy and not self._chunked_prefill_warned:
                    logger.warning("Rollout chunked prefill currently covers greedy generation only; "
                                   "sampling rollouts prefill the prompt in a single forward.")
                    self._chunked_prefill_warned = True
                temperature = max(sampling.temperature, 1e-8)
                do_sample = not is_greedy
                generate_kwargs = {
                    "attention_mask": prompt_attn,
                    "max_new_tokens": max_new_tokens,
                    # ZeRO-3 gathers parameters during each decode forward, so every
                    # data-parallel rank must execute the same number of iterations.
                    "eos_token_id": None,
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else 1.0,
                    "top_p": sampling.top_p if do_sample else 1.0,
                    "pad_token_id": pad_token_id,
                }
                if do_sample:
                    generate_kwargs["top_k"] = max(sampling.top_k, 0)
                output_ids = module.generate(prompt_ids, **generate_kwargs)
        finally:
            for handle in shared_prefill_handles:
                handle.remove()
            for handle in forward_profile_handles:
                handle.remove()

        if self.enable_profiling:
            accelerator.synchronize()
            generation_end = time.perf_counter()

        # Generation deliberately ignores EOS above so ZeRO-3 ranks execute
        # the same number of parameter-gather collectives. Restore the usual
        # generation semantics before returning: retain the first EOS in each
        # response and replace every later token with padding.
        output_ids, response_attn = self._pad_after_eos(
            output_ids,
            response_start=prompt_len,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
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
        original_request = request
        requests = tuple(
            RolloutRequest(request.prompt_ids[index:index + 1], request.prompt_attention_mask[index:index + 1])
            for index in range(request.prompt_ids.shape[0]))
        self._validate_continuous_inputs(requests, sampling, max_batch_size)
        if self.use_graph_capture:
            return self._generate_continuous_graph(original_request, requests, sampling, max_batch_size)

        module = self.engine.module
        profile = self._start_continuous_profile() if self.enable_profiling else None
        profile_accelerator = profile["accelerator"] if profile is not None else None
        profile_start = profile["start"] if profile is not None else None
        prompt_lengths = {
            request_id: (int(request.prompt_attention_mask.sum().item())
                         if self.align_decode_fronts else request.prompt_ids.shape[1])
            for request_id, request in enumerate(requests)
        }
        prompt_len = max(prompt_lengths.values())
        max_positions = getattr(module.config, "max_position_embeddings", None)
        if max_positions is not None:
            if any(length + sampling.max_new_tokens > max_positions for length in prompt_lengths.values()):
                raise ValueError("continuous batching request exceeds the model maximum position embeddings")
        if self.continuous_cache_capacity is not None:
            if self.continuous_cache_capacity < prompt_len:
                raise ValueError("continuous_cache_capacity must be at least the padded prompt length")
            max_cache_len = self.continuous_cache_capacity
        else:
            max_cache_len = (prompt_len + sampling.max_new_tokens if self.align_decode_fronts else
                             self._estimate_continuous_cache_len(prompt_len, [sampling.max_new_tokens] *
                                                                 len(requests), max_batch_size))
        if max_positions is not None and max_cache_len > max_positions:
            raise ValueError("continuous batching cache exceeds the model maximum position embeddings")
        if getattr(module, "_supports_cache_class", None) is False:
            raise ValueError("continuous batching requires a model with cache-class support; use the default "
                             "generate() path or upgrade transformers")

        from transformers import StaticCache
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        device = requests[0].prompt_ids.device
        model_dtype = next(module.parameters()).dtype

        scheduler_start = self._profile_start(profile)
        scheduler = ContinuousBatchScheduler(max_batch_size, sampling.max_new_tokens)
        request_by_id = {}
        responses = {}
        request_order = (sorted(range(len(requests)), key=lambda request_id: -prompt_lengths[request_id])
                         if self.align_decode_fronts else range(len(requests)))
        for request_id in request_order:
            request = requests[request_id]
            scheduler.submit(ContinuousBatchRequest(request_id))
            request_by_id[request_id] = request
            responses[request_id] = []
        self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)

        cache_start = self._profile_start(profile)
        cache = DeepSpeedStaticCache(
            module.config,
            batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=model_dtype,
        )
        write_positions = torch.full((max_batch_size, ), -1, dtype=torch.long, device=device)
        cache.set_write_position(write_positions)
        attention_mask = torch.zeros((max_batch_size, max_cache_len), dtype=torch.long, device=device)
        stats = {
            "cache_capacity":
            max_cache_len,
            "peak_cache_length":
            prompt_len,
            "cache_memory_bytes":
            sum(layer.keys.numel() * layer.keys.element_size() + layer.values.numel() * layer.values.element_size()
                for layer in cache.layers) + attention_mask.numel() * attention_mask.element_size() +
            write_positions.numel() * write_positions.element_size(),
            "trim_count":
            0,
            "trimmed_columns":
            0,
            "trim_latency_ms":
            0.0,
            "decode_steps":
            0,
        }
        next_tokens = {}
        logical_positions = dict(prompt_lengths)
        cache_position = prompt_len
        span_starts = [0] * max_batch_size
        prefill_chunks = {}
        chunk_stats = {"chunks": 0, "peak_staging_bytes": 0}
        chunk_size = self.prefill_chunk_size if self.enable_chunked_prefill else 0

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
                    if self.align_decode_fronts:
                        span_starts[:survivor_count] = [span_starts[index] for index in update.keep_slots]
                        span_starts[survivor_count:] = [0] * (max_batch_size - survivor_count)

                dead_prefix = 0
                if self.enable_cache_trimming:
                    if self.align_decode_fronts:
                        dead_prefix = min(span_starts[:survivor_count])
                    else:
                        trim_threshold = max(1, prompt_len)
                        dead_prefix = self._continuous_dead_prefix(attention_mask, survivor_count)
                        if dead_prefix < trim_threshold and cache_position < max_cache_len - trim_threshold:
                            dead_prefix = 0
                elif cache_position >= max_cache_len:
                    if self.align_decode_fronts:
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
                    if self.align_decode_fronts:
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
                self.align_decode_fronts,
                profile,
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
            if self.align_decode_fronts:
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
                if self.align_decode_fronts:
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
                    attention_mask[decodable_rows, row_positions[decodable_rows]] = 1
                    decode_cache_position = latest_position.reshape(1)
                if self.align_decode_fronts:
                    position_ids = torch.tensor([logical_positions[request_id] for request_id in active_ids],
                                                dtype=torch.long,
                                                device=device).unsqueeze(1)
                else:
                    position_ids = attention_mask[:active_count].sum(dim=1, keepdim=True)
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
        return output

    @torch.no_grad()
    def _generate_continuous_graph(self, original_request, requests, sampling, max_batch_size):
        """Run continuous greedy decoding with a fixed-capacity graph.

        Scheduler slots stay attached to physical cache rows for their whole
        lifetime. Retired rows are masked until a new prompt is prefetched into
        them, so the captured decode always receives the same batch shape and
        static tensor addresses.

        With ``enable_chunked_prefill``, prompts longer than ``prefill_chunk_size``
        are staged and filled across steps instead of in a single forward, so the
        decode rows in the batch keep producing tokens while a long prompt fills.
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

        profile = self._start_continuous_profile() if self.enable_profiling else None
        profile_accelerator = profile["accelerator"] if profile is not None else None
        profile_start = profile["start"] if profile is not None else None
        minimum_cache_len = prompt_len + sampling.max_new_tokens
        max_cache_len = (self.continuous_cache_capacity
                         if self.continuous_cache_capacity is not None else minimum_cache_len)
        if max_cache_len < minimum_cache_len:
            raise ValueError("continuous_cache_capacity must fit the prompt and all generated tokens")
        max_positions = getattr(module.config, "max_position_embeddings", None)
        if max_positions is not None and max_cache_len > max_positions:
            raise ValueError("continuous batching cache exceeds the model maximum position embeddings")
        model_dtype = next(module.parameters()).dtype
        scheduler = ContinuousBatchScheduler(max_batch_size, sampling.max_new_tokens)
        prompt_lengths = {request_id: prompt_len for request_id in range(len(requests))}
        request_by_id = {}
        responses = {}
        for request_id, request in enumerate(requests):
            scheduler.submit(ContinuousBatchRequest(request_id))
            request_by_id[request_id] = request
            responses[request_id] = []

        state = self._get_continuous_graph_state(module, max_batch_size, prompt_len, max_cache_len, device,
                                                 model_dtype, DeepSpeedStaticCache)
        cache = state.cache
        static_input = state.static_input
        static_attention = state.static_attention
        static_causal_attention = state.static_causal_attention
        static_position_ids = state.static_position_ids
        static_write_positions = state.static_write_positions
        graph = state.graph
        static_logits = state.static_logits
        stats = {
            "cache_capacity":
            max_cache_len,
            "peak_cache_length":
            prompt_len,
            "cache_memory_bytes":
            sum(layer.keys.numel() * layer.keys.element_size() + layer.values.numel() * layer.values.element_size()
                for layer in cache.layers) + static_input.numel() * static_input.element_size() +
            static_attention.numel() * static_attention.element_size() +
            static_causal_attention.numel() * static_causal_attention.element_size() +
            static_position_ids.numel() * static_position_ids.element_size() +
            static_write_positions.numel() * static_write_positions.element_size(),
            "trim_count":
            0,
            "trimmed_columns":
            0,
            "trim_latency_ms":
            0.0,
            "trim_bytes_moved":
            0,
            "decode_steps":
            0,
            "chunked_prefill_chunks":
            0,
            "chunked_prefill_staging_bytes":
            0,
        }

        slot_by_request = {}
        write_positions = {}
        next_tokens = {}
        prefill_chunks = {}
        chunk_stats = {"chunks": 0, "peak_staging_bytes": 0}
        chunk_size = self.prefill_chunk_size if self.enable_chunked_prefill else 0
        update = scheduler.schedule()
        while update.active:
            if profile is not None:
                profile["active_batch_sizes"].append(len(update.active))

            for request_id in update.retired:
                static_attention[slot_by_request[request_id]].zero_()
                slot_by_request.pop(request_id, None)
                write_positions.pop(request_id, None)
                next_tokens.pop(request_id, None)

            free_slots = [slot for slot in range(max_batch_size) if slot not in slot_by_request.values()]
            admitted_ids = tuple(request.request_id for request in update.admitted)
            admitted_slots = []
            for request_id in admitted_ids:
                slot = free_slots.pop(0)
                slot_by_request[request_id] = slot
                write_positions[request_id] = prompt_len
                admitted_slots.append(slot)

            cache_start = self._profile_start(profile)
            if admitted_ids:
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
                    profile=profile,
                    target_slots=tuple(admitted_slots),
                    prefill_chunks=prefill_chunks,
                    chunk_size=chunk_size,
                )
                next_tokens.update(admitted_tokens)
            published_ids = set()
            if prefill_chunks:
                chunked_tokens = self._run_chunked_prefill(module,
                                                           prefill_chunks,
                                                           chunk_size,
                                                           device,
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
            self._profile_end(profile, "cache_management_overhead_ms", cache_start)

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
                decode_start = self._profile_start(profile)
                get_accelerator().replay_graph(graph)
                self._profile_end(profile, "decode_forward_ms", decode_start, count="num_decode_forwards")
                decode_logits = torch.cat(
                    [static_logits[slot_by_request[request_id], -1, :].unsqueeze(0) for request_id in decode_ids],
                    dim=0)
                decoded = self._continuous_next_tokens(
                    decode_logits,
                    decode_ids,
                    request_by_id,
                    responses,
                    module,
                )
                for request_id, token in zip(decode_ids, decoded.split(1, dim=0)):
                    decoded_tokens[request_id] = token
                    write_positions[request_id] += 1
                stats["decode_steps"] += 1
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
        stats["chunked_prefill_chunks"] = chunk_stats["chunks"]
        stats["chunked_prefill_staging_bytes"] = chunk_stats["peak_staging_bytes"]
        self._last_continuous_stats = stats
        if profile is not None:
            self._finish_continuous_profile(profile, original_request, responses, max_batch_size, prompt_len,
                                            generation_end, post_processing_end)
        return output

    def _validate_continuous_graph_options(self, prompt_len, sampling):
        if self.align_decode_fronts:
            raise ValueError("continuous graph capture does not support aligned decode fronts")
        if self.enable_cache_trimming:
            raise ValueError("continuous graph capture does not support cache trimming")
        if self.enable_chunked_prefill and self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive when chunked prefill is enabled")
        if (self.continuous_cache_capacity is not None
                and self.continuous_cache_capacity < prompt_len + sampling.max_new_tokens):
            raise ValueError("continuous_cache_capacity must fit the prompt and all generated tokens")

    def _get_continuous_graph_state(self, module, max_batch_size, prompt_len, max_cache_len, device, model_dtype,
                                    cache_type):
        signature = self._continuous_graph_signature(module, max_batch_size, prompt_len, max_cache_len, device,
                                                     model_dtype)
        state = self._continuous_graph_state
        if state is None or state.signature != signature:
            cache = cache_type(
                module.config,
                batch_size=max_batch_size,
                max_cache_len=max_cache_len,
                device=device,
                dtype=model_dtype,
            )
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
            graph, static_logits = self._capture_continuous_graph(
                module,
                cache,
                static_input,
                static_causal_attention,
                static_write_positions,
                static_cache_position,
                static_position_ids,
            )
            state = _ContinuousGraphState(
                signature,
                cache,
                static_input,
                static_attention,
                static_causal_attention,
                static_position_ids,
                static_write_positions,
                static_cache_position,
                graph,
                static_logits,
            )
            self._continuous_graph_state = state

        state.cache.reset()
        state.static_input.zero_()
        state.static_attention.zero_()
        state.static_causal_attention.zero_()
        state.static_position_ids.zero_()
        state.static_write_positions.zero_()
        return state

    @staticmethod
    def _continuous_graph_signature(module, max_batch_size, prompt_len, max_cache_len, device, model_dtype):
        parameter_signature = tuple((parameter.data_ptr(), parameter._version) for parameter in module.parameters())
        return (id(module), device.type, device.index, model_dtype, max_batch_size, prompt_len, max_cache_len,
                parameter_signature)

    @staticmethod
    def _update_static_causal_attention(static_attention, static_causal_attention):
        static_causal_attention.copy_(static_attention[:, None, None, :])
        static_causal_attention.neg_().add_(1).mul_(torch.finfo(static_causal_attention.dtype).min)

    @staticmethod
    def _capture_continuous_graph(module, cache, static_input, static_attention, static_write_positions,
                                  static_cache_position, static_position_ids):
        """Capture one fixed-capacity decode forward and return its static logits."""
        accelerator = get_accelerator()
        saved_pre = dict(module._forward_pre_hooks)
        saved_post = dict(module._forward_hooks)
        module._forward_pre_hooks.clear()
        module._forward_hooks.clear()
        try:
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

    def _validate_continuous_inputs(self, requests, sampling, max_batch_size):
        if not requests:
            raise ValueError("continuous batching requires at least one request")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.use_shared_prefill:
            raise ValueError("continuous batching does not support shared prompt prefill")
        if self.enable_chunked_prefill and self.prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive when chunked prefill is enabled")
        if self.enable_chunked_prefill and self.align_decode_fronts:
            raise ValueError("chunked prefill does not support aligned decode fronts")

        prompt_len = requests[0].prompt_ids.shape[1]
        device = requests[0].prompt_ids.device
        if sampling.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if sampling.temperature > 0:
            raise ValueError("continuous batching currently supports greedy decoding only")
        if sampling.n_samples_per_prompt != 1:
            raise ValueError("continuous batching currently supports one sample per prompt")
        for request in requests:
            if request.prompt_ids.shape[0] != 1:
                raise ValueError("continuous batching requires one prompt row per request")
            if request.prompt_ids.shape[1] != prompt_len:
                raise ValueError("continuous batching currently requires equal padded prompt widths")
            if request.prompt_ids.device != device:
                raise ValueError("continuous batching requests must use the same device")
            if not request.prompt_attention_mask.any():
                raise ValueError("continuous batching requires at least one prompt token per request")

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
                            align_decode_fronts=False,
                            profile=None,
                            target_slots=None,
                            prefill_chunks=None,
                            chunk_size=0):
        if not update.admitted:
            return {}

        if not align_decode_fronts:
            admitted_ids = tuple(request.request_id for request in update.admitted)
            if target_slots is None:
                target_slots = update.admitted_slots
            if len(target_slots) != len(admitted_ids):
                raise ValueError("continuous prefill needs one target slot per admitted request")
            if prefill_chunks is not None and chunk_size > 0:
                admitted_ids, target_slots = self._stage_long_prompts(module, admitted_ids, target_slots,
                                                                      request_by_id, prefill_chunks, chunk_size,
                                                                      static_cache_type, model_dtype, device)
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
                target_layer = cache.layers[layer_idx]
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
        for admitted, target_row in zip(update.admitted, update.admitted_slots):
            request_id = admitted.request_id
            request = request_by_id[request_id]
            prompt_len = prompt_lengths[request_id]
            valid = request.prompt_attention_mask[0].bool()
            prompt_ids = request.prompt_ids[:, valid]
            prompt_attention = torch.ones_like(prompt_ids)
            prefill_start = self._profile_start(profile)
            prefill_tokens, prefill_cache = self._continuous_prefill_cache(
                module,
                static_cache_type,
                prompt_ids,
                prompt_attention,
                model_dtype,
                device,
            )
            self._profile_end(profile, "prefill_forward_ms", prefill_start, count="num_prefill_forwards")
            cache_copy_start = self._profile_start(profile)
            admitted_tokens[request_id] = prefill_tokens
            cache_start = cache_position - prompt_len
            if cache_start < 0:
                raise ValueError("continuous batching prompt does not fit behind the decode front")
            for layer_idx in range(self._cache_layer_count(prefill_cache)):
                prefill_keys, prefill_values = self._cache_layer_tensors(prefill_cache, layer_idx)
                target_layer = cache.layers[layer_idx]
                target_layer.keys[target_row, :, cache_start:cache_position].copy_(prefill_keys[0])
                target_layer.values[target_row, :, cache_start:cache_position].copy_(prefill_values[0])
            attention_mask[target_row, cache_start:cache_position].fill_(1)
            write_positions[target_row] = cache_position
            self._profile_end(profile, "cache_management_overhead_ms", cache_copy_start)
        return admitted_tokens

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
            staging_cache = self._create_static_cache(static_cache_type, module.config, prompt_ids.shape[0],
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
        never of how long the prompt spent staging. Returns the column the row's first
        generated token goes into.
        """
        prompt_end = chunk.prompt_len
        for layer_idx in range(self._cache_layer_count(chunk.cache)):
            staged_keys, staged_values = self._cache_layer_tensors(chunk.cache, layer_idx)
            target_layer = cache.layers[layer_idx]
            target_layer.keys[chunk.slot, :, :prompt_end].copy_(staged_keys[0])
            target_layer.values[chunk.slot, :, :prompt_end].copy_(staged_values[0])
        attention_mask[chunk.slot, :prompt_end].copy_(chunk.prompt_attention[0])
        write_positions[chunk.slot] = prompt_end
        return prompt_end

    def _run_chunked_prefill(self,
                             module,
                             prefill_chunks,
                             chunk_size,
                             device,
                             cache,
                             attention_mask,
                             write_positions,
                             chunk_stats,
                             profile=None,
                             request_write_positions=None):
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
            prompt_end = self._publish_prefill_chunk(chunk, cache, attention_mask, write_positions)
            if request_write_positions is not None:
                request_write_positions[request_id] = prompt_end
            completed[request_id] = first_token
            del prefill_chunks[request_id]
        return completed

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
        """Prefill a prompt batch in windows and return its first response token and cache.

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
        prefill_cache = self._create_static_cache(static_cache_type, module.config, prompt_ids.shape[0], max_cache_len,
                                                  device, model_dtype)
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
        return output.logits[:, -1, :].argmax(dim=-1, keepdim=True), prefill_cache

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

    def _continuous_prefill_cache(self, module, static_cache_type, prompt_ids, prompt_attention, model_dtype, device):
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
                do_sample=False,
                eos_token_id=None,
                pad_token_id=pad_token_id,
                return_dict_in_generate=True,
            )
            return prefill_output.sequences[:, -1:], prefill_output.past_key_values

        prompt_len = prompt_ids.shape[1]
        prefill_cache = self._create_static_cache(static_cache_type, module.config, prompt_ids.shape[0], prompt_len,
                                                  device, model_dtype)
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
        return prefill_output.logits[:, -1, :].argmax(dim=-1, keepdim=True), prefill_cache

    @staticmethod
    def _continuous_next_tokens(logits, request_ids, request_by_id, responses, module):
        repetition_penalty = getattr(getattr(module, "generation_config", None), "repetition_penalty", 1.0)
        if repetition_penalty is None:
            repetition_penalty = 1.0
        if repetition_penalty == 1.0:
            return logits.argmax(dim=-1, keepdim=True)

        next_tokens = []
        for row, request_id in enumerate(request_ids):
            input_ids = torch.cat((request_by_id[request_id].prompt_ids, *responses[request_id]), dim=1)
            row_logits = logits[row:row + 1].clone()
            scores = row_logits.gather(1, input_ids)
            scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
            row_logits.scatter_(1, input_ids, scores)
            next_tokens.append(row_logits.argmax(dim=-1, keepdim=True))
        return torch.cat(next_tokens, dim=0)

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

    @staticmethod
    def _create_static_cache(static_cache_type, config, batch_size, max_cache_len, device, dtype):
        """Construct StaticCache across Transformers' batch-size API variants."""
        cache_config = config
        if not hasattr(config, "num_key_value_heads") or config.num_key_value_heads is None:
            cache_config = copy(config)
            cache_config.num_key_value_heads = config.num_attention_heads
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
    def _pad_after_eos(output_ids, response_start, eos_token_id, pad_token_id):
        """Retain the first response EOS and pad every subsequent position."""
        response_ids = output_ids[:, response_start:]
        response_attn = (response_ids != pad_token_id)

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
        response_attn = ((response_ids != pad_token_id) | first_eos) & ~after_first_eos
        return output_ids, response_attn.long()

    # ------------------------------------------------------------------
    # Graph capture decode loop (greedy only)
    # ------------------------------------------------------------------

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
        first_token, cache = self._chunked_prefill_cache(module,
                                                         StaticCache,
                                                         prompt_ids,
                                                         prompt_attn,
                                                         model_dtype,
                                                         device,
                                                         self.prefill_chunk_size,
                                                         max_cache_len=prompt_len + max_new_tokens)
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

    def _generate_graph(self, prompt_ids, prompt_attn, max_new_tokens, pad_token_id, module, device):
        """Greedy decode with DeepSpeedStaticCache + graph capture."""
        from transformers import StaticCache
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        max_len = prompt_len + max_new_tokens
        eos_token_id = self.tokenizer.eos_token_id
        model_dtype = next(module.parameters()).dtype

        # --- Prefill with HF StaticCache (correct attention semantics) ---
        prefill_attn = torch.ones(batch_size, prompt_len, dtype=torch.long, device=device)
        prefill_attn[:, :prompt_len] = prompt_attn
        if self.enable_chunked_prefill and prompt_len > self.prefill_chunk_size:
            next_token, prefill_cache = self._chunked_prefill_cache(module, StaticCache, prompt_ids, prefill_attn,
                                                                    model_dtype, device, self.prefill_chunk_size)
        else:
            prefill_cache = self._create_static_cache(StaticCache, module.config, batch_size, max_len, device,
                                                      model_dtype)
            prefill_out = module(
                prompt_ids,
                attention_mask=prefill_attn,
                past_key_values=prefill_cache,
                use_cache=True,
                cache_position=torch.arange(prompt_len, device=device),
                **self._last_logits_kwargs(module),
            )
            next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # --- Copy prefill KV into DeepSpeedStaticCache ---
        write_pos = torch.tensor(prompt_len - 1, dtype=torch.long, device=device)
        ds_cache = DeepSpeedStaticCache(
            module.config,
            batch_size=batch_size,
            max_cache_len=max_len,
            device=device,
            dtype=model_dtype,
        )
        ds_cache.set_write_position(write_pos)
        # Trigger lazy init then copy real data
        for layer_idx in range(self._cache_layer_count(ds_cache)):
            ds_layer = ds_cache.layers[layer_idx]
            hf_keys, hf_values = self._cache_layer_tensors(prefill_cache, layer_idx)
            if not ds_layer.is_initialized:
                ds_layer.lazy_initialization(hf_keys, hf_values)
            ds_layer.keys[:, :, :prompt_len, :].copy_(hf_keys[:, :, :prompt_len, :])
            ds_layer.values[:, :, :prompt_len, :].copy_(hf_values[:, :, :prompt_len, :])

        output_ids = [prompt_ids, next_token]

        # --- Static buffers for graph capture ---
        static_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        static_attn = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        static_attn[:, :prompt_len] = prompt_attn
        static_attn[:, prompt_len] = 1  # first decode position
        static_pos = torch.tensor(prompt_len, dtype=torch.long, device=device)
        static_cache_pos = static_pos.unsqueeze(0)  # [1] for cache_position
        static_pos_ids = static_pos.reshape(1, 1).expand(batch_size, 1)  # [batch, 1]

        write_pos.fill_(prompt_len)

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
                        attention_mask=static_attn,
                        past_key_values=ds_cache,
                        use_cache=True,
                        cache_position=static_cache_pos,
                        position_ids=static_pos_ids,
                    )
            get_accelerator().current_stream().wait_stream(s)

            # Capture
            graph = get_accelerator().create_graph()
            with get_accelerator().capture_to_graph(graph):
                out = module(
                    static_token,
                    attention_mask=static_attn,
                    past_key_values=ds_cache,
                    use_cache=True,
                    cache_position=static_cache_pos,
                    position_ids=static_pos_ids,
                )
            static_logits = out.logits
        finally:
            module._forward_pre_hooks.update(saved_pre)
            module._forward_hooks.update(saved_post)

        # --- Decode loop ---
        eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for step in range(max_new_tokens - 1):
            if eos_mask.all():
                output_ids.append(torch.full((batch_size, 1), pad_token_id, dtype=torch.long, device=device))
                continue

            # Update static inputs
            static_token.copy_(next_token)
            pos = prompt_len + step
            write_pos.fill_(pos)
            static_cache_pos.fill_(pos)
            static_pos_ids.fill_(pos)
            static_attn[:, pos] = 1

            # Replay
            get_accelerator().replay_graph(graph)
            next_token = static_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            output_ids.append(next_token)
            eos_mask |= (next_token.squeeze(1) == eos_token_id)

        return torch.cat(output_ids, dim=1)

    def sync_weights(self, step: int) -> None:  # noqa: ARG002
        """No-op: hybrid engine reads model weights live."""
        return None

# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Rollout engine backed by DeepSpeed's hybrid engine.

Generation paths:
  1. **device-side sampling**: a static cache plus one device-side draw per step
     (``deepspeed.runtime.rollout.sampling``).  Used for rollouts that sample and
     record per-token log-probabilities.
  2. **graph capture + DeepSpeedStaticCache**: only for greedy (temperature=0).
     Pre-allocates a StaticCache, captures the decode forward pass into a
     graph, and replays it for each decode step.  Eliminates kernel launch
     overhead.
  3. **chunked prefill**: opt-in staging of long prompts so the prefill advances
     in bounded windows instead of one large forward (greedy paths).
  4. **continuous batching (experimental)**: an opt-in bounded greedy path
     selected through ``SamplingConfig.continuous_batch_size``. With graph
     capture it dispatches each step to the smallest captured batch width from a
     ladder - separately for the decode step and for the chunked-prefill window -
     so a rollout never replays a wider graph than the rows it is running.
  5. **model.generate()** (fallback): delegates to HuggingFace generate for
     models or option combinations the paths above do not cover.
"""

import time
from copy import copy
from dataclasses import dataclass, field
from inspect import signature
from typing import Dict, Optional, Tuple

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutEngine, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.continuous_batching import ContinuousBatchRequest, ContinuousBatchScheduler
from deepspeed.runtime.rollout.sampling import sample
from deepspeed.utils import logger


@torch._dynamo.disable
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

    @torch._dynamo.disable
    def get_stage_times(self):
        if not self.measurements:
            return None, None, 0

        if not self.use_host_timer:
            get_accelerator().synchronize()
        durations = [self._elapsed_time(start, end) for start, end in self.measurements]
        return durations[0], sum(durations[1:]), len(durations) - 1

    @torch._dynamo.disable
    def _start_forward(self, _module, _args):
        self.active_start = self._record_time()

    @torch._dynamo.disable
    def _end_forward(self, _module, _args, _output):
        end = self._record_time()
        self.measurements.append((self.active_start, end))
        self.active_start = None

    @torch._dynamo.disable
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
    Sampling rollouts prefill in a single forward and log a warning. The
    cost is one staging cache per prompt being filled, sized to that prompt's
    effective length.

    ``graph_capture_batch_sizes`` controls the captured-batch ladder of the
    continuous graph path. The loop dispatches each step to the smallest
    captured batch that fits the rows doing work, so a request arriving on its
    own replays a one-row graph instead of the full batch, and a step that
    stages one prompt captures prefill chunks at one row instead of padding up
    to the whole batch. Defaults to the powers of two up to ``max_batch_size``
    plus ``max_batch_size`` itself. Capture is lazy: a bucket is captured the
    first time the loop needs it, and a bucket whose capture fails runs that
    forward eagerly instead.
    """
    use_graph_capture: bool = False
    enable_profiling: bool = False
    use_shared_prefill: bool = False
    align_decode_fronts: bool = False
    enable_cache_trimming: bool = False
    continuous_cache_capacity: Optional[int] = None
    enable_chunked_prefill: bool = False
    prefill_chunk_size: int = 512
    graph_capture_batch_sizes: Optional[Tuple[int, ...]] = None
    use_paged_cache: bool = False
    paged_block_size: Optional[int] = None


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
    row: int
    prompt_ids: torch.Tensor
    prompt_attention: torch.Tensor
    position_ids: torch.Tensor
    start: int = 0

    @property
    def prompt_len(self):
        """Padded prompt width, which every request in a continuous batch shares."""
        return self.prompt_ids.shape[1]


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
        self.graph_capture_batch_sizes = getattr(cfg, 'graph_capture_batch_sizes', None) if cfg else None
        self.use_paged_cache = getattr(cfg, 'use_paged_cache', False) if cfg else False
        self.paged_block_size = getattr(cfg, 'paged_block_size', None) if cfg else None
        self._last_profile = None
        self._continuous_graph_state = None
        self._last_continuous_stats = None
        self._chunked_prefill_warned = False
        if self.use_paged_cache:
            from deepspeed.runtime.rollout.paged_cache import install_paged_attention
            install_paged_attention(self.engine.module)

    @torch.no_grad()
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        if sampling.continuous_batch_size is not None:
            return self._generate_continuous(request, sampling, sampling.continuous_batch_size)
        if self.use_paged_cache:
            raise ValueError("paged cache requires continuous batching (set SamplingConfig.continuous_batch_size)")

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
        response_logprobs = None
        try:
            if self.use_shared_prefill and n > 1:
                if self.use_graph_capture:
                    raise RuntimeError("Shared prefill does not support graph capture")
                self.engine.prepare_shared_prefill(B, n, prompt_len)
                shared_prefill_handles = self._register_shared_prefill_hooks(module, B, n)
            if self.use_graph_capture:
                output_ids, response_logprobs = self._generate_graph(prompt_ids, prompt_attn, max_new_tokens,
                                                                     pad_token_id, module, device, sampling)
            elif (self.enable_chunked_prefill and is_greedy and prompt_len > self.prefill_chunk_size
                  and getattr(module, "_supports_cache_class", None) is not False):
                output_ids = self._generate_from_staged_prefill(prompt_ids, prompt_attn, max_new_tokens, pad_token_id,
                                                                module, device)
            elif (not is_greedy and not self.use_shared_prefill
                  and getattr(module, "_supports_cache_class", None) is not False):
                output_ids, response_logprobs = self._generate_sampling(prompt_ids, prompt_attn, max_new_tokens,
                                                                        module, device, sampling)
            else:
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
                if getattr(module, "_supports_cache_class", None) is not False and not self.use_shared_prefill:
                    # HuggingFace's default cache reallocates and concatenates K/V on every decode
                    # step; a static cache writes in place. Shared prefill manages the cache itself.
                    generate_kwargs["cache_implementation"] = "static"
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
        if response_logprobs is not None:
            # Keep the log-probabilities consistent with the mask: positions the pad-after-EOS
            # pass retires report zero rather than the probability of a token that no longer exists.
            response_logprobs = response_logprobs.masked_fill(response_attn == 0, 0.0)

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
            response_logprobs=response_logprobs,
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
        """Generate independent requests in an experimental continuous batch.

        Continuous batching requires one prompt row per request and one sample per prompt.
        Completed rows retire immediately and pending prompts prefill into the released rows
        before the next decode step. Greedy rows take the argmax of each step's logits;
        sampling rows draw them on the device. Sampling rollouts prefill in a single forward
        (staging is greedy-only), so the first response token comes from the sampler too.
        """
        original_request = request
        requests = tuple(
            RolloutRequest(request.prompt_ids[index:index + 1], request.prompt_attention_mask[index:index + 1])
            for index in range(request.prompt_ids.shape[0]))
        self._validate_continuous_inputs(requests, sampling, max_batch_size)
        is_greedy = sampling.temperature <= 0.0
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
        if self.use_paged_cache:
            from deepspeed.runtime.rollout.paged_cache import PagedAttentionCache
            cache = PagedAttentionCache(
                module.config,
                batch_size=max_batch_size,
                max_cache_len=max_cache_len,
                device=device,
                dtype=model_dtype,
                block_size=self.paged_block_size,
            )
        else:
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
        if self.use_paged_cache:
            cache_memory_bytes = cache.memory_bytes
        else:
            cache_memory_bytes = sum(layer.keys.numel() * layer.keys.element_size() +
                                     layer.values.numel() * layer.values.element_size() for layer in cache.layers)
        stats = {
            "cache_capacity":
            max_cache_len,
            "peak_cache_length":
            prompt_len,
            "cache_memory_bytes":
            cache_memory_bytes + attention_mask.numel() * attention_mask.element_size() +
            write_positions.numel() * write_positions.element_size(),
            "trim_count":
            0,
            "trimmed_columns":
            0,
            "trim_latency_ms":
            0.0,
            "decode_steps":
            0,
            "paged_cache":
            self.use_paged_cache,
        }
        next_tokens = {}
        next_logprobs = {}
        response_logprobs = {request_id: [] for request_id in request_by_id} if not is_greedy else None
        logical_positions = dict(prompt_lengths)
        cache_position = prompt_len
        span_starts = [0] * max_batch_size
        prefill_chunks = {}
        chunk_stats = {"chunks": 0, "peak_staging_bytes": 0}
        # Long prompts are staged for every decoding mode; the completing chunk draws the
        # first response token with the same sampler as the decode steps.
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
                    if self.use_paged_cache:
                        if dead_prefix % cache.block_size:
                            raise RuntimeError("paged cache trimming must be block aligned; increase "
                                               "continuous_cache_capacity or disable trimming")
                        moved_bytes = 0
                    else:
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
            admitted_tokens, admitted_logprobs = self._continuous_prefill(
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
                sampling=sampling,
            )
            if admitted_logprobs is not None:
                next_logprobs.update(admitted_logprobs)
            published_ids = set()
            if prefill_chunks:
                published, published_logprobs = self._run_chunked_prefill(module,
                                                                          prefill_chunks,
                                                                          chunk_size,
                                                                          device,
                                                                          cache,
                                                                          attention_mask,
                                                                          write_positions,
                                                                          chunk_stats,
                                                                          profile,
                                                                          sampling=sampling)
                admitted_tokens.update(published)
                next_logprobs.update(published_logprobs)
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
                if self.use_paged_cache:
                    model_input = decode_input[decodable_rows]
                    model_positions = row_positions[decodable_rows]
                    cache.set_active_rows(decodable_rows)
                    model_attention_mask = {"full_attention": None, "sliding_attention": None}
                    model_position_ids = model_positions.unsqueeze(1)
                    model_cache_position = model_positions
                else:
                    model_input = decode_input
                    model_attention_mask = attention_mask[:active_count]
                    if self.align_decode_fronts:
                        model_position_ids = torch.tensor([logical_positions[request_id] for request_id in active_ids],
                                                          dtype=torch.long,
                                                          device=device).unsqueeze(1)
                    else:
                        model_position_ids = attention_mask[:active_count].sum(dim=1, keepdim=True)
                    model_cache_position = decode_cache_position
                decode_start = self._profile_start(profile)
                output = self._call_model(
                    module,
                    model_input,
                    attention_mask=model_attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=model_cache_position,
                    position_ids=model_position_ids,
                )
                self._profile_end(profile, "decode_forward_ms", decode_start, count="num_decode_forwards")
                if self.use_paged_cache:
                    decode_logits = output.logits[:, -1, :]
                else:
                    decode_logits = output.logits[decodable_rows, -1, :]
                decoded, decoded_logprobs = self._continuous_next_tokens(
                    decode_logits,
                    decodable_ids,
                    request_by_id,
                    responses,
                    module,
                    sampling,
                )
                stats["decode_steps"] += 1
                decoded_tokens = dict(zip(decodable_ids, decoded.split(1, dim=0)))
                if decoded_logprobs is not None:
                    next_logprobs.update(dict(zip(decodable_ids, decoded_logprobs.split(1, dim=0))))
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
                if response_logprobs is not None:
                    response_logprobs[request_id].append(next_logprobs.pop(request_id))
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
        output = self._build_continuous_batch(original_request, responses, response_logprobs)
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
        if self.use_paged_cache:
            stats["paged_block_size"] = cache.block_size
            stats["blocks_in_use"] = cache.blocks_in_use
            stats["block_utilization"] = cache.blocks_in_use / cache.block_ids.numel()
        self._last_continuous_stats = stats
        if profile is not None:
            self._finish_continuous_profile(profile, original_request, responses, max_batch_size, prompt_len,
                                            generation_end, post_processing_end)
        return output

    @torch.no_grad()
    def _generate_continuous_graph(self, original_request, requests, sampling, max_batch_size):
        """Run continuous greedy decoding with captured graphs.

        Scheduler slots stay attached to physical cache rows for their whole lifetime.
        Retired rows are masked until a new prompt is prefetched into them, so a captured
        forward always receives the same batch shape and static tensor addresses.

        Each step dispatches to the smallest captured batch that covers the rows doing
        work, so a step runs at most two replays: the chunk window of the staged prompts,
        then the decode token of the rows that are already generating. Buckets are captured
        the first time the loop needs them, and a bucket whose capture fails runs that
        forward eagerly from the same static buffers.

        With ``enable_chunked_prefill``, prompts longer than ``prefill_chunk_size`` are
        staged in one shared cache and filled a chunk at a time, so the decode rows in the
        batch keep producing tokens while a long prompt fills.
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
        is_greedy = sampling.temperature <= 0.0
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
        staging_cache = state.staging_cache
        staging_bytes = self._cache_bytes(staging_cache)
        captures_before = state.captures
        stats = {
            "cache_capacity":
            max_cache_len,
            "peak_cache_length":
            prompt_len,
            "cache_memory_bytes":
            self._cache_bytes(cache) + staging_bytes + static_input.numel() * static_input.element_size() +
            static_attention.numel() * static_attention.element_size() +
            static_causal_attention.numel() * static_causal_attention.element_size() +
            static_position_ids.numel() * static_position_ids.element_size() +
            static_write_positions.numel() * static_write_positions.element_size(),
            "graph_capture_ladder":
            list(state.ladder),
            "graph_captures":
            0,
            "decode_bucket_steps":
            0,
            "decode_padded_rows":
            0,
            "decode_bucket_widths": {},
            "prefill_bucket_steps":
            0,
            "prefill_padded_rows":
            0,
            "prefill_bucket_widths": {},
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
            staging_bytes,
        }

        slot_by_request = {}
        write_positions = {}
        next_tokens = {}
        next_logprobs = {}
        response_logprobs = {request_id: [] for request_id in request_by_id} if not is_greedy else None
        prefill_chunks = {}
        chunk_stats = {"chunks": 0, "peak_staging_bytes": 0}
        # Long prompts are staged for every decoding mode; the completing chunk draws the
        # first response token with the same sampler as the decode steps.
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

            cache_start = self._profile_start(profile)
            if admitted_ids:
                admitted_tokens, admitted_logprobs = self._continuous_prefill(
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
                    stage_prompts=self._stage_staged_prompt,
                    sampling=sampling,
                )
                next_tokens.update(admitted_tokens)
                if admitted_logprobs is not None:
                    next_logprobs.update(admitted_logprobs)
            published_ids = set()
            if prefill_chunks:
                chunked_tokens, chunked_logprobs = self._run_staged_prefill(module,
                                                                            prefill_chunks,
                                                                            chunk_size,
                                                                            prompt_len,
                                                                            cache,
                                                                            static_attention,
                                                                            static_write_positions,
                                                                            chunk_stats,
                                                                            profile,
                                                                            request_write_positions=write_positions,
                                                                            sampling=sampling)
                next_tokens.update(chunked_tokens)
                next_logprobs.update(chunked_logprobs)
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
                # Only the rows that are decoding need to be covered, so the step replays the
                # smallest captured batch that holds them instead of the whole physical batch.
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
                decoded, decoded_logprobs = self._continuous_next_tokens(
                    decode_logits,
                    decode_ids,
                    request_by_id,
                    responses,
                    module,
                    sampling,
                )
                for request_id, token in zip(decode_ids, decoded.split(1, dim=0)):
                    decoded_tokens[request_id] = token
                    write_positions[request_id] += 1
                if decoded_logprobs is not None:
                    next_logprobs.update(dict(zip(decode_ids, decoded_logprobs.split(1, dim=0))))
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
                if response_logprobs is not None:
                    response_logprobs[request_id].append(next_logprobs.pop(request_id))
                if self._is_eos(token):
                    finished_ids.append(request_id)

            scheduler_start = self._profile_start(profile)
            update = scheduler.advance(finished_ids, prefilling_ids=tuple(prefill_chunks))
            self._profile_end(profile, "scheduler_overhead_ms", scheduler_start)
            if write_positions:
                stats["peak_cache_length"] = max(stats["peak_cache_length"], max(write_positions.values()))

        generation_end = self._profile_start(profile)
        output = self._build_continuous_batch(original_request, responses, response_logprobs)
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
        if self.use_paged_cache:
            stats["paged_cache"] = True
            stats["paged_block_size"] = cache.block_size
            stats["blocks_in_use"] = cache.blocks_in_use
            stats["block_utilization"] = cache.blocks_in_use / cache.block_ids.numel()
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

    def _get_continuous_graph_state(self, module, max_batch_size, prompt_len, max_cache_len, device, model_dtype,
                                    cache_type):
        ladder = self._graph_capture_ladder(max_batch_size, self.graph_capture_batch_sizes)
        signature = self._continuous_graph_signature(module, max_batch_size, prompt_len, max_cache_len, device,
                                                     model_dtype, ladder)
        signature = signature + (self.use_paged_cache, self.paged_block_size)
        state = self._continuous_graph_state
        if state is None or state.signature != signature:
            if self.use_paged_cache:
                from deepspeed.runtime.rollout.paged_cache import PagedAttentionCache
                cache = PagedAttentionCache(
                    module.config,
                    batch_size=max_batch_size,
                    max_cache_len=max_cache_len,
                    device=device,
                    dtype=model_dtype,
                    block_size=self.paged_block_size,
                )
                cache.enable_graph_safe()
            else:
                cache = cache_type(
                    module.config,
                    batch_size=max_batch_size,
                    max_cache_len=max_cache_len,
                    device=device,
                    dtype=model_dtype,
                )
            staging_cache = None
            if self.enable_chunked_prefill:
                # One staging cache serves every staged prompt and every captured prefill
                # width: rows are staged prompts and columns are the padded prompt width,
                # which is the same for every request in a continuous batch. Rows are
                # reused across prompts, but a staged prompt always writes its own columns
                # forward before the attention can read them, so stale columns are never
                # read and the cache does not need to be cleared between rollouts.
                from deepspeed.utils.static_cache import DeepSpeedStaticCache
                staging_cache = self._create_static_cache(DeepSpeedStaticCache, module.config, max_batch_size,
                                                          prompt_len, device, model_dtype)
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
            # Captured chunk forwards short-circuit on the additive mask, so their
            # cache_position is not read at all.
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

    @staticmethod
    def _bucket_attention_mask(cache, bucket):
        """Paged decode passes a dict so Qwen skips the dense causal-mask build."""
        from deepspeed.runtime.rollout.paged_cache import PagedAttentionCache
        if isinstance(cache, PagedAttentionCache):
            cache.set_graph_mode(bucket.width)
            return {"full_attention": None, "sliding_attention": None}
        return bucket.causal_attention

    def _capture_bucket(self, module, cache, bucket, cache_position, write_position):
        """Capture one bucket's forward, falling back to eager forwards if capture fails.

        The capture runs the model for real during its warmup, with the buffers the next
        replay will use, so it must be called only once the caller has filled them.
        """
        try:
            from deepspeed.runtime.rollout.paged_cache import PagedAttentionCache
            is_paged = isinstance(cache, PagedAttentionCache)
            if is_paged:
                cache.set_write_position(self._continuous_graph_state.static_write_positions)
            else:
                cache.set_write_position(write_position)
            attention_mask = self._bucket_attention_mask(cache, bucket)
            if is_paged:
                cache.update_graph_atoms()
            bucket.graph, bucket.logits = self._capture_continuous_graph(module, cache, bucket.input_ids,
                                                                         attention_mask, write_position,
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
        from deepspeed.runtime.rollout.paged_cache import PagedAttentionCache
        is_paged = isinstance(cache, PagedAttentionCache)
        attention_mask = self._bucket_attention_mask(cache, bucket)
        if is_paged:
            cache.update_graph_atoms()
        if bucket.graph is not None:
            get_accelerator().replay_graph(bucket.graph)
            return bucket.logits
        # The eager fallback reads the same static buffers, so it stays token-identical
        # to a replay; only the launch overhead differs.
        if is_paged:
            cache.set_write_position(self._continuous_graph_state.static_write_positions)
        else:
            cache.set_write_position(write_position)
        output = self._call_model(
            module,
            bucket.input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            position_ids=bucket.position_ids,
            **self._last_logits_kwargs(module),
        )
        return output.logits

    @staticmethod
    def _continuous_graph_signature(module, max_batch_size, prompt_len, max_cache_len, device, model_dtype, ladder):
        parameter_signature = tuple((parameter.data_ptr(), parameter._version) for parameter in module.parameters())
        return (id(module), device.type, device.index, model_dtype, max_batch_size, prompt_len, max_cache_len, ladder,
                parameter_signature)

    @staticmethod
    def _update_static_causal_attention(static_attention, static_causal_attention):
        static_causal_attention.copy_(static_attention[:, None, None, :])
        static_causal_attention.neg_().add_(1).mul_(torch.finfo(static_causal_attention.dtype).min)

    @staticmethod
    def _capture_continuous_graph(module,
                                  cache,
                                  static_input,
                                  attention_mask,
                                  static_write_positions,
                                  static_cache_position,
                                  static_position_ids,
                                  logits_kwargs=None):
        """Capture one fixed-shape forward and return its graph and static logits.

        The caller fills the static buffers before capturing, because the warmup forwards
        run the model for real: they write the same KV the following replay writes.
        """
        logits_kwargs = logits_kwargs or {}
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
                        attention_mask=attention_mask,
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
                    attention_mask=attention_mask,
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
        if self.use_paged_cache:
            if self.align_decode_fronts:
                raise ValueError("paged cache does not support aligned decode fronts yet")
            if self.enable_cache_trimming:
                raise ValueError("paged cache only supports block-aligned trimming")

        prompt_len = requests[0].prompt_ids.shape[1]
        device = requests[0].prompt_ids.device
        if sampling.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
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
                            chunk_size=0,
                            stage_prompts=None,
                            sampling=None):
        """Prefill newly admitted prompts and return their first response tokens.

        Returns ``(tokens, logprobs)`` keyed by request id; ``logprobs`` is ``None`` for a
        greedy rollout, and the first response token of a sampling rollout is drawn from the
        prefill distribution rather than argmaxed.
        """
        if not update.admitted:
            return {}, None

        if not align_decode_fronts:
            admitted_ids = tuple(request.request_id for request in update.admitted)
            if target_slots is None:
                target_slots = update.admitted_slots
            if len(target_slots) != len(admitted_ids):
                raise ValueError("continuous prefill needs one target slot per admitted request")
            if prefill_chunks is not None and chunk_size > 0:
                stager = stage_prompts if stage_prompts is not None else self._stage_long_prompts
                admitted_ids, target_slots = stager(module, admitted_ids, target_slots, request_by_id, prefill_chunks,
                                                    chunk_size, static_cache_type, model_dtype, device)
                if not admitted_ids:
                    return {}, None
            prompt_ids = torch.cat([request_by_id[request_id].prompt_ids for request_id in admitted_ids], dim=0)
            prompt_attention = torch.cat(
                [request_by_id[request_id].prompt_attention_mask for request_id in admitted_ids], dim=0)
            prompt_len = prompt_ids.shape[1]
            prefill_start = self._profile_start(profile)
            prefill_tokens, prefill_logprobs, prefill_cache = self._continuous_prefill_cache(
                module,
                static_cache_type,
                prompt_ids,
                prompt_attention,
                model_dtype,
                device,
                sampling=sampling,
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
            if self.use_paged_cache:
                for layer_idx in range(self._cache_layer_count(prefill_cache)):
                    prefill_keys, prefill_values = self._cache_layer_tensors(prefill_cache, layer_idx)
                    for source_row, target_row in enumerate(target_slots):
                        valid = prompt_attention[source_row].bool()
                        cache.copy_from_dense(layer_idx, prefill_keys[source_row], prefill_values[source_row],
                                              target_row, valid)
                for source_row, target_row in enumerate(target_slots):
                    valid_len = int(prompt_attention[source_row].sum())
                    attention_mask[target_row].zero_()
                    attention_mask[target_row, :valid_len] = 1
                    write_positions[target_row] = valid_len
            else:
                for layer_idx in range(self._cache_layer_count(prefill_cache)):
                    prefill_keys, prefill_values = self._cache_layer_tensors(prefill_cache, layer_idx)
                    target_layer = cache.layers[layer_idx]
                    for source_row, target_row in enumerate(target_slots):
                        target_layer.keys[target_row, :,
                                          cache_start:cache_start + prompt_len].copy_(prefill_keys[source_row])
                        target_layer.values[target_row, :,
                                            cache_start:cache_start + prompt_len].copy_(prefill_values[source_row])
                for source_row, target_row in enumerate(target_slots):
                    attention_mask[target_row,
                                   cache_start:cache_start + prompt_len].copy_(prompt_attention[source_row])
                    write_positions[target_row] = cache_start + prompt_len
            self._profile_end(profile, "cache_management_overhead_ms", cache_copy_start)
            tokens = dict(zip(admitted_ids, prefill_tokens.split(1, dim=0)))
            if prefill_logprobs is None:
                return tokens, None
            return tokens, dict(zip(admitted_ids, prefill_logprobs.split(1, dim=0)))

        admitted_tokens = {}
        admitted_logprobs = {} if sampling is not None and sampling.temperature > 0.0 else None
        for admitted, target_row in zip(update.admitted, update.admitted_slots):
            request_id = admitted.request_id
            request = request_by_id[request_id]
            prompt_len = prompt_lengths[request_id]
            valid = request.prompt_attention_mask[0].bool()
            prompt_ids = request.prompt_ids[:, valid]
            prompt_attention = torch.ones_like(prompt_ids)
            prefill_start = self._profile_start(profile)
            prefill_tokens, prefill_logprobs, prefill_cache = self._continuous_prefill_cache(
                module,
                static_cache_type,
                prompt_ids,
                prompt_attention,
                model_dtype,
                device,
                sampling=sampling,
            )
            self._profile_end(profile, "prefill_forward_ms", prefill_start, count="num_prefill_forwards")
            cache_copy_start = self._profile_start(profile)
            admitted_tokens[request_id] = prefill_tokens
            if admitted_logprobs is not None:
                admitted_logprobs[request_id] = prefill_logprobs
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
        return admitted_tokens, admitted_logprobs

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

    def _advance_prefill_chunk(self, module, chunk, width, device, profile=None, sampling=None):
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
            return None, None
        # The completing chunk predicts the first response token; sampling rollouts must
        # draw it from the same distribution as the rest of the response.
        return self._draw_token(output.logits[:, -1, :], sampling)

    def _publish_prefill_chunk(self, chunk, cache, attention_mask, write_positions):
        """Move a completed staged prompt into its row of the decode cache.

        The prompt lands at column zero of its own row: the cache is row-major, so a row's
        columns are its own and do not compete with the decode front of the rows around it.
        That keeps the cache capacity a function of the prompt and the response budget only,
        never of how long the prompt spent staging. Returns the column the row's first
        generated token goes into.
        """
        prompt_end = chunk.prompt_len
        if self.use_paged_cache:
            for layer_idx in range(self._cache_layer_count(chunk.cache)):
                staged_keys, staged_values = self._cache_layer_tensors(chunk.cache, layer_idx)
                cache.copy_from_dense(layer_idx, staged_keys[0], staged_values[0], chunk.slot)
            attention_mask[chunk.slot].zero_()
            attention_mask[chunk.slot, :prompt_end] = 1
        else:
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
                             request_write_positions=None,
                             sampling=None):
        """Advance the staged prompts by at most ``chunk_size`` tokens in total.

        The budget is spent in admission order, so the number of extra forwards in a step
        is bounded no matter how many prompts are waiting. Returns the first response
        token and its logprob (when sampling) of every prompt that completed in this step.
        """
        budget = chunk_size
        completed = {}
        completed_logprobs = {}
        for request_id in list(prefill_chunks):
            if budget <= 0:
                break
            chunk = prefill_chunks[request_id]
            width = min(chunk.remaining, budget)
            first_token, first_logprob = self._advance_prefill_chunk(module,
                                                                     chunk,
                                                                     width,
                                                                     device,
                                                                     profile,
                                                                     sampling=sampling)
            budget -= width
            chunk_stats["chunks"] += 1
            chunk_stats["peak_staging_bytes"] = max(chunk_stats["peak_staging_bytes"], chunk.staging_bytes())
            if first_token is None:
                continue
            prompt_end = self._publish_prefill_chunk(chunk, cache, attention_mask, write_positions)
            if request_write_positions is not None:
                request_write_positions[request_id] = prompt_end
            completed[request_id] = first_token
            if first_logprob is not None:
                completed_logprobs[request_id] = first_logprob
            del prefill_chunks[request_id]
        return completed, completed_logprobs

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
                            request_write_positions=None,
                            sampling=None):
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
        completed_logprobs = {}
        for row_index, (request_id, chunk) in enumerate(wave):
            chunk.start = end
            if not completion:
                continue
            # The chunk that completes the prompt also predicts the first response token.
            first_token, first_logprob = self._draw_token(logits[row_index:row_index + 1, -1, :], sampling)
            self._publish_staged_prompt(chunk, state, cache, attention_mask, write_positions)
            if request_write_positions is not None:
                request_write_positions[request_id] = prompt_len
            completed[request_id] = first_token
            if first_logprob is not None:
                completed_logprobs[request_id] = first_logprob
            del prefill_chunks[request_id]
        chunk_stats["chunks"] += len(wave)
        chunk_stats["graph_steps"] = chunk_stats.get("graph_steps", 0) + 1
        chunk_stats["padded_rows"] = chunk_stats.get("padded_rows", 0) + width - len(wave)
        widths = chunk_stats.setdefault("bucket_widths", {})
        widths[width] = widths.get(width, 0) + 1
        return completed, completed_logprobs

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
        if self.use_paged_cache:
            for layer_idx in range(self._cache_layer_count(state.staging_cache)):
                staged_keys, staged_values = self._cache_layer_tensors(state.staging_cache, layer_idx)
                cache.copy_from_dense(layer_idx, staged_keys[chunk.row], staged_values[chunk.row], chunk.slot)
            attention_mask[chunk.slot].zero_()
            attention_mask[chunk.slot, :prompt_len] = 1
        else:
            for layer_idx in range(self._cache_layer_count(state.staging_cache)):
                staged_keys, staged_values = self._cache_layer_tensors(state.staging_cache, layer_idx)
                target_layer = cache.layers[layer_idx]
                target_layer.keys[chunk.slot, :, :prompt_len].copy_(staged_keys[chunk.row, :, :prompt_len])
                target_layer.values[chunk.slot, :, :prompt_len].copy_(staged_values[chunk.row, :, :prompt_len])
            attention_mask[chunk.slot, :prompt_len].copy_(chunk.prompt_attention[0])
        write_positions[chunk.slot] = prompt_len

    @staticmethod
    def _cache_bytes(cache):
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

    def _continuous_prefill_cache(self,
                                  module,
                                  static_cache_type,
                                  prompt_ids,
                                  prompt_attention,
                                  model_dtype,
                                  device,
                                  sampling=None):
        """Prefill a prompt batch; return ``(first tokens, their logprobs, cache)``.

        Greedy rollouts argmax the prompt's last logits, sampling rollouts draw them, so the
        first response token comes from the same distribution as the rest of the response.
        """
        position_ids = self._prefill_position_ids(prompt_attention)
        # Models that do not declare cache-class support get their prefill from a one-token
        # generate(); that shortcut cannot expose the prefill distribution, so sampling
        # rollouts take the cache path (transformers versions that omit the attribute are the
        # ones that support cache classes; older ones fail here instead of silently argmaxing).
        declares_legacy_cache = getattr(module, "_supports_cache_class", None) is None
        if declares_legacy_cache and (sampling is None or sampling.temperature <= 0.0):
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
            return prefill_output.sequences[:, -1:], None, prefill_output.past_key_values

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
        tokens, logprobs = self._draw_token(prefill_output.logits[:, -1, :], sampling)
        return tokens, logprobs, prefill_cache

    @staticmethod
    def _draw_token(logits, sampling=None):
        """Pick one token per row of ``logits`` [B, V]: greedy argmax, or a device draw.

        Returns ``(tokens [B, 1], logprobs [B] or None)``, so callers that sample can record
        the log-probability of the token they drew.
        """
        if sampling is None or sampling.temperature <= 0.0:
            return logits.argmax(dim=-1, keepdim=True), None
        tokens, logprobs = sample(logits, temperature=sampling.temperature, top_k=sampling.top_k, top_p=sampling.top_p)
        return tokens.unsqueeze(-1), logprobs

    @staticmethod
    def _repetition_penalized_logits(logits, request_ids, request_by_id, responses, module):
        """Copy of ``logits`` with HuggingFace's repetition penalty applied from each row's history."""
        repetition_penalty = getattr(getattr(module, "generation_config", None), "repetition_penalty", 1.0)
        if repetition_penalty is None:
            repetition_penalty = 1.0
        if repetition_penalty == 1.0:
            return logits

        penalized = logits.clone()
        for row, request_id in enumerate(request_ids):
            input_ids = torch.cat((request_by_id[request_id].prompt_ids, *responses[request_id]), dim=1)
            row_logits = penalized[row:row + 1]
            scores = row_logits.gather(1, input_ids)
            scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
            row_logits.scatter_(1, input_ids, scores)
        return penalized

    @staticmethod
    def _continuous_next_tokens(logits, request_ids, request_by_id, responses, module, sampling=None):
        """One token per active row: a device draw when sampling, otherwise the greedy argmax.

        Returns ``(tokens [R, 1], logprobs [R] or None)``.
        """
        logits = HybridEngineRollout._repetition_penalized_logits(logits, request_ids, request_by_id, responses,
                                                                  module)
        if sampling is None or sampling.temperature <= 0.0:
            return logits.argmax(dim=-1, keepdim=True), None
        tokens, logprobs = sample(logits, temperature=sampling.temperature, top_k=sampling.top_k, top_p=sampling.top_p)
        return tokens.unsqueeze(-1), logprobs

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

    def _build_continuous_batch(self, request, responses, response_logprobs=None):
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
        logprob_rows = []
        if response_logprobs is not None:
            for index, response in enumerate(response_ids):
                row = torch.cat(response_logprobs[index], dim=0)
                padding = max_response_len - row.shape[0]
                if padding:
                    # Padding positions are masked in attention_mask; report zero there.
                    row = torch.cat((row, torch.zeros(padding, dtype=row.dtype, device=row.device)), dim=0)
                logprob_rows.append(row.unsqueeze(0))
        return RolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_start_idx=torch.full((request.prompt_ids.shape[0], ),
                                          response_start,
                                          dtype=torch.long,
                                          device=input_ids.device),
            response_logprobs=torch.cat(logprob_rows, dim=0) if logprob_rows else None,
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

    def _generate_graph(self, prompt_ids, prompt_attn, max_new_tokens, pad_token_id, module, device, sampling):
        """Decode with DeepSpeedStaticCache + graph capture.

        Greedy rollouts take the argmax of the replayed logits; sampling rollouts draw them on
        the device (``runtime/rollout/sampling.py``). The captured forward is the same for both,
        so only the token selection differs, and sampling no longer gives up the graph.
        """
        from transformers import StaticCache
        from deepspeed.utils.static_cache import DeepSpeedStaticCache

        if getattr(module.config, "_attn_implementation", None) == "flash_attention_2":
            raise ValueError("graph capture does not support flash_attention_2")

        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        max_len = prompt_len + max_new_tokens
        eos_token_id = self.tokenizer.eos_token_id
        model_dtype = next(module.parameters()).dtype
        is_greedy = sampling.temperature <= 0.0

        # --- Prefill with HF StaticCache (correct attention semantics) ---
        prefill_attn = torch.ones(batch_size, prompt_len, dtype=torch.long, device=device)
        prefill_attn[:, :prompt_len] = prompt_attn
        if self.enable_chunked_prefill and is_greedy and prompt_len > self.prefill_chunk_size:
            next_token, prefill_cache = self._chunked_prefill_cache(module, StaticCache, prompt_ids, prefill_attn,
                                                                    model_dtype, device, self.prefill_chunk_size)
            next_logprob = None
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
            next_token, next_logprob = self._draw_token(prefill_out.logits[:, -1, :], sampling)

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
        response_logprobs = None
        if not is_greedy:
            response_logprobs = torch.zeros(batch_size, max_new_tokens, dtype=torch.float32, device=device)
            response_logprobs[:, 0] = next_logprob

        # --- Static buffers for graph capture ---
        static_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        static_attn = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        static_attn[:, :prompt_len] = prompt_attn
        static_attn[:, prompt_len] = 1  # first decode position
        # Transformers builds its SDPA mask from a 2-D attention_mask with host-side checks
        # (``padding_mask.all()`` in ``_ignore_causal_mask_sdpa``). Those synchronize the stream and
        # invalidate CUDA graph capture, so the captured forward receives a pre-built 4-D additive
        # mask instead, as the continuous graph path does; each decode step unmasks one more column.
        static_causal_attn = torch.zeros((batch_size, 1, 1, max_len), dtype=model_dtype, device=device)
        self._update_static_causal_attention(static_attn, static_causal_attn)
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
                        attention_mask=static_causal_attn,
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
                    attention_mask=static_causal_attn,
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
            self._update_static_causal_attention(static_attn, static_causal_attn)

            # Replay
            get_accelerator().replay_graph(graph)
            next_token, next_logprob = self._draw_token(static_logits[:, -1, :], sampling)
            output_ids.append(next_token)
            if response_logprobs is not None:
                response_logprobs[:, step + 1] = next_logprob
            eos_mask |= (next_token.squeeze(1) == eos_token_id)

        return torch.cat(output_ids, dim=1), response_logprobs

    @torch.no_grad()
    def _generate_sampling(self, prompt_ids, prompt_attn, max_new_tokens, module, device, sampling):
        """Sampling decode on a static cache, with the draw made on the device.

        Sampling used to be delegated to ``module.generate``, which runs HuggingFace's logits
        warpers and ``multinomial`` on the host and reallocates a dynamic KV cache on every
        step. Here the prompt is prefilled once into a static cache and each step is one
        forward plus one device-side draw (``deepspeed.runtime.rollout.sampling``), which also
        reports the log-probability of the token it drew.
        """
        from transformers import StaticCache

        batch_size, prompt_len = prompt_ids.shape
        max_len = prompt_len + max_new_tokens
        model_dtype = next(module.parameters()).dtype

        # Prompts arrive left-padded. The cache column and the causal position stay column
        # indices (the mask hides the pad columns), while the rotary position continues each
        # row's *real* prompt length so padding does not shift the embedding.
        first_position = prompt_attn.sum(dim=-1, keepdim=True)
        position_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        cache_position = torch.zeros(1, dtype=torch.long, device=device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        attention_mask[:, :prompt_len] = prompt_attn

        cache = self._create_static_cache(StaticCache, module.config, batch_size, max_len, device, model_dtype)
        logits = module(
            prompt_ids,
            attention_mask=attention_mask[:, :prompt_len],
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(prompt_len, device=device),
            position_ids=self._prefill_position_ids(prompt_attn),
            **self._last_logits_kwargs(module),
        ).logits[:, -1, :]

        next_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        generated = torch.zeros(batch_size, max_new_tokens, dtype=torch.long, device=device)
        logprobs = torch.zeros(batch_size, max_new_tokens, dtype=torch.float32, device=device)
        for step in range(max_new_tokens):
            token, logprob = sample(logits,
                                    temperature=sampling.temperature,
                                    top_k=sampling.top_k,
                                    top_p=sampling.top_p)
            generated[:, step] = token
            logprobs[:, step] = logprob
            if step + 1 == max_new_tokens:
                break
            next_token.copy_(token.unsqueeze(-1))
            column = prompt_len + step
            attention_mask[:, column] = 1
            cache_position.fill_(column)
            position_ids.copy_(first_position).add_(step)
            logits = module(
                next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_ids=position_ids,
            ).logits[:, -1, :]

        return torch.cat([prompt_ids, generated], dim=1), logprobs

    def sync_weights(self, step: int) -> None:  # noqa: ARG002
        """No-op: hybrid engine reads model weights live."""
        return None

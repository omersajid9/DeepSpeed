# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import pytest
import torch

from types import SimpleNamespace

from deepspeed.inference.v2.kernels.ragged_ops.blocked_flash.blocked_flash import get_kv_block_size
from deepspeed.runtime.rollout.paged_cache import PagedAttentionCache, HybridPagedAttentionCache


def test_paged_decode_matches_dense_oracle():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim, num_kv_heads, num_heads = 64, 2, 4
    block_size = get_kv_block_size(head_dim)
    config = SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=num_kv_heads,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        hidden_size=num_heads * head_dim,
    )
    cache = PagedAttentionCache(
        config,
        batch_size=2,
        max_cache_len=2 * block_size,
        device=device,
        dtype=dtype,
        block_size=block_size,
    )

    prompt_len = 5
    keys = torch.randn(2, num_kv_heads, prompt_len, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    valid = torch.tensor([True, False, True, True, True], device=device)
    cache.copy_from_dense(0, keys[0], values[0], 0)
    cache.copy_from_dense(0, keys[1], values[1], 1, valid)

    positions = torch.tensor([prompt_len, int(valid.sum())], device=device, dtype=torch.long)
    cache.set_write_position(positions)
    cache.set_active_rows([0, 1])
    new_keys = torch.randn(2, num_kv_heads, 1, head_dim, device=device, dtype=dtype)
    new_values = torch.randn_like(new_keys)
    cache.write(0, new_keys, new_values)
    cache.update_graph_atoms()

    query = torch.randn(2, num_heads, 1, head_dim, device=device, dtype=dtype)
    actual = cache.attention(0, query, 1.0 / (head_dim**0.5))
    for row, length in ((0, prompt_len + 1), (1, int(valid.sum()) + 1)):
        row_keys, row_values = cache.materialize(0, row, length)
        ratio = num_heads // num_kv_heads
        dense_keys = row_keys.unsqueeze(0).repeat_interleave(ratio, dim=1)
        dense_values = row_values.unsqueeze(0).repeat_interleave(ratio, dim=1)
        expected = torch.nn.functional.scaled_dot_product_attention(query[row:row + 1], dense_keys, dense_values)
        # attention() returns [rows, q_len, heads * head_dim] head-major; restore the
        # [rows, heads, q_len, head_dim] layout the dense oracle produces.
        actual_row = actual[row:row + 1].view(1, num_heads, 1, head_dim)
        torch.testing.assert_close(actual_row, expected, atol=0.25, rtol=0.25)


def test_paged_compact_releases_and_reorders_rows():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim, num_kv_heads = 64, 2
    block_size = get_kv_block_size(head_dim)
    config = SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=num_kv_heads,
        num_attention_heads=4,
        head_dim=head_dim,
        hidden_size=4 * head_dim,
    )
    cache = PagedAttentionCache(
        config,
        batch_size=2,
        max_cache_len=2 * block_size,
        device=device,
        dtype=dtype,
        block_size=block_size,
    )
    keys = torch.randn(2, num_kv_heads, 3, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    cache.copy_from_dense(0, keys[0], values[0], 0)
    cache.copy_from_dense(0, keys[1], values[1], 1)

    cache.compact(torch.tensor([1], device=device))

    assert cache.row_len.tolist()[:1] == [3]
    assert cache.blocks_in_use == 1
    row_keys, row_values = cache.materialize(0, 0, 3)
    torch.testing.assert_close(row_keys, keys[1])
    torch.testing.assert_close(row_values, values[1])


def test_graph_safe_paged_decode_matches_dense_oracle():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim, num_kv_heads, num_heads = 64, 2, 4
    block_size = get_kv_block_size(head_dim)
    config = SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=num_kv_heads,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        hidden_size=num_heads * head_dim,
    )
    cache = PagedAttentionCache(
        config,
        batch_size=2,
        max_cache_len=2 * block_size,
        device=device,
        dtype=dtype,
        block_size=block_size,
    )
    cache.enable_graph_safe()
    prompt_len = 5
    keys = torch.randn(2, num_kv_heads, prompt_len, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    cache.copy_from_dense(0, keys[0], values[0], 0)
    cache.copy_from_dense(0, keys[1], values[1], 1)
    positions = torch.tensor([prompt_len, prompt_len], device=device, dtype=torch.long)
    cache.set_write_position(positions)
    cache.set_graph_mode(2)
    new_keys = torch.randn(2, num_kv_heads, 1, head_dim, device=device, dtype=dtype)
    new_values = torch.randn_like(new_keys)
    cache.write(0, new_keys, new_values)
    cache.update_graph_atoms()

    query = torch.randn(2, num_heads, 1, head_dim, device=device, dtype=dtype)
    actual = cache.attention(0, query, 1.0 / (head_dim**0.5))
    for row in range(2):
        row_keys, row_values = cache.materialize(0, row, prompt_len + 1)
        ratio = num_heads // num_kv_heads
        expected = torch.nn.functional.scaled_dot_product_attention(
            query[row:row + 1],
            row_keys.unsqueeze(0).repeat_interleave(ratio, dim=1),
            row_values.unsqueeze(0).repeat_interleave(ratio, dim=1),
        )
        actual_row = actual[row:row + 1].view(1, num_heads, 1, head_dim)
        torch.testing.assert_close(actual_row, expected, atol=0.25, rtol=0.25)


def test_paged_prefill_attention_matches_dense_oracle():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(3)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim, num_kv_heads, num_heads = 64, 2, 4
    block_size = get_kv_block_size(head_dim)
    config = SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=num_kv_heads,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        hidden_size=num_heads * head_dim,
    )
    cache = PagedAttentionCache(
        config,
        batch_size=1,
        max_cache_len=512,
        device=device,
        dtype=dtype,
        block_size=block_size,
    )
    prompt_len = 260
    keys = torch.randn(1, num_kv_heads, prompt_len, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    cache.set_write_position(torch.tensor([0], device=device, dtype=torch.long))
    cache.set_active_rows([0])
    cache.write(0, keys, values)
    query = torch.randn(1, num_heads, prompt_len, head_dim, device=device, dtype=dtype)
    actual = cache.attention(0, query, 1.0 / (head_dim**0.5))
    ratio = num_heads // num_kv_heads
    dense_keys = keys[0].unsqueeze(0).repeat_interleave(ratio, dim=1)
    dense_values = values[0].unsqueeze(0).repeat_interleave(ratio, dim=1)
    expected = torch.nn.functional.scaled_dot_product_attention(query, dense_keys, dense_values, is_causal=True)
    actual = actual.view(1, prompt_len, num_heads, head_dim).permute(0, 2, 1, 3)
    torch.testing.assert_close(actual, expected, atol=0.25, rtol=0.25)


def test_flashinfer_decode_matches_dense_oracle():
    pytest.importorskip("flashinfer")
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(5)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    # Qwen3-4B head config: GQA ratio 4, head_dim 128, which routes the
    # FlashInfer decode to the fa2 tensor-core split-KV kernel.
    head_dim, num_kv_heads, num_heads = 128, 8, 32
    block_size = get_kv_block_size(head_dim)
    config = SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=num_kv_heads,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        hidden_size=num_heads * head_dim,
    )
    cache = PagedAttentionCache(config,
                                batch_size=2,
                                max_cache_len=8 * block_size,
                                device=device,
                                dtype=dtype,
                                block_size=block_size,
                                use_flashinfer_decode=True)

    prompt_len = 5 * block_size + 3
    keys = torch.randn(2, num_kv_heads, prompt_len, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    cache.copy_from_dense(0, keys[0], values[0], 0)
    cache.copy_from_dense(0, keys[1], values[1], 1)

    positions = torch.tensor([prompt_len, prompt_len], device=device, dtype=torch.long)
    cache.set_write_position(positions)
    cache.set_active_rows([0, 1])
    new_keys = torch.randn(2, num_kv_heads, 1, head_dim, device=device, dtype=dtype)
    new_values = torch.randn_like(new_keys)
    cache.write(0, new_keys, new_values)

    query = torch.randn(2, num_heads, 1, head_dim, device=device, dtype=dtype)
    actual = cache.attention(0, query, 1.0 / (head_dim**0.5))
    for row in range(2):
        row_keys, row_values = cache.materialize(0, row, prompt_len + 1)
        ratio = num_heads // num_kv_heads
        expected = torch.nn.functional.scaled_dot_product_attention(
            query[row:row + 1],
            row_keys.unsqueeze(0).repeat_interleave(ratio, dim=1),
            row_values.unsqueeze(0).repeat_interleave(ratio, dim=1),
        )
        actual_row = actual[row:row + 1].view(1, num_heads, 1, head_dim)
        torch.testing.assert_close(actual_row, expected, atol=0.25, rtol=0.25)


def test_prefix_reuse_shares_blocks_and_releases_after_last_row():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(4)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim, num_kv_heads = 64, 2
    block_size = get_kv_block_size(head_dim)
    config = SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=num_kv_heads,
        num_attention_heads=4,
        head_dim=head_dim,
        hidden_size=4 * head_dim,
    )
    cache = PagedAttentionCache(
        config,
        batch_size=2,
        max_cache_len=256,
        device=device,
        dtype=dtype,
        block_size=block_size,
    )
    keys = torch.randn(1, num_kv_heads, 3, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    cache.copy_from_dense(0, keys[0], values[0], 0)
    prefix_hash = b"same-prefix"
    cache.register_prefix([prefix_hash], 0, 1)

    assert cache.prepare_prefix([prefix_hash], 1) == 1
    assert cache.block_ids[1, 0].item() == cache.block_ids[0, 0].item()
    row_keys, row_values = cache.materialize(0, 1, 3)
    torch.testing.assert_close(row_keys, keys[0])
    torch.testing.assert_close(row_values, values[0])

    cache.clear_row(0)
    assert cache.blocks_in_use == 1
    row_keys, _ = cache.materialize(0, 1, 3)
    torch.testing.assert_close(row_keys, keys[0])
    cache.clear_row(1)
    assert cache.blocks_in_use == 0


def _hybrid_config():
    # Qwen3.5-style interleaving: GDN layers at even indices, full attention at odd.
    num_layers = 4
    layer_types = ["linear_attention" if i % 2 == 0 else "full_attention" for i in range(num_layers)]
    return SimpleNamespace(
        num_hidden_layers=num_layers,
        layer_types=layer_types,
        num_key_value_heads=2,
        num_attention_heads=4,
        head_dim=64,
        hidden_size=4 * 64,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
    )


def _hybrid_cache(device, dtype, batch_size=3, max_cache_len=None):
    config = _hybrid_config()
    block_size = get_kv_block_size(config.head_dim)
    if max_cache_len is None:
        max_cache_len = 2 * block_size
    return HybridPagedAttentionCache(
        config,
        batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
        block_size=block_size,
    ), block_size


# Conv channels: 2 * key_dim + value_dim = 2 * (1 * 16) + (2 * 16).
HYBRID_CONV_DIM = 64


def test_hybrid_paged_kv_pool_is_sized_for_full_attention_layers():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(5)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cache, block_size = _hybrid_cache(device, dtype)
    assert cache.num_layers == 4
    assert cache.num_kv_layers == 2
    # layers spans every HF layer index: GDN objects on the linear-attention
    # layers, KV layer views on the full-attention ones.
    layers = cache.layers
    assert len(layers) == 4
    assert hasattr(layers[0], "conv_states") and hasattr(layers[2], "recurrent_states")
    assert hasattr(layers[1], "keys") and hasattr(layers[3], "keys")
    # The KV pool holds two layers, so it matches a two-layer pure paged cache
    # and the GDN pools add on top of it.
    pure = PagedAttentionCache(
        SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=2,
            num_attention_heads=4,
            head_dim=64,
            hidden_size=4 * 64,
        ),
        batch_size=3,
        max_cache_len=2 * block_size,
        device=device,
        dtype=dtype,
        block_size=block_size,
    )
    assert cache.memory_bytes > pure.memory_bytes
    # KV operations reject the GDN layer indices.
    cache.set_write_position(torch.zeros(3, dtype=torch.long, device=device))
    cache.set_active_rows([0, 1, 2])
    keys = torch.randn(3, 2, 1, 64, device=device, dtype=dtype)
    with pytest.raises(ValueError):
        cache.write(0, keys, keys)
    with pytest.raises(ValueError):
        cache.materialize(0, 0)
    with pytest.raises(ValueError):
        cache.attention(0, torch.randn(3, 4, 1, 64, device=device, dtype=dtype), 1.0 / 8)


def test_hybrid_paged_gdn_prefill_protocol():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(6)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cache, _ = _hybrid_cache(device, dtype)
    gdn, kernel = 0, 4
    conv_dim = HYBRID_CONV_DIM
    cache.begin_prefill_row(1)
    cache.set_recording(True)
    cache.set_active_rows([1])
    # A fresh row reports no conv state and the first update stores the input
    # as-is, returning it unchanged for the chunked conv.
    assert cache.get_conv_state(gdn) is None
    x1 = torch.randn(1, conv_dim, 3, device=device, dtype=dtype)
    torch.testing.assert_close(cache.update_conv_state(x1, gdn), x1)
    # The prompt is shorter than the kernel, so the stored window is
    # left-padded with zeros (the causal conv's boundary context).
    expected_state1 = torch.nn.functional.pad(x1, (kernel - 3, 0))
    assert torch.equal(cache.get_conv_state(gdn), expected_state1)
    # A second chunk sees the stored state and returns state-plus-input.
    x2 = torch.randn(1, conv_dim, 2, device=device, dtype=dtype)
    full = torch.cat([expected_state1, x2], dim=-1)
    torch.testing.assert_close(cache.update_conv_state(x2, gdn), full)
    assert torch.equal(cache.get_conv_state(gdn), full[..., -kernel:])
    # The recurrent state lands on the row and the returned view is the state.
    state1 = torch.randn(1, 2, 16, 16, device=device, dtype=dtype)
    view = cache.update_recurrent_state(state1, gdn)
    torch.testing.assert_close(view, state1)
    assert torch.equal(cache.get_recurrent_state(gdn), state1)
    assert cache.is_recording_past(gdn)
    assert cache.has_previous_state(gdn)
    # A different fresh row starts from an empty state, not row 1's.
    cache.begin_prefill_row(0)
    cache.set_active_rows([0])
    assert cache.get_conv_state(gdn) is None
    assert torch.equal(cache.get_recurrent_state(gdn), torch.zeros(1, 2, 16, 16, device=device, dtype=dtype))


def test_hybrid_paged_gdn_decode_updates_pool_rows():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cache, _ = _hybrid_cache(device, dtype)
    gdn = 0
    conv_dim = HYBRID_CONV_DIM
    for row in (0, 1):
        cache.begin_prefill_row(row)
        cache.set_recording(True)
        cache.set_active_rows([row])
        cache.update_conv_state(torch.full((1, conv_dim, 6), float(row + 1), device=device, dtype=dtype), gdn)
        cache.update_recurrent_state(torch.full((1, 2, 16, 16), float(row + 1), device=device, dtype=dtype), gdn)
    cache.set_recording(False)
    cache.set_active_width(2)
    layer = cache.layers[gdn]
    assert layer.record_past is False
    # The decode kernels update conv_states[0] in place, so the view must be
    # the pool row range itself, not a copy.
    conv_view = layer.conv_states[0]
    conv_view[:, :, 1:] = conv_view[:, :, :-1]
    conv_view[:, :, 0] = 9
    state_before = cache.get_conv_state(gdn).clone()
    assert torch.equal(conv_view, state_before)
    # A chunked decode forward extends by the stored state and re-stores it.
    x = torch.zeros(2, conv_dim, 1, device=device, dtype=dtype)
    extended = cache.update_conv_state(x, gdn)
    assert extended.shape == (2, conv_dim, 5)
    torch.testing.assert_close(extended, torch.cat([state_before, x], dim=-1))
    torch.testing.assert_close(cache.get_conv_state(gdn), torch.cat([state_before, x], dim=-1)[..., -4:])
    # The recurrent update writes the whole active prefix.
    new_state = torch.randn(2, 2, 16, 16, device=device, dtype=dtype)
    cache.update_recurrent_state(new_state, gdn)
    assert torch.equal(cache.get_recurrent_state(gdn), new_state)
    cache.layers[gdn].recurrent_states[0][0, 0, 0, 0] = -7
    assert cache.get_recurrent_state(gdn)[0, 0, 0, 0].item() == -7


def test_hybrid_paged_compact_moves_gdn_pool_rows():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(8)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cache, _ = _hybrid_cache(device, dtype)
    gdn, kv = 0, 1
    conv_dim = HYBRID_CONV_DIM
    for row in range(3):
        cache.begin_prefill_row(row)
        cache.set_recording(True)
        cache.set_active_rows([row])
        cache.update_conv_state(torch.full((1, conv_dim, 4), float(row + 1), device=device, dtype=dtype), gdn)
        cache.update_recurrent_state(torch.full((1, 2, 16, 16), float(row + 1), device=device, dtype=dtype), gdn)
    cache.set_recording(False)
    keys = torch.randn(3, 2, 1, 64, device=device, dtype=dtype)
    cache.set_write_position(torch.zeros(3, dtype=torch.long, device=device))
    cache.set_active_rows([0, 1, 2])
    cache.write(kv, keys, keys)
    cache.compact(torch.tensor([0, 2], device=device))
    # KV row 1 now holds old row 2's blocks...
    row_keys, _ = cache.materialize(kv, 1, 1)
    torch.testing.assert_close(row_keys, keys[2])
    # ...and its GDN pool rows moved with it while the retired row cleared.
    cache.set_active_width(2)
    conv = cache.get_conv_state(gdn)
    assert conv.shape == (2, conv_dim, 4)
    torch.testing.assert_close(conv[0], torch.full((conv_dim, 4), 1.0, device=device, dtype=dtype))
    torch.testing.assert_close(conv[1], torch.full((conv_dim, 4), 3.0, device=device, dtype=dtype))
    cache.set_recording(True)
    cache.set_active_rows([1])
    torch.testing.assert_close(
        cache.get_conv_state(gdn)[0], torch.full((conv_dim, 4), 3.0, device=device, dtype=dtype))


def test_hybrid_paged_invalid_combinations_fail_closed():
    if not torch.cuda.is_available():  #ignore-cuda
        pytest.skip("paged cache needs a CUDA device")
    torch.manual_seed(9)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cache, _ = _hybrid_cache(device, dtype)
    # Blocks carry no GDN state, and trimmed tokens are lost from the state.
    with pytest.raises(ValueError):
        cache.prepare_prefix([], 0)
    with pytest.raises(ValueError):
        cache.register_prefix([], 0, 0)
    with pytest.raises(ValueError):
        cache.trim_left(1)
    # Decode requires the active rows to be the contiguous prefix...
    cache.set_active_rows([0, 1, 2])
    with pytest.raises(ValueError):
        cache.set_active_rows([1, 2])
    # ...while paged prefill targets exactly one row.
    cache.set_recording(True)
    with pytest.raises(ValueError):
        cache.set_active_rows([0, 1])
    # KV layers hold no GDN state.
    with pytest.raises(ValueError):
        cache.get_conv_state(1)


def test_partial_rotary_matches_transformers_oracle():
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    except ImportError:
        pytest.skip("transformers qwen3_5 modeling not available")
    from deepspeed.runtime.rollout.paged_cache import _apply_partial_rotary
    torch.manual_seed(10)
    batch, heads, seq_len = 2, 4, 5
    # Partial width: the pass-through dims must survive unrotated.
    head_dim, rotary_dim = 256, 64
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    cos = torch.randn(batch, seq_len, rotary_dim)
    sin = torch.randn(batch, seq_len, rotary_dim)
    expected_q, expected_k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    actual_q, actual_k = _apply_partial_rotary(q, k, cos, sin)
    torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(actual_k, expected_k, atol=0, rtol=0)
    # Full-width table takes the no-concat fast path.
    q2 = torch.randn(batch, heads, seq_len, 128)
    k2 = torch.randn(batch, heads, seq_len, 128)
    cos2 = torch.randn(batch, seq_len, 128)
    sin2 = torch.randn(batch, seq_len, 128)
    expected_q, expected_k = apply_rotary_pos_emb(q2, k2, cos2, sin2, unsqueeze_dim=1)
    actual_q, actual_k = _apply_partial_rotary(q2, k2, cos2, sin2)
    torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(actual_k, expected_k, atol=0, rtol=0)

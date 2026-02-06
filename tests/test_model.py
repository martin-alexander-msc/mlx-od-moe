"""
Tests for the full transformer model with OD-MoE layers.

Uses small configs for fast testing.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx_od_moe.model import (
    KimiODMoEConfig,
    KimiODMoEModel,
    KVCache,
    Attention,
    TransformerBlock,
    _create_causal_mask,
    _sample_top_p,
    _expand_kv_heads,
)


def small_config():
    """Small config for fast testing."""
    return KimiODMoEConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts_per_tok=2,
        num_local_experts=8,
        max_position_embeddings=512,
    )


class TestKimiODMoEConfig:
    def test_default_config(self):
        config = KimiODMoEConfig()
        assert config.vocab_size == 102400
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 28
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 8
        assert config.head_dim == 128
        assert config.eos_token_id == 0
        assert config.shadow_lookahead == 2

    def test_custom_config(self):
        config = small_config()
        assert config.head_dim == 32  # 128 / 4
        assert config.vocab_size == 1000

    def test_custom_eos_and_lookahead(self):
        config = KimiODMoEConfig(eos_token_id=2, shadow_lookahead=1)
        assert config.eos_token_id == 2
        assert config.shadow_lookahead == 1


class TestKVCache:
    def test_empty_cache(self):
        cache = KVCache()
        assert cache.offset == 0
        assert cache.keys is None

    def test_first_update(self):
        cache = KVCache()
        keys = mx.random.normal((1, 4, 8, 32))
        values = mx.random.normal((1, 4, 8, 32))
        k, v = cache.update(keys, values)
        assert cache.offset == 8
        assert k.shape == (1, 4, 8, 32)

    def test_incremental_update(self):
        cache = KVCache()
        # First update: 8 tokens
        k1 = mx.random.normal((1, 4, 8, 32))
        v1 = mx.random.normal((1, 4, 8, 32))
        cache.update(k1, v1)
        assert cache.offset == 8

        # Second update: 1 token
        k2 = mx.random.normal((1, 4, 1, 32))
        v2 = mx.random.normal((1, 4, 1, 32))
        k, v = cache.update(k2, v2)
        assert cache.offset == 9
        assert k.shape == (1, 4, 9, 32)

    def test_multiple_updates(self):
        cache = KVCache()
        for i in range(5):
            k = mx.random.normal((1, 2, 1, 16))
            v = mx.random.normal((1, 2, 1, 16))
            cache.update(k, v)
        assert cache.offset == 5

    def test_preallocated_buffer_reuses_memory(self):
        """Cache should pre-allocate in chunks, not copy every step."""
        cache = KVCache(step=8)
        k1 = mx.random.normal((1, 2, 3, 16))
        v1 = mx.random.normal((1, 2, 3, 16))
        cache.update(k1, v1)

        # Internal buffer should be larger than used (pre-allocated)
        assert cache.keys.shape[2] >= 3
        buf_size_after_first = cache.keys.shape[2]

        # Single token updates should not change buffer size
        for _ in range(4):
            k = mx.random.normal((1, 2, 1, 16))
            v = mx.random.normal((1, 2, 1, 16))
            cache.update(k, v)

        assert cache.offset == 7
        assert cache.keys.shape[2] == buf_size_after_first

    def test_values_preserved_across_growth(self):
        """When buffer grows, existing values should be preserved."""
        cache = KVCache(step=4)

        # Fill initial buffer
        k1 = mx.ones((1, 1, 3, 4))
        v1 = mx.ones((1, 1, 3, 4)) * 2.0
        cache.update(k1, v1)
        mx.eval(cache.keys, cache.values)

        # Force growth by exceeding step size
        k2 = mx.ones((1, 1, 3, 4)) * 3.0
        v2 = mx.ones((1, 1, 3, 4)) * 4.0
        k_out, v_out = cache.update(k2, v2)
        mx.eval(k_out, v_out)

        # Old values should still be there
        assert k_out[0, 0, 0, 0].item() == 1.0
        assert v_out[0, 0, 0, 0].item() == 2.0
        # New values too
        assert k_out[0, 0, 3, 0].item() == 3.0
        assert v_out[0, 0, 3, 0].item() == 4.0


class TestCausalMask:
    def test_single_token_no_mask(self):
        mask = _create_causal_mask(1)
        assert mask is None

    def test_mask_shape(self):
        mask = _create_causal_mask(8)
        assert mask.shape == (8, 8)

    def test_mask_is_causal(self):
        mask = _create_causal_mask(4)
        mx.eval(mask)

        # Diagonal and below should be 0 (allowed)
        for i in range(4):
            for j in range(i + 1):
                assert mask[i, j].item() == 0.0

        # Above diagonal should be large negative (blocked)
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[i, j].item() < -1e8

    def test_mask_with_cache_offset(self):
        """Mask should account for cached tokens in multi-token decode."""
        # 3 new query tokens, 5 already cached -> kv_len=8
        mask = _create_causal_mask(query_len=3, kv_len=8)
        mx.eval(mask)
        assert mask.shape == (3, 8)

        # First query token (position 5) can attend to kv positions 0-5
        for j in range(6):
            assert mask[0, j].item() == 0.0
        for j in range(6, 8):
            assert mask[0, j].item() < -1e8

        # Last query token (position 7) can attend to everything
        for j in range(8):
            assert mask[2, j].item() == 0.0

    def test_mask_caching(self):
        """Same parameters should return cached mask."""
        m1 = _create_causal_mask(4, 4)
        m2 = _create_causal_mask(4, 4)
        # Should be the exact same object (cached)
        assert m1 is m2


class TestExpandKVHeads:
    def test_no_expansion(self):
        x = mx.random.normal((1, 4, 8, 32))
        result = _expand_kv_heads(x, 1)
        assert result is x  # Same object, no copy

    def test_expansion_shape(self):
        x = mx.random.normal((1, 2, 8, 32))
        result = _expand_kv_heads(x, 4)
        assert result.shape == (1, 8, 8, 32)

    def test_expansion_values_correct(self):
        """Each KV head should be repeated n_rep times."""
        x = mx.array([[[[1.0, 2.0]], [[3.0, 4.0]]]])  # (1, 2, 1, 2)
        result = _expand_kv_heads(x, 2)
        mx.eval(result)
        assert result.shape == (1, 4, 1, 2)
        # Heads 0,1 should be copies of kv head 0
        assert result[0, 0, 0, 0].item() == 1.0
        assert result[0, 1, 0, 0].item() == 1.0
        # Heads 2,3 should be copies of kv head 1
        assert result[0, 2, 0, 0].item() == 3.0
        assert result[0, 3, 0, 0].item() == 3.0


class TestAttention:
    def test_output_shape(self):
        config = small_config()
        attn = Attention(config)
        x = mx.random.normal((2, 8, 128))
        output = attn(x)
        assert output.shape == (2, 8, 128)

    def test_with_causal_mask(self):
        config = small_config()
        attn = Attention(config)
        x = mx.random.normal((1, 16, 128))
        mask = _create_causal_mask(16)
        output = attn(x, mask=mask)
        assert output.shape == (1, 16, 128)

    def test_with_kv_cache_prefill(self):
        config = small_config()
        attn = Attention(config)
        cache = KVCache()

        # Prefill: 8 tokens
        x = mx.random.normal((1, 8, 128))
        mask = _create_causal_mask(8)
        output = attn(x, mask=mask, cache=cache)
        assert output.shape == (1, 8, 128)
        assert cache.offset == 8

    def test_with_kv_cache_decode(self):
        config = small_config()
        attn = Attention(config)
        cache = KVCache()

        # Prefill
        x1 = mx.random.normal((1, 8, 128))
        mask = _create_causal_mask(8)
        attn(x1, mask=mask, cache=cache)
        assert cache.offset == 8

        # Decode step: single token
        x2 = mx.random.normal((1, 1, 128))
        output = attn(x2, cache=cache)
        assert output.shape == (1, 1, 128)
        assert cache.offset == 9

    def test_gqa_head_expansion(self):
        config = small_config()
        # 4 query heads, 2 KV heads -> groups = 2
        attn = Attention(config)
        assert attn.num_kv_groups == 2

        x = mx.random.normal((1, 4, 128))
        output = attn(x)
        assert output.shape == (1, 4, 128)


class TestTransformerBlock:
    def test_output_shape(self):
        config = small_config()
        block = TransformerBlock(config, layer_idx=0)
        x = mx.random.normal((1, 8, 128))
        output = block(x)
        assert output.shape == (1, 8, 128)

    def test_with_mask_and_cache(self):
        config = small_config()
        block = TransformerBlock(config, layer_idx=0)
        cache = KVCache()

        x = mx.random.normal((1, 8, 128))
        mask = _create_causal_mask(8)
        output = block(x, mask=mask, cache=cache)
        assert output.shape == (1, 8, 128)
        assert cache.offset == 8

    def test_residual_connection(self):
        """With no MoE, the MoE residual is skipped, but attention residual works."""
        config = small_config()
        block = TransformerBlock(config, layer_idx=0)
        assert block.moe is None

        x = mx.random.normal((1, 4, 128))
        output = block(x)
        mx.eval(output)

        # Output should differ from input (attention applied)
        assert not mx.allclose(output, x, atol=1e-6)

    def test_with_moe_layer(self):
        """Block should use MoE layer when set."""
        from mlx_od_moe.od_moe_layer import ODMoELayer

        config = small_config()
        block = TransformerBlock(config, layer_idx=0)
        block.moe = ODMoELayer(
            layer_idx=0,
            hidden_dim=128,
            ffn_dim=256,
            num_experts=8,
            top_k=2,
        )

        x = mx.random.normal((1, 4, 128))
        output = block(x)
        assert output.shape == (1, 4, 128)


class TestKimiODMoEModel:
    def test_forward_pass_shape(self):
        config = small_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        logits = model(input_ids)
        assert logits.shape == (1, 5, 1000)

    def test_forward_batch(self):
        config = small_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3], [4, 5, 6]])
        logits = model(input_ids)
        assert logits.shape == (2, 3, 1000)

    def test_forward_with_cache(self):
        config = small_config()
        model = KimiODMoEModel(config)
        cache = [KVCache() for _ in model.layers]

        # Prefill
        input_ids = mx.array([[1, 2, 3, 4]])
        logits = model(input_ids, cache=cache)
        assert logits.shape == (1, 4, 1000)
        assert cache[0].offset == 4

        # Decode step
        input_ids = mx.array([[5]])
        logits = model(input_ids, cache=cache)
        assert logits.shape == (1, 1, 1000)
        assert cache[0].offset == 5

    def test_logits_not_nan(self):
        config = small_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3]])
        logits = model(input_ids)
        mx.eval(logits)
        assert not mx.any(mx.isnan(logits))

    def test_layer_count(self):
        config = small_config()
        model = KimiODMoEModel(config)
        assert len(model.layers) == 2

    def test_generate_produces_tokens(self):
        config = small_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3]])

        tokens = list(model.generate(input_ids, max_new_tokens=5, temperature=0.8))
        assert len(tokens) > 0
        assert len(tokens) <= 5
        for t in tokens:
            assert isinstance(t, int)
            assert 0 <= t < config.vocab_size

    def test_generate_greedy(self):
        config = small_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3]])

        # Greedy should be deterministic
        tokens1 = list(model.generate(input_ids, max_new_tokens=3, temperature=0))
        tokens2 = list(model.generate(input_ids, max_new_tokens=3, temperature=0))
        assert tokens1 == tokens2

    def test_generate_stops_on_default_eos(self):
        """Generation should stop when default EOS token (0) is produced."""
        config = small_config()
        model = KimiODMoEModel(config)

        # Bias the LM head to produce token 0
        model.lm_head.weight = mx.zeros_like(model.lm_head.weight)
        input_ids = mx.array([[1, 2]])

        tokens = list(model.generate(input_ids, max_new_tokens=100, temperature=0))
        assert len(tokens) == 1
        assert tokens[0] == 0

    def test_generate_custom_eos(self):
        """Generation should respect custom eos_token_id parameter."""
        config = small_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2]])

        # First, generate without EOS to see what token the model produces
        tokens_no_eos = list(
            model.generate(input_ids, max_new_tokens=5, temperature=0, eos_token_id=-1)
        )
        first_token = tokens_no_eos[0]
        assert len(tokens_no_eos) == 5  # Should run to max_new_tokens

        # Now set EOS to that first token — should stop after 1 token
        tokens_with_eos = list(
            model.generate(
                input_ids, max_new_tokens=100, temperature=0, eos_token_id=first_token
            )
        )
        assert len(tokens_with_eos) == 1
        assert tokens_with_eos[0] == first_token

    def test_generate_log_interval(self):
        """log_interval=0 should produce no output."""
        config = small_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3]])
        # Should not raise with log_interval=0 (default) or any positive value
        list(model.generate(input_ids, max_new_tokens=3, temperature=0.5, log_interval=0))
        list(model.generate(input_ids, max_new_tokens=3, temperature=0.5, log_interval=1))


class TestSampleTopP:
    def test_basic_sampling(self):
        logits = mx.array([[0.0, 1.0, 2.0, 3.0]])
        token = _sample_top_p(logits, p=0.9)
        mx.eval(token)
        assert 0 <= token.item() < 4

    def test_p_1_uses_all_tokens(self):
        logits = mx.array([[1.0, 1.0, 1.0, 1.0]])
        token = _sample_top_p(logits, p=1.0)
        mx.eval(token)
        assert 0 <= token.item() < 4

    def test_low_p_concentrates(self):
        """Low top_p should concentrate on the highest probability token."""
        # Strong preference for last token
        logits = mx.array([[-100.0, -100.0, -100.0, 100.0]])
        samples = []
        for _ in range(10):
            token = _sample_top_p(logits, p=0.1)
            mx.eval(token)
            samples.append(token.item())
        # Should almost always be token 3
        assert all(s == 3 for s in samples)

    def test_batch_sampling(self):
        logits = mx.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
        tokens = _sample_top_p(logits, p=0.9)
        mx.eval(tokens)
        assert tokens.shape == (2,)

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

    def test_custom_config(self):
        config = small_config()
        assert config.head_dim == 32  # 128 / 4
        assert config.vocab_size == 1000


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

    def test_generate_stops_on_eos(self):
        """Generation should stop when EOS token (0) is produced."""
        config = small_config()
        model = KimiODMoEModel(config)

        # Bias the LM head to produce token 0
        model.lm_head.weight = mx.zeros_like(model.lm_head.weight)
        input_ids = mx.array([[1, 2]])

        tokens = list(model.generate(input_ids, max_new_tokens=100, temperature=0))
        # Should stop early (first token should be 0 with zero weights -> argmax = 0)
        assert len(tokens) == 1
        assert tokens[0] == 0


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

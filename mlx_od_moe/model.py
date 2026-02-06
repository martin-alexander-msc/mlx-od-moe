"""
Full Model - Transformer with OD-MoE layers

Implements the complete inference pipeline:
- GQA (Grouped Query Attention) with RoPE
- KV cache for efficient autoregressive generation
- OD-MoE FFN layers with on-demand expert loading
- Top-p (nucleus) sampling
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Generator
from pathlib import Path

from .expert_store import UnifiedMemoryExpertStore
from .shadow_model import ShadowRunner
from .od_moe_layer import ODMoELayer


class KimiODMoEConfig:
    """Configuration matching Kimi-K2.5 architecture."""

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=28,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_experts_per_tok=8,
        num_local_experts=384,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.head_dim = hidden_size // num_attention_heads


class KVCache:
    """Key-Value cache for autoregressive generation."""

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None

    @property
    def offset(self) -> int:
        return self.keys.shape[2] if self.keys is not None else 0

    def update(self, keys: mx.array, values: mx.array):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys, self.values


class Attention(nn.Module):
    """Grouped Query Attention with RoPE."""

    def __init__(self, config: KimiODMoEConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rope = nn.RoPE(self.head_dim, base=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape: (B, L, num_heads, head_dim) -> (B, num_heads, L, head_dim)
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE with cache offset
        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        # Update KV cache
        if cache is not None:
            keys, values = cache.update(keys, values)

        # GQA: expand KV heads to match query heads
        if self.num_kv_groups > 1:
            keys = mx.repeat(keys, self.num_kv_groups, axis=1)
            values = mx.repeat(values, self.num_kv_groups, axis=1)

        # Scaled dot-product attention
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        output = weights @ values

        # Reshape back: (B, num_heads, L, head_dim) -> (B, L, hidden_size)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: Attention + OD-MoE FFN."""

    def __init__(self, config: KimiODMoEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = Attention(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # MoE layer initialized later via setup_od_moe
        self.moe = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        # Pre-norm attention with residual
        r = self.attention(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r

        # Pre-norm MoE FFN with residual
        if self.moe is not None:
            r = self.moe(self.post_attention_layernorm(h))
            h = h + r

        return h


def _create_causal_mask(seq_len: int) -> Optional[mx.array]:
    """Create additive causal attention mask."""
    if seq_len <= 1:
        return None

    indices = mx.arange(seq_len)
    mask = (indices[:, None] < indices[None, :]).astype(mx.float32) * -1e9
    return mask


def _sample_top_p(logits: mx.array, p: float) -> mx.array:
    """
    Top-p (nucleus) sampling.

    Args:
        logits: Un-normalized logits (batch, vocab_size)
        p: Cumulative probability threshold

    Returns:
        Sampled token IDs (batch,)
    """
    if p >= 1.0:
        return mx.random.categorical(logits)

    probs = mx.softmax(logits, axis=-1)
    sorted_indices = mx.argsort(probs, axis=-1)[:, ::-1]
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Mask tokens beyond the cumulative threshold (keep at least one)
    mask = (cumulative_probs - sorted_probs) >= p
    sorted_probs = mx.where(mask, mx.zeros_like(sorted_probs), sorted_probs)

    # Re-normalize
    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)

    # Sample from filtered distribution
    sampled_idx = mx.random.categorical(mx.log(sorted_probs + 1e-10))

    # Map back to original vocabulary indices
    next_token = mx.take_along_axis(sorted_indices, sampled_idx[:, None], axis=-1)[:, 0]
    return next_token


class KimiODMoEModel(nn.Module):
    """
    Full Kimi-K2.5 model with OD-MoE layers.

    Memory breakdown at inference:
    - Base model (embeddings, norms, attention): ~35GB
    - Shadow predictor: ~0.5GB
    - Expert working set: 8 experts x 28 layers x ~50MB = ~11GB
    - KV cache 256K: ~7GB (GQA compressed)

    Total: ~54GB resident + 325GB memory-mapped on SSD
    """

    def __init__(self, config: KimiODMoEConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # OD-MoE components (initialized later)
        self.expert_store: Optional[UnifiedMemoryExpertStore] = None
        self.shadow_runner: Optional[ShadowRunner] = None

    def setup_od_moe(
        self,
        expert_dir: str,
        predictor_path: Optional[str] = None,
        cache_size_gb: int = 48,
    ):
        """Initialize OD-MoE after base model weights are loaded."""
        print("Setting up OD-MoE...")

        self.expert_store = UnifiedMemoryExpertStore(
            expert_dir,
            cache_size_gb=cache_size_gb,
            num_layers=self.config.num_hidden_layers,
            num_experts_per_layer=self.config.num_local_experts,
        )

        self.shadow_runner = ShadowRunner(predictor_path)

        for layer in self.layers:
            layer.moe = ODMoELayer(
                layer_idx=layer.layer_idx,
                hidden_dim=self.config.hidden_size,
                ffn_dim=self.config.intermediate_size,
                num_experts=self.config.num_local_experts,
                top_k=self.config.num_experts_per_tok,
                expert_store=self.expert_store,
                shadow_runner=self.shadow_runner,
            )

        print(f"OD-MoE setup complete: {len(self.layers)} layers")

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[KVCache]] = None,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            cache: Optional list of KV caches per layer

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        h = self.embed_tokens(input_ids)

        mask = _create_causal_mask(h.shape[1])

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            h = layer(h, mask=mask, cache=layer_cache)

            # Trigger shadow model predictions for prefetch
            if self.shadow_runner and i < self.config.num_hidden_layers - 4:
                self.shadow_runner.predict_async(h, i)

        h = self.norm(h)
        return self.lm_head(h)

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> Generator[int, None, None]:
        """
        Streaming autoregressive generation with KV cache.

        Args:
            input_ids: Prompt token IDs (1, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold

        Yields:
            Generated token IDs one at a time
        """
        cache = [KVCache() for _ in self.layers]

        # Prefill: process full prompt
        logits = self(input_ids, cache=cache)
        mx.eval(logits)

        for step in range(max_new_tokens):
            next_token_logits = logits[:, -1, :]

            if temperature == 0:
                next_token = mx.argmax(next_token_logits, axis=-1)
            else:
                next_token = _sample_top_p(next_token_logits / temperature, top_p)
            mx.eval(next_token)

            token_id = next_token.item()
            yield token_id

            if token_id == 0:
                break

            # Decode step: process just the new token with KV cache
            logits = self(next_token.reshape(1, 1), cache=cache)
            mx.eval(logits)

            if step % 50 == 0 and self.expert_store:
                stats = self.expert_store.get_stats()
                print(f"Step {step}: Cache hit rate {stats['hit_rate']:.2%}")

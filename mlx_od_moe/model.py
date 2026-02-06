"""
Full Model - Transformer with OD-MoE layers

Implements the complete inference pipeline:
- GQA (Grouped Query Attention) with RoPE
- Pre-allocated KV cache for efficient autoregressive generation
- OD-MoE FFN layers with on-demand expert loading
- Top-p (nucleus) sampling

NOTE: This uses GQA attention. Kimi-K2.5 uses MLA (Multi-head Latent Attention)
which compresses KV projections into a low-rank latent space for 8x KV cache
reduction. GQA is used here as a compatible foundation that also supports
Qwen2-MoE and Mixtral architectures. MLA can be added as an alternative
Attention class when targeting Kimi-K2.5 specifically.
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
        eos_token_id=0,
        shadow_lookahead=2,
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
        self.eos_token_id = eos_token_id
        self.shadow_lookahead = shadow_lookahead


class KVCache:
    """
    Pre-allocated KV cache for autoregressive generation.

    Allocates buffers in chunks (default 256 tokens) and fills them via
    slice assignment, avoiding O(T) copies on every decode step.
    """

    def __init__(self, step: int = 256):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self._offset = 0
        self._step = step

    @property
    def offset(self) -> int:
        return self._offset

    def update(self, keys: mx.array, values: mx.array):
        new_tokens = keys.shape[2]

        if self.keys is None or (self._offset + new_tokens) > self.keys.shape[2]:
            # Need to allocate or grow the buffer
            B, H, _, D = keys.shape
            n_steps = (self._step + new_tokens - 1) // self._step
            alloc_len = n_steps * self._step
            new_k = mx.zeros((B, H, self._offset + alloc_len, D), dtype=keys.dtype)
            new_v = mx.zeros((B, H, self._offset + alloc_len, D), dtype=values.dtype)

            if self.keys is not None:
                # Copy existing cached data into new buffer
                new_k[:, :, : self._offset, :] = self.keys[:, :, : self._offset, :]
                new_v[:, :, : self._offset, :] = self.values[:, :, : self._offset, :]

            self.keys = new_k
            self.values = new_v

        # Write new keys/values into the pre-allocated slot
        self.keys[:, :, self._offset : self._offset + new_tokens, :] = keys
        self.values[:, :, self._offset : self._offset + new_tokens, :] = values
        self._offset += new_tokens

        return self.keys[:, :, : self._offset, :], self.values[:, :, : self._offset, :]


def _expand_kv_heads(x: mx.array, n_rep: int) -> mx.array:
    """Expand KV heads to match query heads via zero-copy broadcast."""
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return mx.broadcast_to(
        x[:, :, None, :, :], (B, H, n_rep, T, D)
    ).reshape(B, H * n_rep, T, D)


# Cache for commonly used causal masks to avoid recomputation
_mask_cache: dict = {}


def _create_causal_mask(
    query_len: int, kv_len: Optional[int] = None
) -> Optional[mx.array]:
    """
    Create additive causal attention mask.

    Args:
        query_len: Number of query tokens
        kv_len: Total KV length (query_len + cache offset). If None, equals query_len.

    Returns:
        Mask of shape (query_len, kv_len) or None if no masking needed.
    """
    if query_len <= 1:
        return None

    if kv_len is None:
        kv_len = query_len

    cache_key = (query_len, kv_len)
    if cache_key in _mask_cache:
        return _mask_cache[cache_key]

    # Row indices: positions in the query (offset by kv_len - query_len)
    q_positions = mx.arange(query_len) + (kv_len - query_len)
    k_positions = mx.arange(kv_len)
    mask = (q_positions[:, None] < k_positions[None, :]).astype(mx.float32) * -1e9

    # Cache masks for common sizes (limit cache to avoid unbounded growth)
    if len(_mask_cache) < 64:
        _mask_cache[cache_key] = mask

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

    # Sample: use -inf for zeroed-out tokens to avoid epsilon-induced distribution shift
    log_probs = mx.where(
        sorted_probs > 0,
        mx.log(sorted_probs),
        mx.full(sorted_probs.shape, -1e9),
    )
    sampled_idx = mx.random.categorical(log_probs)

    # Map back to original vocabulary indices
    next_token = mx.take_along_axis(sorted_indices, sampled_idx[:, None], axis=-1)[:, 0]
    return next_token


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

        # GQA: expand KV heads to match query heads (zero-copy broadcast)
        keys = _expand_kv_heads(keys, self.num_kv_groups)
        values = _expand_kv_heads(values, self.num_kv_groups)

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

        # Build causal mask accounting for cached tokens
        cache_offset = cache[0].offset if cache else 0
        kv_len = h.shape[1] + cache_offset
        mask = _create_causal_mask(h.shape[1], kv_len)

        lookahead = self.config.shadow_lookahead
        eval_interval = max(1, self.config.num_hidden_layers // 4)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            h = layer(h, mask=mask, cache=layer_cache)

            # Trigger shadow model predictions for prefetch (configurable lookahead)
            if self.shadow_runner and i < self.config.num_hidden_layers - lookahead:
                self.shadow_runner.predict_async(h, i)

            # Periodic evaluation to bound graph memory on large models
            if i > 0 and i % eval_interval == 0:
                mx.eval(h)

        h = self.norm(h)
        return self.lm_head(h)

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 0.6,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        log_interval: int = 0,
    ) -> Generator[int, None, None]:
        """
        Streaming autoregressive generation with KV cache.

        Args:
            input_ids: Prompt token IDs (1, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            eos_token_id: Stop token (defaults to config.eos_token_id)
            log_interval: Print cache stats every N steps (0 = disabled)

        Yields:
            Generated token IDs one at a time
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

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

            if token_id == eos_token_id:
                break

            # Decode step: process just the new token with KV cache
            logits = self(next_token.reshape(1, 1), cache=cache)
            mx.eval(logits)

            if log_interval > 0 and step % log_interval == 0 and self.expert_store:
                stats = self.expert_store.get_stats()
                print(f"Step {step}: Cache hit rate {stats['hit_rate']:.2%}")

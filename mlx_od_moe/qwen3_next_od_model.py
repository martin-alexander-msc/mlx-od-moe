"""
Qwen3Next OD-MoE runtime model.

Uses Qwen3Next hybrid attention stack (GatedDeltaNet + GatedAttention) and
keeps routed experts on-demand via ODMoELayer.
"""

from __future__ import annotations

from typing import Optional, List, Generator, Any

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import KVCache, MambaCache
from mlx_lm.models.qwen3_next import Qwen3NextAttention, Qwen3NextGatedDeltaNet, Qwen3NextMLP

from .expert_store import UnifiedMemoryExpertStore
from .gguf_expert_store import GGUFOnDemandExpertStore
from .shadow_model import ShadowRunner
from .od_moe_layer import ODMoELayer


class Qwen3NextODConfig:
    """Runtime config for Qwen3Next OD-MoE."""

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        num_hidden_layers: int = 48,
        intermediate_size: int = 5120,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        partial_rotary_factor: float = 0.25,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5000000.0,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        rope_scaling: Optional[dict] = None,
        full_attention_interval: int = 4,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        num_local_experts: int = 512,
        num_experts_per_tok: int = 10,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        norm_topk_prob: bool = True,
        eos_token_id: int = 0,
        shadow_lookahead: int = 2,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.rope_scaling = rope_scaling
        self.full_attention_interval = full_attention_interval
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.eos_token_id = eos_token_id
        self.shadow_lookahead = shadow_lookahead


class Qwen3NextODSparseMoeBlock(nn.Module):
    """Sparse MoE block using on-demand local experts + resident shared expert."""

    def __init__(
        self,
        config: Qwen3NextODConfig,
        layer_idx: int,
        expert_store: Optional[UnifiedMemoryExpertStore] = None,
        shadow_runner: Optional[ShadowRunner] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.local_moe = ODMoELayer(
            layer_idx=layer_idx,
            hidden_dim=config.hidden_size,
            ffn_dim=config.moe_intermediate_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            expert_store=expert_store,
            shadow_runner=shadow_runner,
        )
        self.shared_expert = Qwen3NextMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def set_runtime(
        self,
        expert_store: Optional[UnifiedMemoryExpertStore],
        shadow_runner: Optional[ShadowRunner],
    ):
        self.local_moe.expert_store = expert_store
        self.local_moe.shadow_runner = shadow_runner

    def __call__(self, x: mx.array) -> mx.array:
        local_y = self.local_moe(x)
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return local_y + shared_y


class Qwen3NextODDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3NextODConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear = (layer_idx + 1) % config.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = Qwen3NextGatedDeltaNet(config)
        else:
            self.self_attn = Qwen3NextAttention(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3NextODSparseMoeBlock(config, layer_idx)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask=mask, cache=cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3NextODMoEModel(nn.Module):
    """
    Qwen3Next hybrid runtime with OD expert serving.
    """

    def __init__(self, config: Qwen3NextODConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen3NextODDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.expert_store: Optional[UnifiedMemoryExpertStore] = None
        self.shadow_runner: Optional[ShadowRunner] = None
        self.ssm_idx = 0
        self.fa_idx = config.full_attention_interval - 1

    def setup_od_moe(
        self,
        expert_dir: Optional[str] = None,
        gguf_expert_path: Optional[str] = None,
        predictor_path: Optional[str] = None,
        cache_size_gb: int = 48,
        enable_prefetch: bool = False,
    ):
        print("Setting up OD-MoE...")
        if gguf_expert_path:
            self.expert_store = GGUFOnDemandExpertStore(
                gguf_expert_path,
                cache_size_gb=cache_size_gb,
                num_layers=self.config.num_hidden_layers,
                num_experts_per_layer=self.config.num_local_experts,
            )
        else:
            if not expert_dir:
                raise ValueError("expert_dir is required when gguf_expert_path is not set")
            self.expert_store = UnifiedMemoryExpertStore(
                expert_dir,
                cache_size_gb=cache_size_gb,
                num_layers=self.config.num_hidden_layers,
                num_experts_per_layer=self.config.num_local_experts,
            )

        if enable_prefetch:
            self.shadow_runner = ShadowRunner(
                predictor_path=predictor_path,
                hidden_dim=self.config.hidden_size,
                num_experts=self.config.num_local_experts,
                top_k=self.config.num_experts_per_tok,
            )
            print("Shadow prefetch enabled")
        else:
            self.shadow_runner = None
            print("Shadow prefetch disabled")
        for layer in self.layers:
            layer.mlp.set_runtime(self.expert_store, self.shadow_runner)
        print(f"OD-MoE setup complete: {len(self.layers)} layers")

    def make_cache(self):
        return [MambaCache() if layer.is_linear else KVCache() for layer in self.layers]

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(input_ids)
        if cache is None:
            cache = self.make_cache()

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        for layer, layer_cache in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=layer_cache)

        h = self.norm(h)
        return self.lm_head(h)

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 0.6,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> Generator[int, None, None]:
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        cache = self.make_cache()
        logits = self(input_ids, cache=cache)
        mx.eval(logits)

        for _ in range(max_new_tokens):
            next_token_logits = logits[:, -1, :]
            if temperature == 0:
                next_token = mx.argmax(next_token_logits, axis=-1)
            else:
                probs = mx.softmax(next_token_logits / temperature, axis=-1)
                sorted_indices = mx.argsort(probs, axis=-1)[:, ::-1]
                sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
                cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
                mask = (cumulative_probs - sorted_probs) >= top_p
                sorted_probs = mx.where(mask, mx.zeros_like(sorted_probs), sorted_probs)
                sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)
                log_probs = mx.where(
                    sorted_probs > 0,
                    mx.log(sorted_probs),
                    mx.full(sorted_probs.shape, -1e9),
                )
                sampled_idx = mx.random.categorical(log_probs)
                next_token = mx.take_along_axis(
                    sorted_indices,
                    sampled_idx[:, None],
                    axis=-1,
                )[:, 0]
            mx.eval(next_token)

            token_id = int(next_token.item())
            yield token_id
            if token_id == eos_token_id:
                break

            logits = self(next_token.reshape(1, 1), cache=cache)
            mx.eval(logits)

"""
Utilities for loading base model weights from GGUF-converted outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import re
import json
import math

from safetensors import safe_open


BASE_WEIGHT_FILES = (
    "embeddings.safetensors",
    "attention_layers.safetensors",
    "norms.safetensors",
    "lm_head.safetensors",
)


def _resolve_base_component_files(base_path: Path) -> list[Path]:
    if base_path.is_dir():
        component_files = [
            base_path / name for name in BASE_WEIGHT_FILES if (base_path / name).exists()
        ]
        if not component_files:
            raise ValueError(f"No base safetensors files found in directory: {base_path}")
        return component_files
    return [base_path]


def inspect_base_weight_shapes(base_weights: str) -> dict[str, tuple[int, ...]]:
    """Read tensor shapes from safetensors metadata without loading tensor payloads."""
    base_path = Path(base_weights)
    if not base_path.exists():
        raise FileNotFoundError(f"Base weights not found: {base_weights}")

    tensor_shapes: dict[str, tuple[int, ...]] = {}
    for component_path in _resolve_base_component_files(base_path):
        with safe_open(component_path, framework="numpy") as reader:
            for key in reader.keys():
                tensor_shapes[key] = tuple(int(dim) for dim in reader.get_slice(key).get_shape())
    return tensor_shapes


def infer_config_overrides_from_base_shapes(base_weights: str) -> dict[str, int]:
    """
    Infer basic runtime config fields from converted base weights.

    This is intentionally conservative and only sets values that can be
    inferred reliably from tensor shapes.
    """
    tensor_shapes = inspect_base_weight_shapes(base_weights)

    emb_shape = tensor_shapes.get("token_embd.weight") or tensor_shapes.get("embed_tokens.weight")
    if emb_shape is None or len(emb_shape) != 2:
        raise ValueError("Could not infer vocab/hidden size (missing token embeddings)")

    layer_indices = set()
    for key in tensor_shapes:
        m = re.match(r"blk\.(\d+)\.", key)
        if m:
            layer_indices.add(int(m.group(1)))
        m2 = re.match(r"layers\.(\d+)\.", key)
        if m2:
            layer_indices.add(int(m2.group(1)))

    if not layer_indices:
        raise ValueError("Could not infer number of layers from base weights")

    layer0_attn_q = tensor_shapes.get("blk.0.attn_q.weight") or tensor_shapes.get(
        "layers.0.attention.q_proj.weight"
    )
    layer0_attn_k = tensor_shapes.get("blk.0.attn_k.weight") or tensor_shapes.get(
        "layers.0.attention.k_proj.weight"
    )
    layer0_attn_v = tensor_shapes.get("blk.0.attn_v.weight") or tensor_shapes.get(
        "layers.0.attention.v_proj.weight"
    )
    if layer0_attn_v and len(layer0_attn_v) == 2:
        # In both (hidden, kv_hidden) and (kv_hidden, hidden), hidden is the larger axis.
        hidden_size = int(max(layer0_attn_v))
    elif layer0_attn_q and len(layer0_attn_q) == 2:
        hidden_size = int(max(layer0_attn_q))
    else:
        # Fallback for models without recognizable attention names.
        hidden_size = int(min(emb_shape))

    emb0, emb1 = int(emb_shape[0]), int(emb_shape[1])
    if emb0 == hidden_size and emb1 != hidden_size:
        vocab_size = emb1
    elif emb1 == hidden_size and emb0 != hidden_size:
        vocab_size = emb0
    else:
        # If ambiguous, assume the larger axis is vocab.
        vocab_size = max(emb0, emb1)

    def _infer_proj_out(shape: tuple[int, ...] | None) -> int | None:
        if not shape or len(shape) != 2:
            return None
        a, b = int(shape[0]), int(shape[1])
        if a == hidden_size and b != hidden_size:
            return b
        if b == hidden_size and a != hidden_size:
            return a
        return max(a, b)

    q_proj_out = _infer_proj_out(layer0_attn_q)
    kv_proj_out = _infer_proj_out(layer0_attn_k) or _infer_proj_out(layer0_attn_v)

    # Fallback inference for non-square projections when GGUF metadata is unavailable.
    inferred_num_heads = None
    inferred_num_kv_heads = None
    inferred_head_dim = None
    if q_proj_out and kv_proj_out:
        common = math.gcd(q_proj_out, kv_proj_out)
        for candidate_head_dim in (128, 96, 80, 64, 40, 32, 16, 8):
            if (
                candidate_head_dim <= common
                and q_proj_out % candidate_head_dim == 0
                and kv_proj_out % candidate_head_dim == 0
            ):
                inferred_head_dim = candidate_head_dim
                inferred_num_heads = q_proj_out // candidate_head_dim
                inferred_num_kv_heads = kv_proj_out // candidate_head_dim
                if inferred_num_heads % inferred_num_kv_heads == 0:
                    break

    inferred = {
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "num_hidden_layers": max(layer_indices) + 1,
    }
    if inferred_num_heads and inferred_num_kv_heads and inferred_head_dim:
        inferred["num_attention_heads"] = int(inferred_num_heads)
        inferred["num_key_value_heads"] = int(inferred_num_kv_heads)
        inferred["head_dim"] = int(inferred_head_dim)
    return inferred


def validate_expert_conversion(expert_dir: str) -> None:
    """
    Fail fast when the converted expert set is empty/invalid.
    """
    expert_path = Path(expert_dir)
    if not expert_path.exists():
        raise FileNotFoundError(f"Expert directory not found: {expert_dir}")

    registry_path = expert_path / "registry.json"
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as handle:
            registry = json.load(handle)
        if not registry:
            raise ValueError("Expert registry is empty")

        non_empty = sum(1 for item in registry.values() if item.get("tensors"))
        if non_empty == 0:
            raise ValueError(
                "All converted experts are empty. The GGUF model likely does not match "
                "the converter's MoE expert tensor naming, or wrong --num-layers/--num-experts "
                "were used during conversion."
            )
        return

    files = sorted(expert_path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No expert .safetensors files found in {expert_dir}")

    # Typical empty safetensors file generated by converter is 16 bytes.
    non_empty_files = [f for f in files if f.stat().st_size > 16]
    if not non_empty_files:
        raise ValueError(
            "All expert files are empty (16B). Conversion produced no expert tensors."
        )


def infer_num_local_experts(expert_dir: str) -> int | None:
    """Infer experts per layer from registry/file naming when possible."""
    expert_path = Path(expert_dir)
    registry_path = expert_path / "registry.json"
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as handle:
            registry = json.load(handle)
        expert_ids = [
            int(item.get("expert_id"))
            for item in registry.values()
            if item.get("tensors") and item.get("expert_id") is not None
        ]
        if expert_ids:
            return max(expert_ids) + 1
        return None

    expert_ids = []
    pattern = re.compile(r"layer_\d+_expert_(\d+)\.safetensors$")
    for path in expert_path.glob("layer_*_expert_*.safetensors"):
        if path.stat().st_size <= 16:
            continue
        m = pattern.match(path.name)
        if m:
            expert_ids.append(int(m.group(1)))
    if expert_ids:
        return max(expert_ids) + 1
    return None


def map_gguf_key_to_model_key(key: str) -> str:
    """Map GGUF tensor names to KimiODMoEModel parameter names."""
    if key == "token_embd.weight":
        return "embed_tokens.weight"
    if key == "output.weight":
        return "lm_head.weight"
    if key == "output_norm.weight":
        return "norm.weight"

    attn_norm = re.fullmatch(r"blk\.(\d+)\.attn_norm\.weight", key)
    if attn_norm:
        layer = attn_norm.group(1)
        return f"layers.{layer}.input_layernorm.weight"

    ffn_norm = re.fullmatch(r"blk\.(\d+)\.ffn_norm\.weight", key)
    if ffn_norm:
        layer = ffn_norm.group(1)
        return f"layers.{layer}.post_attention_layernorm.weight"

    gate = re.fullmatch(r"blk\.(\d+)\.ffn\.gate\.weight", key)
    if gate:
        layer = gate.group(1)
        return f"layers.{layer}.moe.gate.weight"

    attn_q = re.fullmatch(r"blk\.(\d+)\.attn_q\.weight", key)
    if attn_q:
        layer = attn_q.group(1)
        return f"layers.{layer}.attention.q_proj.weight"

    attn_k = re.fullmatch(r"blk\.(\d+)\.attn_k\.weight", key)
    if attn_k:
        layer = attn_k.group(1)
        return f"layers.{layer}.attention.k_proj.weight"

    attn_v = re.fullmatch(r"blk\.(\d+)\.attn_v\.weight", key)
    if attn_v:
        layer = attn_v.group(1)
        return f"layers.{layer}.attention.v_proj.weight"

    attn_o = re.fullmatch(r"blk\.(\d+)\.attn_output\.weight", key)
    if attn_o:
        layer = attn_o.group(1)
        return f"layers.{layer}.attention.o_proj.weight"

    return key


def _as_tuple(shape: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


def _transpose_if_needed(tensor: Any, expected_shape: tuple[int, ...], source_key: str, model_key: str) -> Any:
    tensor_shape = _as_tuple(tensor.shape)
    if tensor_shape == expected_shape:
        return tensor
    if len(tensor_shape) == 2 and tensor_shape[::-1] == expected_shape:
        return tensor.T
    hint = ""
    if source_key.startswith("blk.") and ".attn_" in source_key:
        hint = (
            " This source attention tensor layout is incompatible with the current "
            "OD-MoE GQA model implementation (likely MLA/variant attention)."
        )
    raise ValueError(
        f"Shape mismatch for {source_key} -> {model_key}: got {tensor_shape}, "
        f"expected {expected_shape}.{hint}"
    )


def _load_weight_dict(path: Path, load_fn: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    loaded = load_fn(str(path))
    if hasattr(loaded, "items"):
        return dict(loaded.items())
    raise TypeError(f"Loaded object from {path} is not dict-like")


def load_base_weight_items(
    base_weights: str,
    load_fn: Callable[[str], dict[str, Any]],
    expected_shapes: dict[str, tuple[int, ...]],
) -> tuple[list[tuple[str, Any]], dict[str, int]]:
    """
    Load base weights from a file or from a converted base_model directory.

    Returns:
        (weights, stats) where weights is suitable for model.load_weights.
    """
    base_path = Path(base_weights)
    if not base_path.exists():
        raise FileNotFoundError(f"Base weights not found: {base_weights}")

    source_dict: dict[str, Any] = {}
    component_files = _resolve_base_component_files(base_path)
    for component in component_files:
        source_dict.update(_load_weight_dict(component, load_fn))

    result: list[tuple[str, Any]] = []
    skipped = 0
    for source_key, tensor in source_dict.items():
        model_key = map_gguf_key_to_model_key(source_key)
        expected_shape = expected_shapes.get(model_key)
        if expected_shape is None:
            skipped += 1
            continue
        tensor = _transpose_if_needed(tensor, expected_shape, source_key, model_key)
        result.append((model_key, tensor))

    if not result:
        raise ValueError("No compatible base-model weights found after key mapping")

    stats = {
        "loaded": len(result),
        "skipped": skipped,
        "source_tensors": len(source_dict),
    }
    return result, stats

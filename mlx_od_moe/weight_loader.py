"""
Utilities for loading base model weights from GGUF-converted outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import re


BASE_WEIGHT_FILES = (
    "embeddings.safetensors",
    "attention_layers.safetensors",
    "norms.safetensors",
    "lm_head.safetensors",
)


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


def _transpose_if_needed(tensor: Any, expected_shape: tuple[int, ...]) -> Any:
    tensor_shape = _as_tuple(tensor.shape)
    if tensor_shape == expected_shape:
        return tensor
    if len(tensor_shape) == 2 and tensor_shape[::-1] == expected_shape:
        return tensor.T
    raise ValueError(
        f"Shape mismatch: got {tensor_shape}, expected {expected_shape}"
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
    if base_path.is_dir():
        component_files = [base_path / name for name in BASE_WEIGHT_FILES if (base_path / name).exists()]
        if not component_files:
            raise ValueError(
                f"No base safetensors files found in directory: {base_path}"
            )
        for component in component_files:
            source_dict.update(_load_weight_dict(component, load_fn))
    else:
        source_dict = _load_weight_dict(base_path, load_fn)

    result: list[tuple[str, Any]] = []
    skipped = 0
    for source_key, tensor in source_dict.items():
        model_key = map_gguf_key_to_model_key(source_key)
        expected_shape = expected_shapes.get(model_key)
        if expected_shape is None:
            skipped += 1
            continue
        tensor = _transpose_if_needed(tensor, expected_shape)
        result.append((model_key, tensor))

    if not result:
        raise ValueError("No compatible base-model weights found after key mapping")

    stats = {
        "loaded": len(result),
        "skipped": skipped,
        "source_tensors": len(source_dict),
    }
    return result, stats

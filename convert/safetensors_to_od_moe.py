"""
Safetensors -> OD-MoE Converter.

Converts model safetensors weights into per-expert OD-MoE layout:
- base_model/*.safetensors
- experts/layer_XX_expert_YYY.safetensors
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Dict

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


BASE_WEIGHT_FILES = (
    "embeddings.safetensors",
    "attention_layers.safetensors",
    "norms.safetensors",
    "lm_head.safetensors",
)


def _resolve_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(p for p in input_path.glob("*.safetensors") if p.is_file())
        if not files:
            raise ValueError(f"No .safetensors files found in: {input_path}")
        return files
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _load_tensors_from_safetensors(input_path: Path) -> Dict[str, np.ndarray]:
    tensors: Dict[str, np.ndarray] = {}
    for file_path in _resolve_input_files(input_path):
        try:
            with safe_open(file_path, framework="numpy") as reader:
                for key in reader.keys():
                    tensors[key] = reader.get_tensor(key)
        except TypeError as e:
            if "bfloat16" not in str(e).lower():
                raise
            # Fallback for numpy builds without BF16 dtype support.
            tensors.update(_load_tensors_from_safetensors_raw(file_path))
    if not tensors:
        raise ValueError(f"No tensors loaded from {input_path}")
    return tensors


def _decode_bfloat16_bytes(raw: bytes, shape: list[int]) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    f32 = u32.view(np.float32)
    return f32.reshape(tuple(shape))


def _decode_tensor_bytes(dtype_tag: str, raw: bytes, shape: list[int]) -> np.ndarray:
    dtype_map = {
        "F64": np.float64,
        "F32": np.float32,
        "F16": np.float16,
        "I64": np.int64,
        "I32": np.int32,
        "I16": np.int16,
        "I8": np.int8,
        "U64": np.uint64,
        "U32": np.uint32,
        "U16": np.uint16,
        "U8": np.uint8,
        "BOOL": np.bool_,
    }
    if dtype_tag == "BF16":
        return _decode_bfloat16_bytes(raw, shape)
    if dtype_tag not in dtype_map:
        raise ValueError(f"Unsupported safetensors dtype tag: {dtype_tag}")
    arr = np.frombuffer(raw, dtype=dtype_map[dtype_tag])
    return arr.reshape(tuple(shape))


def _load_tensors_from_safetensors_raw(file_path: Path) -> Dict[str, np.ndarray]:
    tensors: Dict[str, np.ndarray] = {}
    with open(file_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = f.read()

    for key, info in header.items():
        if key == "__metadata__":
            continue
        dtype_tag = info["dtype"]
        shape = info["shape"]
        start, end = info["data_offsets"]
        tensors[key] = _decode_tensor_bytes(dtype_tag, data[start:end], shape)

    return tensors


def _cast_tensor(x: np.ndarray, output_dtype: np.dtype) -> np.ndarray:
    return x if x.dtype == output_dtype else x.astype(output_dtype)


def _slice_packed_expert_tensor(tensor_data: np.ndarray, expert_idx: int, num_experts: int):
    shape = tuple(int(x) for x in tensor_data.shape)
    matching_axes = [axis for axis, dim in enumerate(shape) if dim == num_experts]
    if not matching_axes:
        raise ValueError(f"No expert axis found for packed tensor shape {shape}")
    if len(matching_axes) > 1:
        raise ValueError(
            f"Ambiguous expert axis for packed tensor shape {shape}: {matching_axes}"
        )
    return tensor_data.take(indices=expert_idx, axis=matching_axes[0])


def _extract_expert_tensors_for_layer(
    tensor_lookup: Dict[str, np.ndarray],
    layer: int,
    expert: int,
    num_experts: int,
):
    # Explicit expert naming.
    w1_name = f"blk.{layer}.ffn.experts.{expert}.w1.weight"
    w2_name = f"blk.{layer}.ffn.experts.{expert}.w2.weight"
    w3_name = f"blk.{layer}.ffn.experts.{expert}.w3.weight"

    tensors = {}
    if w1_name in tensor_lookup:
        tensors["w1.weight"] = tensor_lookup[w1_name]
    if w2_name in tensor_lookup:
        tensors["w2.weight"] = tensor_lookup[w2_name]
    if w3_name in tensor_lookup:
        tensors["w3.weight"] = tensor_lookup[w3_name]
    if tensors:
        return tensors

    # Alternate explicit style.
    g_name = f"layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
    d_name = f"layers.{layer}.mlp.experts.{expert}.down_proj.weight"
    u_name = f"layers.{layer}.mlp.experts.{expert}.up_proj.weight"
    if g_name in tensor_lookup and d_name in tensor_lookup and u_name in tensor_lookup:
        return {
            "w1.weight": tensor_lookup[g_name],
            "w2.weight": tensor_lookup[d_name],
            "w3.weight": tensor_lookup[u_name],
        }

    # Packed per-layer style.
    gate_exps_name = f"blk.{layer}.ffn_gate_exps.weight"
    down_exps_name = f"blk.{layer}.ffn_down_exps.weight"
    up_exps_name = f"blk.{layer}.ffn_up_exps.weight"
    if (
        gate_exps_name in tensor_lookup
        and down_exps_name in tensor_lookup
        and up_exps_name in tensor_lookup
    ):
        gate_exps = tensor_lookup[gate_exps_name]
        down_exps = tensor_lookup[down_exps_name]
        up_exps = tensor_lookup[up_exps_name]
        return {
            "w1.weight": _slice_packed_expert_tensor(gate_exps, expert, num_experts),
            "w2.weight": _slice_packed_expert_tensor(down_exps, expert, num_experts),
            "w3.weight": _slice_packed_expert_tensor(up_exps, expert, num_experts),
        }

    return {}


def _infer_num_layers(tensor_lookup: Dict[str, np.ndarray]) -> int | None:
    indices = set()
    for key in tensor_lookup:
        m = re.match(r"blk\.(\d+)\.", key)
        if m:
            indices.add(int(m.group(1)))
        m2 = re.match(r"layers\.(\d+)\.", key)
        if m2:
            indices.add(int(m2.group(1)))
    return max(indices) + 1 if indices else None


def _infer_num_experts(tensor_lookup: Dict[str, np.ndarray]) -> int | None:
    explicit_ids = set()
    for key in tensor_lookup:
        m = re.match(r"blk\.\d+\.ffn\.experts\.(\d+)\.", key)
        if m:
            explicit_ids.add(int(m.group(1)))
        m2 = re.match(r"layers\.\d+\.mlp\.experts\.(\d+)\.", key)
        if m2:
            explicit_ids.add(int(m2.group(1)))
    if explicit_ids:
        return max(explicit_ids) + 1

    for key in tensor_lookup:
        if key.endswith(".ffn_gate_exps.weight") or key.endswith(".ffn_down_exps.weight") or key.endswith(".ffn_up_exps.weight"):
            shape = tuple(int(x) for x in tensor_lookup[key].shape)
            # Pick smallest axis > 1 as likely expert axis for packed tensors.
            candidates = [d for d in shape if d > 1]
            if candidates:
                return min(candidates)
    return None


def _extract_base_model(
    tensor_lookup: Dict[str, np.ndarray], output_dir: Path, output_dtype: np.dtype
):
    base_dir = output_dir / "base_model"
    base_dir.mkdir(parents=True, exist_ok=True)

    embeddings = {}
    lm_head = {}
    norms = {}
    attention_layers = {}

    for tensor_name, tensor_data in tensor_lookup.items():
        tensor_data = _cast_tensor(tensor_data, output_dtype)

        if "token_embd" in tensor_name:
            embeddings[tensor_name] = tensor_data
        elif tensor_name == "output.weight":
            lm_head[tensor_name] = tensor_data
        elif "norm" in tensor_name and "experts" not in tensor_name:
            norms[tensor_name] = tensor_data
        elif (
            "attn" in tensor_name
            or "ffn.gate" in tensor_name
            or "ffn_gate_inp" in tensor_name
        ):
            attention_layers[tensor_name] = tensor_data

    if embeddings:
        save_file(embeddings, base_dir / "embeddings.safetensors")
    if attention_layers:
        save_file(attention_layers, base_dir / "attention_layers.safetensors")
    if norms:
        save_file(norms, base_dir / "norms.safetensors")
    if lm_head:
        save_file(lm_head, base_dir / "lm_head.safetensors")

    metadata = {
        "extracted": True,
        "output_dtype": str(np.dtype(output_dtype)),
        "components": {
            "embeddings": len(embeddings),
            "attention_layers": len(attention_layers),
            "norms": len(norms),
            "lm_head": len(lm_head),
        },
    }
    with open(base_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _extract_experts(
    tensor_lookup: Dict[str, np.ndarray],
    output_dir: Path,
    num_layers: int,
    num_experts: int,
    output_dtype: np.dtype,
):
    expert_dir = output_dir / "experts"
    expert_dir.mkdir(parents=True, exist_ok=True)

    expert_registry = {}
    for layer in range(num_layers):
        for expert in range(num_experts):
            key = f"layer_{layer:02d}_expert_{expert:03d}"
            expert_tensors = _extract_expert_tensors_for_layer(
                tensor_lookup=tensor_lookup,
                layer=layer,
                expert=expert,
                num_experts=num_experts,
            )
            for tensor_name in list(expert_tensors.keys()):
                expert_tensors[tensor_name] = _cast_tensor(
                    expert_tensors[tensor_name], output_dtype
                )

            save_path = expert_dir / f"{key}.safetensors"
            save_file(expert_tensors, save_path)
            expert_registry[key] = {
                "path": f"experts/{key}.safetensors",
                "size": save_path.stat().st_size,
                "layer": layer,
                "expert_id": expert,
                "tensors": list(expert_tensors.keys()),
            }

    with open(expert_dir / "registry.json", "w", encoding="utf-8") as f:
        json.dump(expert_registry, f, indent=2)


def convert_safetensors_to_od_moe(
    input_path: str,
    output_dir: str,
    num_layers: int | None = None,
    num_experts: int | None = None,
    output_dtype: str = "float16",
):
    input_file = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float16": np.float16, "float32": np.float32}
    if output_dtype not in dtype_map:
        raise ValueError(f"Unsupported output dtype: {output_dtype}")
    np_output_dtype = dtype_map[output_dtype]

    tensors = _load_tensors_from_safetensors(input_file)

    if num_layers is None:
        num_layers = _infer_num_layers(tensors) or 0
    if num_experts is None:
        num_experts = _infer_num_experts(tensors) or 0
    if num_layers <= 0:
        raise ValueError("Could not infer num_layers from safetensors input")
    if num_experts <= 0:
        raise ValueError("Could not infer num_experts from safetensors input")

    print(f"Loaded {len(tensors)} tensors from {input_file}")
    print(f"Using num_layers={num_layers}, num_experts={num_experts}, output_dtype={output_dtype}")

    _extract_base_model(tensors, output_path, np_output_dtype)
    _extract_experts(tensors, output_path, num_layers, num_experts, np_output_dtype)

    print("\n✅ Conversion complete!")
    print(f"   Base model: {output_path}/base_model/")
    print(f"   Experts: {output_path}/experts/")


def main():
    parser = argparse.ArgumentParser(description="Convert safetensors to OD-MoE format")
    parser.add_argument("--input", required=True, help="Input safetensors file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers (default: infer from tensor keys)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Experts per layer (default: infer from tensor keys)",
    )
    parser.add_argument(
        "--output-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Output tensor dtype for saved safetensors (default: float16)",
    )

    args = parser.parse_args()
    convert_safetensors_to_od_moe(
        input_path=args.input,
        output_dir=args.output,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        output_dtype=args.output_dtype,
    )


if __name__ == "__main__":
    main()

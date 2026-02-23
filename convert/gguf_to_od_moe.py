"""
GGUF → OD-MoE Converter

Converts monolithic GGUF model to per-expert safetensors files
for memory-mapped loading.

Input: Kimi-K2.5.gguf (375GB)
Output: 10,752 expert files + base model
"""

from pathlib import Path
import argparse
from typing import Dict
import json
import numpy as np
import gguf
from safetensors.numpy import save_file
from tqdm import tqdm


def _read_tensor_data(tensor):
    """
    Read tensor payload from GGUF.

    Current converter expects dequantized numeric arrays. Quantized GGUF blobs from
    runtimes like Ollama expose packed uint8 blocks, which need explicit dequantization.
    """
    data = tensor.data
    meta_shape = tuple(int(x) for x in tensor.shape)
    data_shape = tuple(int(x) for x in data.shape)
    if data.dtype == np.uint8 and data_shape != meta_shape:
        raise ValueError(
            f"Tensor {tensor.name} is quantized/packed (dtype={data.dtype}, "
            f"data_shape={data_shape}, expected_shape={meta_shape}). "
            "This converter currently requires dequantized/fp tensors."
        )
    return data


def parse_gguf_metadata(filepath: Path) -> Dict:
    """
    Parse GGUF file header and extract metadata.

    GGUF format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

    Returns:
        Metadata dict with tensor info including:
        - architecture: Model architecture name
        - num_layers: Number of transformer layers
        - num_experts: Number of experts per layer
        - dim: Model dimension
        - vocab_size: Vocabulary size
        - tensors: Dict of tensor metadata
        - total_tensors: Total number of tensors
    """
    # Validate file exists
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Open with GGUFReader
    try:
        reader = gguf.GGUFReader(str(filepath))
    except Exception as e:
        raise ValueError(f"Not a GGUF file: {filepath}") from e

    # Extract architecture metadata
    architecture = None
    if "general.architecture" in reader.fields:
        arch_field = reader.fields["general.architecture"]
        # The architecture string is in parts[-1] as bytes
        if hasattr(arch_field.parts, '__iter__') and len(arch_field.parts) > 0:
            arch_bytes = arch_field.parts[-1]
            if hasattr(arch_bytes, 'tobytes'):
                architecture = arch_bytes.tobytes().decode('utf-8')
            elif isinstance(arch_bytes, bytes):
                architecture = arch_bytes.decode('utf-8')
            else:
                architecture = str(arch_bytes)

    # Extract layer count
    num_layers = None
    if architecture:
        layer_key = f"{architecture}.block_count"
        if layer_key in reader.fields:
            # The value is in parts[-1] as numpy array
            value = reader.fields[layer_key].parts[-1]
            num_layers = int(value.item() if hasattr(value, 'item') else value)

    # Extract model dimension
    dim = None
    if architecture:
        dim_key = f"{architecture}.embedding_length"
        if dim_key in reader.fields:
            # The value is in parts[-1] as numpy array
            value = reader.fields[dim_key].parts[-1]
            dim = int(value.item() if hasattr(value, 'item') else value)

    # Extract number of experts
    num_experts = None
    expert_key = "kimi.num_experts_per_layer"
    if expert_key in reader.fields:
        # The value is in parts[-1] as numpy array
        value = reader.fields[expert_key].parts[-1]
        num_experts = int(value.item() if hasattr(value, 'item') else value)
    elif architecture:
        # Llama/Mixtral-style metadata
        arch_expert_key = f"{architecture}.expert_count"
        if arch_expert_key in reader.fields:
            value = reader.fields[arch_expert_key].parts[-1]
            num_experts = int(value.item() if hasattr(value, 'item') else value)

    # Build tensor metadata dict
    tensors = {}
    for tensor in reader.tensors:
        tensors[tensor.name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.tensor_type),
            "offset": tensor.data_offset,
            "n_elements": tensor.n_elements
        }

    # Extract vocab size from token embeddings tensor
    # Token embeddings are [vocab_size, dim] in this converter/test setup.
    vocab_size = None
    if "token_embd.weight" in tensors:
        vocab_size = tensors["token_embd.weight"]["shape"][0]

    total_tensors = len(tensors)

    # Print metadata summary
    print(f"\nGGUF Metadata Summary:")
    print(f"  Architecture: {architecture}")
    print(f"  Layers: {num_layers}")
    print(f"  Experts/layer: {num_experts}")
    print(f"  Dimension: {dim}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Total tensors: {total_tensors}")

    return {
        "architecture": architecture,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "dim": dim,
        "vocab_size": vocab_size,
        "tensors": tensors,
        "total_tensors": total_tensors
    }


def extract_base_model(gguf_path: Path, output_dir: Path):
    """
    Extract base model components (embeddings, attention, norms).

    These are always-resident components that don't change during inference.
    """
    print("Extracting base model...")

    # Create output directory
    base_dir = output_dir / "base_model"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Read GGUF file
    reader = gguf.GGUFReader(str(gguf_path))

    # Categorize tensors into 4 dicts
    embeddings = {}
    lm_head = {}
    norms = {}
    attention_layers = {}

    # Iterate through tensors with progress bar
    for tensor in tqdm(reader.tensors, desc="Reading tensors"):
        tensor_name = tensor.name
        tensor_data = _read_tensor_data(tensor)

        # Categorize based on tensor name
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
            # Attention layers + router/gate (part of base model)
            attention_layers[tensor_name] = tensor_data


def _slice_packed_expert_tensor(tensor_data, expert_idx: int, num_experts: int):
    """
    Slice one expert from packed MoE tensors.

    Supports layouts where one axis equals num_experts, e.g. [H, F, E].
    """
    shape = tuple(int(x) for x in tensor_data.shape)
    matching_axes = [axis for axis, dim in enumerate(shape) if dim == num_experts]
    if not matching_axes:
        raise ValueError(f"No expert axis found for packed tensor shape {shape}")
    if len(matching_axes) > 1:
        raise ValueError(
            f"Ambiguous expert axis for packed tensor shape {shape}: {matching_axes}"
        )

    axis = matching_axes[0]
    return tensor_data.take(indices=expert_idx, axis=axis)


def _extract_expert_tensors_for_layer(
    tensor_lookup: dict,
    layer: int,
    expert: int,
    num_experts: int,
):
    """
    Extract one expert from either:
    - explicit per-expert tensors, or
    - packed per-layer expert tensors (Mixtral/Codextral style).
    """
    # Format A: explicit per-expert names
    w1_name = f"blk.{layer}.ffn.experts.{expert}.w1.weight"
    w2_name = f"blk.{layer}.ffn.experts.{expert}.w2.weight"
    w3_name = f"blk.{layer}.ffn.experts.{expert}.w3.weight"

    expert_tensors = {}
    if w1_name in tensor_lookup:
        expert_tensors["w1.weight"] = _read_tensor_data(tensor_lookup[w1_name])
    if w2_name in tensor_lookup:
        expert_tensors["w2.weight"] = _read_tensor_data(tensor_lookup[w2_name])
    if w3_name in tensor_lookup:
        expert_tensors["w3.weight"] = _read_tensor_data(tensor_lookup[w3_name])
    if expert_tensors:
        return expert_tensors

    # Format B: packed per-layer tensors
    gate_exps_name = f"blk.{layer}.ffn_gate_exps.weight"
    down_exps_name = f"blk.{layer}.ffn_down_exps.weight"
    up_exps_name = f"blk.{layer}.ffn_up_exps.weight"

    if (
        gate_exps_name in tensor_lookup
        and down_exps_name in tensor_lookup
        and up_exps_name in tensor_lookup
    ):
        gate_exps = _read_tensor_data(tensor_lookup[gate_exps_name])
        down_exps = _read_tensor_data(tensor_lookup[down_exps_name])
        up_exps = _read_tensor_data(tensor_lookup[up_exps_name])

        # Runtime expects:
        # w1 = gate projection, w2 = down projection, w3 = up projection
        return {
            "w1.weight": _slice_packed_expert_tensor(gate_exps, expert, num_experts),
            "w2.weight": _slice_packed_expert_tensor(down_exps, expert, num_experts),
            "w3.weight": _slice_packed_expert_tensor(up_exps, expert, num_experts),
        }

    return {}

    # Save categorized tensors as safetensors files
    if embeddings:
        save_file(embeddings, base_dir / "embeddings.safetensors")

    if attention_layers:
        save_file(attention_layers, base_dir / "attention_layers.safetensors")

    if norms:
        save_file(norms, base_dir / "norms.safetensors")

    if lm_head:
        save_file(lm_head, base_dir / "lm_head.safetensors")

    # Save metadata.json
    metadata = {
        'extracted': True,
        'components': {
            'embeddings': len(embeddings),
            'attention_layers': len(attention_layers),
            'norms': len(norms),
            'lm_head': len(lm_head)
        }
    }

    with open(base_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Base model extracted to {base_dir}")


def extract_experts(gguf_path: Path, output_dir: Path, num_layers: int = 28, num_experts: int = 384):
    """
    Extract individual experts to separate safetensors files.

    Creates: experts/layer_XX_expert_XXX.safetensors

    Each expert contains: w1, w2, w3 for Gated MLP
    """
    print(f"Extracting {num_layers * num_experts} experts...")

    # Create output directory
    expert_dir = output_dir / "experts"
    expert_dir.mkdir(parents=True, exist_ok=True)

    # Read GGUF file
    reader = gguf.GGUFReader(str(gguf_path))

    # Build tensor lookup
    tensor_lookup = {tensor.name: tensor for tensor in reader.tensors}

    # Initialize registry
    expert_registry = {}

    # Extract experts with progress bar
    total_experts = num_layers * num_experts
    with tqdm(total=total_experts, desc="Extracting experts") as pbar:
        for layer in range(num_layers):
            for expert in range(num_experts):
                # Build key
                key = f"layer_{layer:02d}_expert_{expert:03d}"

                expert_tensors = _extract_expert_tensors_for_layer(
                    tensor_lookup=tensor_lookup,
                    layer=layer,
                    expert=expert,
                    num_experts=num_experts,
                )

                # Save to safetensors
                save_path = expert_dir / f"{key}.safetensors"
                save_file(expert_tensors, save_path)

                # Add to registry
                expert_registry[key] = {
                    "path": f"experts/{key}.safetensors",  # Relative to output_dir
                    "size": save_path.stat().st_size,
                    "layer": layer,
                    "expert_id": expert,
                    "tensors": list(expert_tensors.keys())
                }

                # Update progress
                pbar.update(1)

    # Save registry
    registry_path = expert_dir / "registry.json"
    with open(registry_path, 'w') as f:
        json.dump(expert_registry, f, indent=2)

    # Print summary
    total_size_bytes = sum(info["size"] for info in expert_registry.values())
    total_size_gb = total_size_bytes / (1024 ** 3)

    print(f"\nExperts extracted to {expert_dir}")
    print(f"Total size: {total_size_gb:.2f} GB")
    print(f"Registry saved: {registry_path}")


def convert_gguf_to_od_moe(
    input_path: str,
    output_dir: str,
    num_layers: int | None = None,
    num_experts: int | None = None,
):
    """
    Full conversion: GGUF → OD-MoE format.
    
    Args:
        input_path: Path to GGUF file
        output_dir: Output directory for experts + base model
        num_layers: Number of transformer layers (auto-detected if None)
        num_experts: Experts per layer (auto-detected if None)
    """
    input_file = Path(input_path)
    output_path = Path(output_dir)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input GGUF not found: {input_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {input_file.name}...")
    print(f"Output: {output_path}")
    
    # Parse metadata
    metadata = parse_gguf_metadata(input_file)

    # Auto-detect layers/experts when not explicitly provided.
    if num_layers is None:
        num_layers = metadata.get("num_layers") or 28
        print(f"Auto-detected num_layers={num_layers}")
    if num_experts is None:
        num_experts = metadata.get("num_experts") or 384
        print(f"Auto-detected num_experts={num_experts}")
    
    # Extract base model
    extract_base_model(input_file, output_path)
    
    # Extract experts
    extract_experts(input_file, output_path, num_layers, num_experts)
    
    print("\n✅ Conversion complete!")
    print(f"   Base model: {output_path}/base_model/")
    print(f"   Experts: {output_path}/experts/")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Convert GGUF to OD-MoE format")
    parser.add_argument("--input", required=True, help="Input GGUF file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers (default: auto-detect from GGUF metadata)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Experts per layer (default: auto-detect from GGUF metadata)",
    )
    
    args = parser.parse_args()
    
    convert_gguf_to_od_moe(
        args.input,
        args.output,
        args.num_layers,
        args.num_experts
    )


if __name__ == "__main__":
    main()

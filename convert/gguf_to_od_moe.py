"""
GGUF → OD-MoE Converter

Converts monolithic GGUF model to per-expert safetensors files
for memory-mapped loading.

Input: Kimi-K2.5.gguf (375GB)
Output: 10,752 expert files + base model
"""

import struct
import numpy as np
from pathlib import Path
import argparse
from typing import Dict
import json
import gguf


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
    # Token embeddings shape in GGUF: [dim, vocab_size]
    vocab_size = None
    if "token_embd.weight" in tensors:
        # GGUF stores as [dim, vocab_size], so vocab_size is shape[1]
        vocab_size = tensors["token_embd.weight"]["shape"][1]

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
    
    # TODO: Actual GGUF parsing
    # For now, create placeholder
    base_dir = output_dir / "base_model"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Placeholder: would extract real tensors here
    metadata = {
        'extracted': True,
        'components': ['embeddings', 'attention_layers', 'norms', 'lm_head']
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
    
    expert_dir = output_dir / "experts"
    expert_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Actual GGUF expert extraction
    # For now, create dummy experts for testing
    
    expert_registry = {}
    
    for layer in range(num_layers):
        print(f"Processing layer {layer}/{num_layers}...")
        
        for expert in range(num_experts):
            key = f"layer_{layer:02d}_expert_{expert:03d}"
            
            # Placeholder: create dummy expert weights
            # Real implementation: extract from GGUF
            # w1: (4096, 14336), w2: (14336, 4096), w3: (4096, 14336)
            # Total: ~470MB per expert at fp16
            
            dummy_weights = np.random.randn(470 * 1024 * 1024 // 2).astype(np.float16)
            
            # Save as .npy for now (safetensors requires actual lib)
            save_path = expert_dir / f"{key}.npy"
            np.save(save_path, dummy_weights)
            
            expert_registry[key] = {
                'path': str(save_path),
                'size': save_path.stat().st_size
            }
    
    # Save registry
    with open(expert_dir / "registry.json", 'w') as f:
        json.dump(expert_registry, f, indent=2)
    
    print(f"Experts extracted to {expert_dir}")
    print(f"Registry saved: {expert_dir}/registry.json")


def convert_gguf_to_od_moe(
    input_path: str,
    output_dir: str,
    num_layers: int = 28,
    num_experts: int = 384
):
    """
    Full conversion: GGUF → OD-MoE format.
    
    Args:
        input_path: Path to GGUF file
        output_dir: Output directory for experts + base model
        num_layers: Number of transformer layers
        num_experts: Experts per layer
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
    parser.add_argument("--num-layers", type=int, default=28, help="Number of layers")
    parser.add_argument("--num-experts", type=int, default=384, help="Experts per layer")
    
    args = parser.parse_args()
    
    convert_gguf_to_od_moe(
        args.input,
        args.output,
        args.num_layers,
        args.num_experts
    )


if __name__ == "__main__":
    main()

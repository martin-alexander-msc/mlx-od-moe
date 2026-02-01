"""
Toy GGUF Model Generator

Creates a tiny GGUF model with the same tensor structure as Kimi-K2.5
(28 layers × 384 experts) but with small dimensions for testing.

This allows testing the converter without needing the 375GB real model.
"""

import argparse
import numpy as np
from pathlib import Path
import gguf


def create_toy_gguf_model(
    output_path: Path,
    num_layers: int = 2,
    num_experts: int = 4,
    dim: int = 512,
    ffn_dim: int = 1024,
    vocab_size: int = 1000,
    num_heads: int = 8
):
    """
    Create a toy GGUF model for testing.

    Args:
        output_path: Where to save the GGUF file
        num_layers: Number of transformer layers (default: 2, real model: 28)
        num_experts: Number of experts per layer (default: 4, real model: 384)
        dim: Model dimension (default: 512, real model: 4096)
        ffn_dim: FFN intermediate dimension (default: 1024, real model: 14336)
        vocab_size: Vocabulary size (default: 1000, real model: ~150k)
        num_heads: Number of attention heads (default: 8)

    The toy model has the same tensor structure as the real Kimi-K2.5 model
    but with tiny dimensions to keep file size small (~2MB instead of 375GB).
    """
    # Ensure output path is a Path object
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create GGUF writer
    writer = gguf.GGUFWriter(path=str(output_path), arch="kimi-moe")

    # Add standard metadata fields
    writer.add_block_count(num_layers)
    writer.add_context_length(4096)
    writer.add_embedding_length(dim)
    writer.add_head_count(num_heads)
    writer.add_feed_forward_length(ffn_dim)

    # Add custom MoE metadata
    writer.add_uint32(f"kimi.num_experts_per_layer", num_experts)
    writer.add_uint32(f"kimi.num_active_experts", 2)  # Top-2 routing

    # Create base model tensors
    # Token embeddings: vocab_size × dim
    token_embd = np.random.randn(vocab_size, dim).astype(np.float32)
    writer.add_tensor("token_embd.weight", token_embd)

    # Output layer: dim × vocab_size
    output = np.random.randn(dim, vocab_size).astype(np.float32)
    writer.add_tensor("output.weight", output)

    # Create tensors for each layer
    for layer_idx in range(num_layers):
        # Attention norm
        attn_norm = np.random.randn(dim).astype(np.float32)
        writer.add_tensor(f"blk.{layer_idx}.attn_norm.weight", attn_norm)

        # Attention weights
        attn_q = np.random.randn(dim, dim).astype(np.float32)
        writer.add_tensor(f"blk.{layer_idx}.attn_q.weight", attn_q)

        attn_k = np.random.randn(dim, dim).astype(np.float32)
        writer.add_tensor(f"blk.{layer_idx}.attn_k.weight", attn_k)

        attn_v = np.random.randn(dim, dim).astype(np.float32)
        writer.add_tensor(f"blk.{layer_idx}.attn_v.weight", attn_v)

        attn_output = np.random.randn(dim, dim).astype(np.float32)
        writer.add_tensor(f"blk.{layer_idx}.attn_output.weight", attn_output)

        # FFN norm
        ffn_norm = np.random.randn(dim).astype(np.float32)
        writer.add_tensor(f"blk.{layer_idx}.ffn_norm.weight", ffn_norm)

        # MoE gate (router)
        gate = np.random.randn(dim, num_experts).astype(np.float32)
        writer.add_tensor(f"blk.{layer_idx}.ffn.gate.weight", gate)

        # Create tensors for each expert
        for expert_idx in range(num_experts):
            # Expert w1: dim × ffn_dim (gate projection)
            w1 = np.random.randn(dim, ffn_dim).astype(np.float32)
            writer.add_tensor(f"blk.{layer_idx}.ffn.experts.{expert_idx}.w1.weight", w1)

            # Expert w2: ffn_dim × dim (down projection)
            w2 = np.random.randn(ffn_dim, dim).astype(np.float32)
            writer.add_tensor(f"blk.{layer_idx}.ffn.experts.{expert_idx}.w2.weight", w2)

            # Expert w3: dim × ffn_dim (up projection)
            w3 = np.random.randn(dim, ffn_dim).astype(np.float32)
            writer.add_tensor(f"blk.{layer_idx}.ffn.experts.{expert_idx}.w3.weight", w3)

    # Write the file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Print summary
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    total_experts = num_layers * num_experts
    print(f"Created toy GGUF model: {output_path}")
    print(f"  Layers: {num_layers}")
    print(f"  Experts per layer: {num_experts}")
    print(f"  Total experts: {total_experts}")
    print(f"  Model dim: {dim}")
    print(f"  FFN dim: {ffn_dim}")
    print(f"  File size: {file_size_mb:.2f} MB")


def main():
    """CLI entry point for creating toy models."""
    parser = argparse.ArgumentParser(
        description="Create a toy GGUF model for testing the converter"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="convert/toy_model.gguf",
        help="Output GGUF file path (default: convert/toy_model.gguf)"
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of layers (default: 2, real model: 28)"
    )
    parser.add_argument(
        "--experts",
        type=int,
        default=4,
        help="Experts per layer (default: 4, real model: 384)"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=512,
        help="Model dimension (default: 512, real model: 4096)"
    )
    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=1024,
        help="FFN dimension (default: 1024, real model: 14336)"
    )

    args = parser.parse_args()

    create_toy_gguf_model(
        output_path=Path(args.output),
        num_layers=args.layers,
        num_experts=args.experts,
        dim=args.dim,
        ffn_dim=args.ffn_dim
    )


if __name__ == "__main__":
    main()

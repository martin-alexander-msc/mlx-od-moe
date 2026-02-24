import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from convert.safetensors_to_od_moe import convert_safetensors_to_od_moe


def _create_toy_safetensors(path: Path, num_layers: int = 2, num_experts: int = 4):
    tensors = {}
    dim = 16
    ffn_dim = 32
    vocab = 100

    tensors["token_embd.weight"] = np.random.randn(vocab, dim).astype(np.float32)
    tensors["output.weight"] = np.random.randn(dim, vocab).astype(np.float32)
    tensors["output_norm.weight"] = np.random.randn(dim).astype(np.float32)

    for layer in range(num_layers):
        tensors[f"blk.{layer}.attn_norm.weight"] = np.random.randn(dim).astype(np.float32)
        tensors[f"blk.{layer}.ffn_norm.weight"] = np.random.randn(dim).astype(np.float32)
        tensors[f"blk.{layer}.attn_q.weight"] = np.random.randn(dim, dim).astype(np.float32)
        tensors[f"blk.{layer}.attn_k.weight"] = np.random.randn(dim, dim).astype(np.float32)
        tensors[f"blk.{layer}.attn_v.weight"] = np.random.randn(dim, dim).astype(np.float32)
        tensors[f"blk.{layer}.attn_output.weight"] = np.random.randn(dim, dim).astype(np.float32)
        tensors[f"blk.{layer}.ffn_gate_inp.weight"] = np.random.randn(dim, num_experts).astype(
            np.float32
        )

        for expert in range(num_experts):
            tensors[f"blk.{layer}.ffn.experts.{expert}.w1.weight"] = np.random.randn(
                dim, ffn_dim
            ).astype(np.float32)
            tensors[f"blk.{layer}.ffn.experts.{expert}.w2.weight"] = np.random.randn(
                ffn_dim, dim
            ).astype(np.float32)
            tensors[f"blk.{layer}.ffn.experts.{expert}.w3.weight"] = np.random.randn(
                dim, ffn_dim
            ).astype(np.float32)

    save_file(tensors, path)


def test_convert_safetensors_to_od_moe(tmp_path):
    source = tmp_path / "model.safetensors"
    out = tmp_path / "od_moe"
    _create_toy_safetensors(source, num_layers=2, num_experts=4)

    convert_safetensors_to_od_moe(str(source), str(out), output_dtype="float16")

    assert (out / "base_model" / "embeddings.safetensors").exists()
    assert (out / "base_model" / "attention_layers.safetensors").exists()
    assert (out / "base_model" / "norms.safetensors").exists()
    assert (out / "base_model" / "lm_head.safetensors").exists()
    assert (out / "experts" / "registry.json").exists()

    with open(out / "experts" / "registry.json", "r", encoding="utf-8") as f:
        registry = json.load(f)
    assert len(registry) == 8
    nonempty = sum(1 for v in registry.values() if set(v["tensors"]) == {"w1.weight", "w2.weight", "w3.weight"})
    assert nonempty == 8

    first_expert = out / "experts" / "layer_00_expert_000.safetensors"
    with safe_open(first_expert, framework="numpy") as f:
        assert "w1.weight" in f.keys()
        assert "w2.weight" in f.keys()
        assert "w3.weight" in f.keys()
        assert f.get_tensor("w1.weight").dtype == np.float16

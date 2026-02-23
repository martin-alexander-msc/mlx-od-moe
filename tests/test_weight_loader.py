from pathlib import Path
import importlib.util

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "mlx_od_moe" / "weight_loader.py"
SPEC = importlib.util.spec_from_file_location("weight_loader", MODULE_PATH)
WEIGHT_LOADER = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(WEIGHT_LOADER)

load_base_weight_items = WEIGHT_LOADER.load_base_weight_items
map_gguf_key_to_model_key = WEIGHT_LOADER.map_gguf_key_to_model_key
infer_config_overrides_from_base_shapes = WEIGHT_LOADER.infer_config_overrides_from_base_shapes
validate_expert_conversion = WEIGHT_LOADER.validate_expert_conversion


def test_map_gguf_key_to_model_key_core_mappings():
    assert map_gguf_key_to_model_key("token_embd.weight") == "embed_tokens.weight"
    assert map_gguf_key_to_model_key("output.weight") == "lm_head.weight"
    assert map_gguf_key_to_model_key("output_norm.weight") == "norm.weight"
    assert map_gguf_key_to_model_key("blk.3.attn_q.weight") == "layers.3.attention.q_proj.weight"
    assert map_gguf_key_to_model_key("blk.3.attn_k.weight") == "layers.3.attention.k_proj.weight"
    assert map_gguf_key_to_model_key("blk.3.attn_v.weight") == "layers.3.attention.v_proj.weight"
    assert map_gguf_key_to_model_key("blk.3.attn_output.weight") == "layers.3.attention.o_proj.weight"
    assert map_gguf_key_to_model_key("blk.3.attn_norm.weight") == "layers.3.input_layernorm.weight"
    assert map_gguf_key_to_model_key("blk.3.ffn_norm.weight") == "layers.3.post_attention_layernorm.weight"
    assert map_gguf_key_to_model_key("blk.3.ffn.gate.weight") == "layers.3.moe.gate.weight"


def test_load_base_weight_items_from_directory_and_transpose(tmp_path):
    base_dir = tmp_path / "base_model"
    base_dir.mkdir()
    for name in (
        "embeddings.safetensors",
        "attention_layers.safetensors",
        "norms.safetensors",
        "lm_head.safetensors",
    ):
        (base_dir / name).touch()

    data_by_file = {
        str(base_dir / "embeddings.safetensors"): {
            "token_embd.weight": np.zeros((11, 7), dtype=np.float32),
        },
        str(base_dir / "attention_layers.safetensors"): {
            "blk.0.attn_q.weight": np.zeros((7, 7), dtype=np.float32),
        },
        str(base_dir / "norms.safetensors"): {
            "blk.0.attn_norm.weight": np.zeros((7,), dtype=np.float32),
            "blk.0.ffn_norm.weight": np.zeros((7,), dtype=np.float32),
        },
        str(base_dir / "lm_head.safetensors"): {
            # Deliberately reversed to verify transpose behavior.
            "output.weight": np.zeros((7, 11), dtype=np.float32),
        },
    }

    def fake_load(path: str):
        return data_by_file[path]

    expected_shapes = {
        "embed_tokens.weight": (11, 7),
        "layers.0.attention.q_proj.weight": (7, 7),
        "layers.0.input_layernorm.weight": (7,),
        "layers.0.post_attention_layernorm.weight": (7,),
        "lm_head.weight": (11, 7),
    }

    weight_items, stats = load_base_weight_items(str(base_dir), fake_load, expected_shapes)
    loaded = dict(weight_items)

    assert stats["loaded"] == 5
    assert stats["source_tensors"] == 5
    assert loaded["embed_tokens.weight"].shape == (11, 7)
    assert loaded["lm_head.weight"].shape == (11, 7)


def test_load_base_weight_items_raises_for_empty_base_dir(tmp_path):
    base_dir = tmp_path / "base_model"
    base_dir.mkdir()

    with pytest.raises(ValueError, match="No base safetensors files found"):
        load_base_weight_items(str(base_dir), lambda _: {}, {})


def test_load_base_weight_items_raises_for_missing_path(tmp_path):
    missing = tmp_path / "missing.safetensors"
    with pytest.raises(FileNotFoundError):
        load_base_weight_items(str(missing), lambda _: {}, {})


def test_infer_config_overrides_from_base_shapes(monkeypatch):
    def fake_shapes(_base_weights: str):
        return {
            "token_embd.weight": (32000, 2304),
            "blk.0.attn_q.weight": (4096, 2304),
            "blk.31.attn_q.weight": (4096, 2304),
        }

    monkeypatch.setattr(WEIGHT_LOADER, "inspect_base_weight_shapes", fake_shapes)
    inferred = infer_config_overrides_from_base_shapes("/tmp/base_model")
    assert inferred == {"vocab_size": 32000, "hidden_size": 2304, "num_hidden_layers": 32}


def test_validate_expert_conversion_rejects_empty_registry(tmp_path):
    expert_dir = tmp_path / "experts"
    expert_dir.mkdir()
    (expert_dir / "registry.json").write_text(
        '{"layer_00_expert_000":{"path":"x","size":16,"layer":0,"expert_id":0,"tensors":[]}}'
    )

    with pytest.raises(ValueError, match="All converted experts are empty"):
        validate_expert_conversion(str(expert_dir))

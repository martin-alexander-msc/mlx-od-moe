import mlx.core as mx

from mlx_od_moe.server import _preprocess_qwen3next_source_tensors


def test_preprocess_qwen3next_fuses_qkvz_and_shifts_norms():
    source = {
        "blk.0.attn_qkv.weight": mx.ones((4, 6)),
        "blk.0.attn_gate.weight": mx.full((4, 2), 2.0),
        "blk.0.attn_norm.weight": mx.zeros((4,)),
        "blk.0.post_attention_norm.weight": mx.full((4,), 0.5),
        "blk.0.attn_q_norm.weight": mx.full((2,), -0.25),
        "blk.0.attn_k_norm.weight": mx.full((2,), 0.75),
        "output_norm.weight": mx.full((3,), 0.1),
        "blk.0.ssm_norm.weight": mx.full((4,), 0.2),
        "blk.0.attn_output.weight": mx.full((6, 4), 3.0),
    }

    processed = _preprocess_qwen3next_source_tensors(source)

    assert "blk.0.attn_qkvz.weight" in processed
    assert "blk.0.attn_qkv.weight" not in processed
    assert "blk.0.attn_gate.weight" not in processed
    assert processed["blk.0.attn_qkvz.weight"].shape == (4, 8)
    assert float(processed["blk.0.attn_qkvz.weight"][0, -1].item()) == 2.0

    assert mx.allclose(processed["blk.0.attn_norm.weight"], mx.ones((4,)))
    assert mx.allclose(processed["blk.0.post_attention_norm.weight"], mx.full((4,), 1.5))
    assert mx.allclose(processed["blk.0.attn_q_norm.weight"], mx.full((2,), 0.75))
    assert mx.allclose(processed["blk.0.attn_k_norm.weight"], mx.full((2,), 1.75))
    assert mx.allclose(processed["output_norm.weight"], mx.full((3,), 1.1))

    # Non-target tensors are unchanged.
    assert mx.allclose(processed["blk.0.ssm_norm.weight"], mx.full((4,), 0.2))
    assert mx.allclose(processed["blk.0.attn_output.weight"], mx.full((6, 4), 3.0))

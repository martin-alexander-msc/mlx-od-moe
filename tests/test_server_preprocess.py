import mlx.core as mx

from mlx_od_moe.server import _preprocess_qwen3next_source_tensors


def test_preprocess_qwen3next_fuses_qkvz_and_shifts_norms():
    source = {
        "blk.0.attn_qkv.weight": mx.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
            ]
        ),
        "blk.0.attn_gate.weight": mx.array(
            [
                [100.0, 101.0],
                [102.0, 103.0],
                [104.0, 105.0],
                [106.0, 107.0],
            ]
        ),
        "blk.0.attn_norm.weight": mx.zeros((4,)),
        "blk.0.post_attention_norm.weight": mx.full((4,), 0.5),
        "blk.0.attn_q_norm.weight": mx.full((2,), -0.25),
        "blk.0.attn_k_norm.weight": mx.full((2,), 0.75),
        "output_norm.weight": mx.full((3,), 0.1),
        "blk.0.ssm_norm.weight": mx.full((4,), 0.2),
        "blk.0.attn_output.weight": mx.full((6, 4), 3.0),
    }

    processed = _preprocess_qwen3next_source_tensors(source, linear_num_key_heads=2)

    assert "blk.0.attn_qkvz.weight" in processed
    assert "blk.0.attn_qkv.weight" not in processed
    assert "blk.0.attn_gate.weight" not in processed
    assert processed["blk.0.attn_qkvz.weight"].shape == (4, 8)
    expected_row0 = mx.array([0.0, 1.0, 2.0, 100.0, 3.0, 4.0, 5.0, 101.0])
    assert mx.allclose(processed["blk.0.attn_qkvz.weight"][0], expected_row0)

    assert mx.allclose(processed["blk.0.attn_norm.weight"], mx.ones((4,)))
    assert mx.allclose(processed["blk.0.post_attention_norm.weight"], mx.full((4,), 1.5))
    assert mx.allclose(processed["blk.0.attn_q_norm.weight"], mx.full((2,), 0.75))
    assert mx.allclose(processed["blk.0.attn_k_norm.weight"], mx.full((2,), 1.75))
    assert mx.allclose(processed["output_norm.weight"], mx.full((3,), 1.1))

    # Non-target tensors are unchanged.
    assert mx.allclose(processed["blk.0.ssm_norm.weight"], mx.full((4,), 0.2))
    assert mx.allclose(processed["blk.0.attn_output.weight"], mx.full((6, 4), 3.0))

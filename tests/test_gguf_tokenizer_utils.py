import numpy as np

from mlx_od_moe.gguf_tokenizer import _to_utf8_text


def test_to_utf8_text_handles_multiple_binary_forms():
    assert _to_utf8_text("hello") == "hello"
    assert _to_utf8_text(b"hello") == "hello"
    assert _to_utf8_text([104, 101, 108, 108, 111]) == "hello"
    assert _to_utf8_text(np.array([104, 101, 108, 108, 111], dtype=np.uint8)) == "hello"


def test_to_utf8_text_handles_joined_string_lists():
    assert _to_utf8_text(["he", "llo"]) == "hello"

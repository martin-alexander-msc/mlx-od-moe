"""
GGUF-backed tokenizer loader.

Supports tokenizer initialization directly from GGUF metadata so server
inference can run without external HF tokenizer assets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional
import json

import gguf


class GGUFTokenizer:
    """Minimal tokenizer wrapper with encode/decode callables."""

    def __init__(
        self,
        encode_fn: Callable[[str], list[int]],
        decode_fn: Callable[[list[int]], str],
        source: str,
    ):
        self._encode = encode_fn
        self._decode = decode_fn
        self.source = source

    def encode(self, text: str, return_tensors: Any = None) -> list[int]:
        del return_tensors
        return self._encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._decode(token_ids)


def _field_contents(reader: gguf.GGUFReader, key: str, default: Any = None) -> Any:
    field = reader.fields.get(key)
    if field is None:
        return default
    try:
        return field.contents()
    except Exception:
        return default


def _load_hf_tokenizer_from_gguf_json(
    hf_json: str,
    source: str,
) -> Optional[GGUFTokenizer]:
    try:
        from tokenizers import Tokenizer
    except Exception:
        return None

    try:
        tok = Tokenizer.from_str(hf_json)
    except Exception:
        return None

    def _encode(text: str) -> list[int]:
        return tok.encode(text).ids

    def _decode(ids: list[int]) -> str:
        return tok.decode(ids)

    return GGUFTokenizer(_encode, _decode, source)


def _load_gpt2_bpe_tokenizer_from_gguf(
    reader: gguf.GGUFReader,
    source: str,
) -> Optional[GGUFTokenizer]:
    model = _field_contents(reader, "tokenizer.ggml.model")
    if str(model).lower() != "gpt2":
        return None

    tokens = _field_contents(reader, "tokenizer.ggml.tokens")
    merges = _field_contents(reader, "tokenizer.ggml.merges")
    if not isinstance(tokens, list) or not isinstance(merges, list):
        return None

    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    except Exception:
        return None

    vocab: dict[str, int] = {}
    for idx, token in enumerate(tokens):
        if isinstance(token, str):
            vocab[token] = idx
        elif isinstance(token, bytes):
            vocab[token.decode("utf-8")] = idx
        else:
            vocab[str(token)] = idx

    parsed_merges: list[tuple[str, str]] = []
    for merge in merges:
        if isinstance(merge, bytes):
            merge = merge.decode("utf-8")
        if not isinstance(merge, str):
            continue
        parts = merge.split(" ", 1)
        if len(parts) != 2:
            continue
        parsed_merges.append((parts[0], parts[1]))

    try:
        bpe = BPE(vocab=vocab, merges=parsed_merges, unk_token=None, fuse_unk=False)
        tokenizer = Tokenizer(bpe)
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.decoder = ByteLevelDecoder()
    except Exception:
        return None

    def _encode(text: str) -> list[int]:
        return tokenizer.encode(text).ids

    def _decode(ids: list[int]) -> str:
        return tokenizer.decode(ids)

    return GGUFTokenizer(_encode, _decode, source)


def load_tokenizer_from_gguf(gguf_path: str) -> GGUFTokenizer:
    """
    Build a tokenizer directly from GGUF metadata.

    Resolution order:
    1. tokenizer.huggingface.json (exact tokenizer JSON embedded in GGUF)
    2. tokenizer.ggml.* gpt2 BPE fields (tokens + merges)
    """
    path = Path(gguf_path)
    if not path.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    reader = gguf.GGUFReader(str(path))

    hf_json = _field_contents(reader, "tokenizer.huggingface.json")
    if isinstance(hf_json, str) and hf_json.strip():
        tok = _load_hf_tokenizer_from_gguf_json(hf_json, f"gguf:{gguf_path}:hf_json")
        if tok is not None:
            return tok
        # validate early for debugging when field exists but malformed
        json.loads(hf_json)

    tok = _load_gpt2_bpe_tokenizer_from_gguf(reader, f"gguf:{gguf_path}:gpt2_bpe")
    if tok is not None:
        return tok

    model = _field_contents(reader, "tokenizer.ggml.model", "unknown")
    raise RuntimeError(
        "GGUF tokenizer metadata is unsupported for direct runtime tokenization. "
        f"model={model!r}. Provide --tokenizer explicitly."
    )


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
import numpy as np


QWEN2_PRETOKENIZE_REGEX = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"""
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


class GGUFTokenizer:
    """Minimal tokenizer wrapper with encode/decode callables."""

    def __init__(
        self,
        encode_fn: Callable[[str], list[int]],
        decode_fn: Callable[[list[int]], str],
        source: str,
        special_token_ids: Optional[dict[str, int]] = None,
        stop_token_ids: Optional[list[int]] = None,
    ):
        self._encode = encode_fn
        self._decode = decode_fn
        self.source = source
        self.special_token_ids = special_token_ids or {}
        self.stop_token_ids = stop_token_ids or []

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


def _to_utf8_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return None
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8:
            try:
                return value.tobytes().decode("utf-8")
            except Exception:
                return None
        if value.ndim == 1 and np.issubdtype(value.dtype, np.integer):
            try:
                return bytes(int(x) & 0xFF for x in value.tolist()).decode("utf-8")
            except Exception:
                return None
    if isinstance(value, list):
        if value and all(isinstance(x, int) and 0 <= x <= 255 for x in value):
            try:
                return bytes(value).decode("utf-8")
            except Exception:
                return None
        if value and all(isinstance(x, str) for x in value):
            return "".join(value)
    return None


def _field_text(reader: gguf.GGUFReader, key: str) -> Optional[str]:
    field = reader.fields.get(key)
    if field is None:
        return None

    text = _to_utf8_text(_field_contents(reader, key, default=None))
    if text is not None and text.strip():
        return text

    parts = getattr(field, "parts", None)
    if parts is None:
        return None

    joined_bytes = bytearray()
    saw_bytes = False
    for part in parts:
        text = _to_utf8_text(part)
        if text is not None and text.strip():
            return text
        if isinstance(part, bytes):
            joined_bytes.extend(part)
            saw_bytes = True
        elif isinstance(part, np.ndarray) and part.dtype == np.uint8:
            joined_bytes.extend(part.tobytes())
            saw_bytes = True
        elif isinstance(part, list) and part and all(
            isinstance(x, int) and 0 <= x <= 255 for x in part
        ):
            joined_bytes.extend(part)
            saw_bytes = True

    if saw_bytes:
        try:
            decoded = bytes(joined_bytes).decode("utf-8")
            if decoded.strip():
                return decoded
        except Exception:
            pass
    return None


def _load_hf_tokenizer_from_gguf_json(
    hf_json: str,
    source: str,
    special_token_ids: Optional[dict[str, int]] = None,
    stop_token_ids: Optional[list[int]] = None,
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

    return GGUFTokenizer(
        _encode,
        _decode,
        source,
        special_token_ids=special_token_ids,
        stop_token_ids=stop_token_ids,
    )


def _load_gpt2_bpe_tokenizer_from_gguf(
    reader: gguf.GGUFReader,
    source: str,
    special_token_ids: Optional[dict[str, int]] = None,
    stop_token_ids: Optional[list[int]] = None,
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
        from tokenizers import AddedToken, Regex
        from tokenizers import normalizers, pre_tokenizers
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

    pre = _field_contents(reader, "tokenizer.ggml.pre", "")
    add_prefix_space = bool(_field_contents(reader, "tokenizer.ggml.add_space_prefix", False))

    try:
        bpe = BPE(vocab=vocab, merges=parsed_merges, unk_token=None, fuse_unk=False)
        tokenizer = Tokenizer(bpe)
        tokenizer.decoder = ByteLevelDecoder()

        # Qwen2/Qwen3 family uses a custom split + byte-level pretokenization.
        if str(pre).lower() == "qwen2":
            tokenizer.normalizer = normalizers.NFC()
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [
                    pre_tokenizers.Split(
                        Regex(QWEN2_PRETOKENIZE_REGEX),
                        behavior="isolated",
                        invert=False,
                    ),
                    pre_tokenizers.ByteLevel(
                        add_prefix_space=add_prefix_space,
                        use_regex=False,
                    ),
                ]
            )
        else:
            tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=add_prefix_space, use_regex=True)

        token_types = _field_contents(reader, "tokenizer.ggml.token_type")
        tokens_list = _field_contents(reader, "tokenizer.ggml.tokens")
        if isinstance(token_types, list) and isinstance(tokens_list, list):
            special_tokens: list[AddedToken] = []
            for idx, typ in enumerate(token_types):
                if int(typ) != int(gguf.TokenType.CONTROL):
                    continue
                if idx >= len(tokens_list):
                    continue
                tok = tokens_list[idx]
                if not isinstance(tok, str):
                    tok = str(tok)
                special_tokens.append(AddedToken(tok, special=True))
            if special_tokens:
                tokenizer.add_special_tokens(special_tokens)
    except Exception:
        return None

    def _encode(text: str) -> list[int]:
        return tokenizer.encode(text).ids

    def _decode(ids: list[int]) -> str:
        return tokenizer.decode(ids)

    return GGUFTokenizer(
        _encode,
        _decode,
        source,
        special_token_ids=special_token_ids,
        stop_token_ids=stop_token_ids,
    )


def infer_gguf_special_token_ids(gguf_path: str) -> dict[str, Any]:
    """Read special token IDs from GGUF tokenizer metadata."""
    path = Path(gguf_path)
    if not path.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    reader = gguf.GGUFReader(str(path))

    eos_id = _field_contents(reader, "tokenizer.ggml.eos_token_id")
    eot_id = _field_contents(reader, "tokenizer.ggml.eot_token_id")
    eom_id = _field_contents(reader, "tokenizer.ggml.eom_token_id")
    bos_id = _field_contents(reader, "tokenizer.ggml.bos_token_id")
    pad_id = _field_contents(reader, "tokenizer.ggml.padding_token_id")
    eos_ids = _field_contents(reader, "tokenizer.ggml.eos_token_ids", default=[])

    def _to_int(v: Any) -> Optional[int]:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return int(v)
        return None

    special: dict[str, int] = {}
    for name, value in (
        ("bos_token_id", bos_id),
        ("eos_token_id", eos_id),
        ("eot_token_id", eot_id),
        ("eom_token_id", eom_id),
        ("pad_token_id", pad_id),
    ):
        parsed = _to_int(value)
        if parsed is not None and parsed >= 0:
            special[name] = parsed

    stop_candidates: list[int] = []
    for key in ("eos_token_id", "eot_token_id", "eom_token_id"):
        if key in special:
            stop_candidates.append(special[key])

    if isinstance(eos_ids, list):
        for value in eos_ids:
            parsed = _to_int(value)
            if parsed is not None and parsed >= 0:
                stop_candidates.append(parsed)

    pad_token_id = special.get("pad_token_id")
    stop_ids: list[int] = []
    for tid in stop_candidates:
        if pad_token_id is not None and tid == pad_token_id:
            continue
        if tid not in stop_ids:
            stop_ids.append(tid)

    # Also stop on <|endoftext|> when present. Many Qwen/GPT2-BPE GGUFs use it as
    # an end marker.
    tokens = _field_contents(reader, "tokenizer.ggml.tokens")
    if isinstance(tokens, list):
        try:
            eot = tokens.index("<|endoftext|>")
            if eot not in stop_ids:
                stop_ids.append(int(eot))
        except ValueError:
            pass

    return {
        **special,
        "stop_token_ids": stop_ids,
    }


def load_tokenizer_from_gguf(
    gguf_path: str,
    *,
    strict_hf_json: bool = False,
) -> GGUFTokenizer:
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
    special_token_ids = infer_gguf_special_token_ids(gguf_path)
    stop_token_ids = special_token_ids.get("stop_token_ids", [])

    hf_json = _field_text(reader, "tokenizer.huggingface.json")
    if hf_json is not None and hf_json.strip():
        tok = _load_hf_tokenizer_from_gguf_json(
            hf_json,
            f"gguf:{gguf_path}:hf_json",
            special_token_ids=special_token_ids,
            stop_token_ids=stop_token_ids,
        )
        if tok is not None:
            return tok

        # Validate early for debugging when field exists but tokenizer init fails.
        try:
            json.loads(hf_json)
        except Exception as e:
            raise RuntimeError(
                "Found tokenizer.huggingface.json in GGUF, but failed to parse it as JSON"
            ) from e
        if strict_hf_json:
            raise RuntimeError(
                "Found tokenizer.huggingface.json in GGUF, but failed to build tokenizer "
                "from it. Refusing fallback in strict mode."
            )

    tok = _load_gpt2_bpe_tokenizer_from_gguf(
        reader,
        f"gguf:{gguf_path}:gpt2_bpe",
        special_token_ids=special_token_ids,
        stop_token_ids=stop_token_ids,
    )
    if tok is not None:
        return tok

    model = _field_contents(reader, "tokenizer.ggml.model", "unknown")
    raise RuntimeError(
        "GGUF tokenizer metadata is unsupported for direct runtime tokenization. "
        f"model={model!r}. Provide --tokenizer explicitly."
    )

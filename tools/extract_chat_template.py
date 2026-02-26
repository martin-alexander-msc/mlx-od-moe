#!/usr/bin/env python3
"""Extract tokenizer.chat_template from a GGUF model.

Usage:
  python tools/extract_chat_template.py /path/to/model.gguf
"""

from __future__ import annotations

import sys

import gguf


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: extract_chat_template.py /path/to/model.gguf")
        return 2

    path = sys.argv[1]
    r = gguf.GGUFReader(path)
    f = r.fields.get("tokenizer.chat_template")
    if f is None:
        print("(no tokenizer.chat_template field)")
        return 1

    v = f.contents()
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")

    print(v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Flask API Server for OD-MoE inference

Endpoints:
- POST /v1/completions - Streaming generation
- GET /health - Server health + cache stats
"""

from flask import Flask, request, jsonify, Response
import mlx.core as mx
import json
import time
from typing import Optional, Any
import re

from .model import KimiODMoEModel, ODMoEConfig
from .qwen3_next_od_model import Qwen3NextODMoEModel, Qwen3NextODConfig
from .gguf_expert_store import infer_gguf_moe_metadata
from .gguf_tokenizer import load_tokenizer_from_gguf, infer_gguf_special_token_ids
from .weight_loader import (
    load_base_weight_items,
    infer_config_overrides_from_base_shapes,
    inspect_base_weight_shapes,
    validate_expert_conversion,
    infer_num_local_experts,
    map_qwen3next_key_to_model_key,
)


app = Flask(__name__)
model: Optional[Any] = None
tokenizer = None
tokenizer_source: Optional[str] = None
stop_token_ids: list[int] = []


def _load_tokenizer(tokenizer_path: str):
    """Load tokenizer from model directory."""
    global tokenizer, tokenizer_source
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer_source = f"hf:{tokenizer_path}"
        print(f"Loaded tokenizer from {tokenizer_path}")
    except Exception as e:
        print(f"Warning: Could not load tokenizer ({e}). Using fallback encoding.")
        tokenizer = None
        tokenizer_source = None


def _load_gguf_tokenizer(gguf_path: str, strict_hf_json: bool = False):
    """Load tokenizer directly from GGUF metadata."""
    global tokenizer, tokenizer_source
    try:
        tokenizer = load_tokenizer_from_gguf(gguf_path, strict_hf_json=strict_hf_json)
        tokenizer_source = tokenizer.source
        print(f"Loaded tokenizer from GGUF metadata ({tokenizer.source})")
    except Exception as e:
        print(f"Warning: Could not load tokenizer from GGUF ({e}). Using fallback encoding.")
        tokenizer = None
        tokenizer_source = None


def _encode_to_token_ids(text: str) -> list[int]:
    """Tokenize text to token IDs."""
    if tokenizer is not None:
        return [int(i) for i in tokenizer.encode(text, return_tensors=None)]
    # Fallback: encode as UTF-8 byte values (limited but functional)
    return [int(i) for i in text.encode("utf-8")]


def _tokenize(text: str) -> mx.array:
    return mx.array([_encode_to_token_ids(text)])


def _detokenize(token_id: int) -> str:
    """Convert token ID back to text."""
    if tokenizer is not None:
        return tokenizer.decode([token_id])
    return chr(token_id) if 32 <= token_id < 127 else ""


def _decode_token_sequence(token_ids: list[int]) -> str:
    """Decode a full token sequence for tokenizer-consistent text rendering."""
    if tokenizer is not None:
        return tokenizer.decode(token_ids)
    chars: list[str] = []
    for token_id in token_ids:
        if 32 <= token_id < 127:
            chars.append(chr(token_id))
    return "".join(chars)


def _token_ids_hex(token_ids: list[int]) -> list[str]:
    return [f"0x{int(token_id):x}" for token_id in token_ids]


def _collect_model_weight_shapes(model: Any) -> dict[str, tuple[int, ...]]:
    """Flatten model.parameters() into {parameter_name: shape}."""
    shapes: dict[str, tuple[int, ...]] = {}

    def _walk(value, prefix: str = ""):
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else key
                _walk(child, child_prefix)
            return
        if isinstance(value, list):
            for idx, child in enumerate(value):
                child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
                _walk(child, child_prefix)
            return

        shape = tuple(int(dim) for dim in value.shape)
        shapes[prefix] = shape

    _walk(model.parameters())
    return shapes


def _is_qwen3next_schema(base_shapes: dict[str, tuple[int, ...]]) -> bool:
    return any(re.fullmatch(r"blk\.\d+\.attn_qkv\.weight", k) for k in base_shapes)


def _validate_qwen3next_base_shapes(base_shapes: dict[str, tuple[int, ...]]) -> None:
    required = [
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_ba.weight",
        "blk.0.ssm_a",
        "blk.0.ssm_dt",
        "blk.0.ssm_out.weight",
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_up_shexp.weight",
        "blk.0.ffn_down_shexp.weight",
    ]
    missing = [k for k in required if k not in base_shapes]
    if missing:
        raise RuntimeError(
            "Qwen3Next base conversion is missing required tensors "
            f"{missing}. Re-run base extraction with the updated converter "
            "(convert/gguf_to_od_moe.py --base-only)."
        )


def _preprocess_qwen3next_source_tensors(
    source_dict: dict[str, Any],
    linear_num_key_heads: int | None = None,
) -> dict[str, Any]:
    """
    Build fused Qwen3Next linear-attention qkvz projection tensor:
    blk.N.attn_qkv.weight + blk.N.attn_gate.weight -> blk.N.attn_qkvz.weight

    The fused tensor must be interleaved per key-head as [q, k, v, z] chunks,
    matching Qwen3NextGatedDeltaNet.fix_query_key_value_ordering().
    """
    processed = dict(source_dict)
    pattern = re.compile(r"blk\.(\d+)\.attn_qkv\.weight$")

    for key in list(source_dict.keys()):
        m = pattern.match(key)
        if not m:
            continue
        layer = m.group(1)
        gate_key = f"blk.{layer}.attn_gate.weight"
        if gate_key not in source_dict:
            continue

        qkv = source_dict[key]
        gate = source_dict[gate_key]
        if len(qkv.shape) != 2 or len(gate.shape) != 2 or qkv.shape[0] != gate.shape[0]:
            raise RuntimeError(
                f"Cannot fuse Qwen3Next qkv/gate for layer {layer}: "
                f"qkv.shape={qkv.shape}, gate.shape={gate.shape}"
            )

        if linear_num_key_heads is not None:
            if linear_num_key_heads <= 0:
                raise RuntimeError(f"Invalid linear_num_key_heads={linear_num_key_heads}")
            if (qkv.shape[1] % linear_num_key_heads) != 0 or (gate.shape[1] % linear_num_key_heads) != 0:
                raise RuntimeError(
                    f"Cannot interleave Qwen3Next qkv/gate for layer {layer}: "
                    f"qkv.shape[1]={qkv.shape[1]}, gate.shape[1]={gate.shape[1]}, "
                    f"linear_num_key_heads={linear_num_key_heads}"
                )
            qkv_chunk = qkv.shape[1] // linear_num_key_heads
            gate_chunk = gate.shape[1] // linear_num_key_heads
            qkv_by_head = qkv.reshape(qkv.shape[0], linear_num_key_heads, qkv_chunk)
            gate_by_head = gate.reshape(gate.shape[0], linear_num_key_heads, gate_chunk)
            fused = mx.concatenate([qkv_by_head, gate_by_head], axis=-1)
            processed[f"blk.{layer}.attn_qkvz.weight"] = fused.reshape(qkv.shape[0], -1)
        else:
            # Conservative fallback for older flows where head count is unavailable.
            processed[f"blk.{layer}.attn_qkvz.weight"] = mx.concatenate([qkv, gate], axis=1)
        processed.pop(key, None)
        processed.pop(gate_key, None)

    # Qwen3Next GGUF norm tensors may be exported in either:
    # - zero-centered offset form (requires +1.0 shift), or
    # - already shifted runtime form (must NOT be shifted again).
    norm_patterns = (
        re.compile(r"blk\.\d+\.attn_norm\.weight$"),
        re.compile(r"blk\.\d+\.post_attention_norm\.weight$"),
        re.compile(r"blk\.\d+\.attn_q_norm\.weight$"),
        re.compile(r"blk\.\d+\.attn_k_norm\.weight$"),
    )
    norm_keys: list[str] = []
    norm_stats: list[tuple[float, float, float]] = []
    for key, tensor in processed.items():
        if key != "output_norm.weight" and not any(p.fullmatch(key) for p in norm_patterns):
            continue
        if not (hasattr(tensor, "ndim") and int(tensor.ndim) == 1):
            continue
        t_min = float(mx.min(tensor).item())
        t_max = float(mx.max(tensor).item())
        t_mean = float(mx.mean(tensor).item())
        norm_keys.append(key)
        norm_stats.append((t_min, t_max, t_mean))

    apply_norm_shift = False
    if norm_stats:
        global_min = min(s[0] for s in norm_stats)
        global_max = max(s[1] for s in norm_stats)
        global_mean = sum(s[2] for s in norm_stats) / len(norm_stats)
        # Shift only when norms look like zero-centered offsets.
        apply_norm_shift = (global_min < -0.05) or (global_mean < 0.5)
        print(
            "Qwen3Next norm preprocessing: "
            f"count={len(norm_stats)}, min={global_min:.4f}, max={global_max:.4f}, "
            f"mean={global_mean:.4f}, shift_applied={apply_norm_shift}"
        )

    if apply_norm_shift:
        for key in norm_keys:
            processed[key] = processed[key] + 1.0

    return processed


def initialize_model(
    expert_dir: Optional[str],
    base_weights: str,
    gguf_experts: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    cache_size_gb: int = 48,
):
    """Initialize model with OD-MoE."""
    global model, stop_token_ids

    print("Initializing OD-MoE model...")
    if gguf_experts is None:
        if not expert_dir:
            raise RuntimeError("expert_dir is required unless --gguf-experts is provided")
        try:
            validate_expert_conversion(expert_dir)
        except Exception as e:
            raise RuntimeError(f"Invalid expert conversion in {expert_dir}: {e}")

    try:
        base_shapes = inspect_base_weight_shapes(base_weights)
        overrides = infer_config_overrides_from_base_shapes(base_weights)
        gguf_meta = infer_gguf_moe_metadata(gguf_experts) if gguf_experts else {}
        gguf_tok_meta = infer_gguf_special_token_ids(gguf_experts) if gguf_experts else {}

        if gguf_experts:
            for key in (
                "num_local_experts",
                "num_hidden_layers",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
                "rope_theta",
            ):
                if key in gguf_meta:
                    overrides[key] = gguf_meta[key]
            if "eos_token_id" in gguf_tok_meta:
                overrides["eos_token_id"] = int(gguf_tok_meta["eos_token_id"])
        else:
            inferred_experts = infer_num_local_experts(expert_dir)
            if inferred_experts is not None:
                overrides["num_local_experts"] = inferred_experts
    except Exception as e:
        raise RuntimeError(f"Failed to infer model config from converted weights: {e}")

    qwen3next_mode = _is_qwen3next_schema(base_shapes)
    map_key_fn = None
    preprocess_fn = None

    if qwen3next_mode:
        try:
            _validate_qwen3next_base_shapes(base_shapes)
        except Exception as e:
            raise RuntimeError(f"Invalid Qwen3Next base conversion: {e}")

        q_overrides = dict(overrides)
        for key in (
            "full_attention_interval",
            "norm_topk_prob",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "partial_rotary_factor",
            "max_position_embeddings",
            "rope_theta",
        ):
            if key in gguf_meta:
                q_overrides[key] = gguf_meta[key]
        if "eos_token_id" in gguf_tok_meta:
            q_overrides["eos_token_id"] = int(gguf_tok_meta["eos_token_id"])

        # Conservative defaults for Qwen3Next-Coder-Next family.
        q_overrides.setdefault("full_attention_interval", 4)
        q_overrides.setdefault("norm_topk_prob", True)
        q_overrides.setdefault("num_experts_per_tok", 10)
        q_overrides.setdefault("moe_intermediate_size", 512)
        q_overrides.setdefault("shared_expert_intermediate_size", 512)
        q_overrides.setdefault("num_attention_heads", 16)
        q_overrides.setdefault("num_key_value_heads", 2)
        q_overrides.setdefault("head_dim", 256)
        q_overrides.setdefault("partial_rotary_factor", 0.25)
        q_overrides.setdefault("linear_num_key_heads", 16)
        q_overrides.setdefault("linear_num_value_heads", 32)
        q_overrides.setdefault("linear_key_head_dim", 128)
        q_overrides.setdefault("linear_value_head_dim", 128)
        q_overrides.setdefault("linear_conv_kernel_dim", 4)

        print(f"Inferred Qwen3Next config overrides: {q_overrides}")
        config = Qwen3NextODConfig(**q_overrides)
        print(
            "Resolved Qwen3Next config: "
            f"vocab={config.vocab_size}, hidden={config.hidden_size}, "
            f"layers={config.num_hidden_layers}, experts/layer={config.num_local_experts}, "
            f"full_attn_interval={config.full_attention_interval}"
        )
        model = Qwen3NextODMoEModel(config)
        map_key_fn = map_qwen3next_key_to_model_key
        preprocess_fn = lambda source: _preprocess_qwen3next_source_tensors(
            source,
            linear_num_key_heads=config.linear_num_key_heads,
        )
    else:
        # Legacy GQA-only ODMoE path.
        print(f"Inferred config overrides: {overrides}")
        config = ODMoEConfig(**overrides)
        print(
            "Resolved config: "
            f"vocab={config.vocab_size}, hidden={config.hidden_size}, "
            f"layers={config.num_hidden_layers}, experts/layer={config.num_local_experts}"
        )
        model = KimiODMoEModel(config)

    # Setup OD-MoE
    try:
        model.setup_od_moe(
            expert_dir=expert_dir,
            gguf_expert_path=gguf_experts,
            cache_size_gb=cache_size_gb,
        )
    except FileNotFoundError:
        if gguf_experts:
            raise FileNotFoundError(f"GGUF expert source not found: {gguf_experts}")
        raise FileNotFoundError(f"Expert directory not found: {expert_dir}")
    except Exception as e:
        source = gguf_experts if gguf_experts else expert_dir
        raise RuntimeError(f"Failed to setup OD-MoE from {source}: {e}")

    # Load base weights (single safetensors file OR converted base_model directory)
    print(f"Loading base weights from {base_weights}...")
    try:
        expected_shapes = _collect_model_weight_shapes(model)
        if qwen3next_mode:
            print("Using Qwen3Next base-weight mapping and preprocessing")
        if map_key_fn is None:
            weight_items, load_stats = load_base_weight_items(
                base_weights,
                mx.load,
                expected_shapes,
                preprocess_fn=preprocess_fn,
            )
        else:
            weight_items, load_stats = load_base_weight_items(
                base_weights,
                mx.load,
                expected_shapes,
                map_key_fn=map_key_fn,
                preprocess_fn=preprocess_fn,
            )
        model.load_weights(weight_items)
        print(
            f"Loaded {load_stats['loaded']} base tensors "
            f"(skipped {load_stats['skipped']} / {load_stats['source_tensors']})"
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Base weights not found: {base_weights}")
    except Exception as e:
        raise RuntimeError(f"Failed to load base weights from {base_weights}: {e}")

    # Load tokenizer
    if tokenizer_path:
        _load_tokenizer(tokenizer_path)
    elif gguf_experts:
        # For Qwen3Next, reject silent fallback when an embedded HF tokenizer JSON
        # is present but malformed/unusable.
        _load_gguf_tokenizer(gguf_experts, strict_hf_json=qwen3next_mode)

    if gguf_experts:
        stop_token_ids = [int(t) for t in gguf_tok_meta.get("stop_token_ids", [])]
    else:
        stop_token_ids = []
    if not stop_token_ids and model is not None and hasattr(model, "config"):
        stop_token_ids = [int(model.config.eos_token_id)]
    print(f"Resolved stop token IDs: {stop_token_ids}")

    print("Model initialized")


@app.route("/v1/completions", methods=["POST"])
def completions():
    """
    OpenAI-compatible completions endpoint with streaming.

    Request:
    {
        "prompt": "Explain quantum computing",
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": true
    }

    Response (streaming):
    data: {"token": "Quantum"}
    data: {"token": " computing"}
    ...
    data: {"done": true, "stats": {...}}
    """
    if not model:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.6)
    top_p = data.get("top_p", 0.9)
    stream = data.get("stream", True)
    debug_tokens = bool(data.get("debug_tokens", False))
    echo_prompt = bool(data.get("echo_prompt", False))

    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    prompt_token_ids = _encode_to_token_ids(prompt)
    input_ids = mx.array([prompt_token_ids])

    start_time = time.time()
    tokens_generated = 0

    def generate_stream():
        nonlocal tokens_generated
        generated_ids: list[int] = []
        emitted_text = ""
        if echo_prompt:
            emitted_text = prompt
            yield f"data: {json.dumps({'token': prompt, 'token_id': None})}\n\n"

        for token_id in model.generate(
            input_ids,
            max_tokens,
            temperature,
            top_p,
            stop_token_ids=stop_token_ids,
        ):
            tokens_generated += 1
            generated_ids.append(int(token_id))

            # Decode cumulatively and emit only the new suffix. This avoids
            # per-token decode artifacts for byte-level BPE tokenizers.
            full_text = _decode_token_sequence(generated_ids)
            if full_text.startswith(emitted_text):
                token_text = full_text[len(emitted_text) :]
            else:
                # Fallback for rare non-prefix decode behavior.
                token_text = full_text
            emitted_text = full_text

            yield f"data: {json.dumps({'token': token_text, 'token_id': token_id})}\n\n"

        # Send completion stats
        total_time = time.time() - start_time
        stats = model.expert_store.get_stats() if model.expert_store else {}
        stop_token_id = int(generated_ids[-1]) if generated_ids else None
        stop_reason = (
            "stop_token" if stop_token_id is not None and stop_token_id in stop_token_ids else "max_tokens"
        )
        done_payload = {
            "done": True,
            "tokens_generated": tokens_generated,
            "time_seconds": round(total_time, 3),
            "tokens_per_second": round(tokens_generated / total_time, 1) if total_time > 0 else 0,
            "cache_stats": stats,
            "stop_reason": stop_reason,
            "stop_token_id": stop_token_id,
        }
        if debug_tokens:
            done_payload.update(
                {
                    "prompt_token_ids": prompt_token_ids,
                    "prompt_token_ids_hex": _token_ids_hex(prompt_token_ids),
                    "generated_token_ids": [int(t) for t in generated_ids],
                    "generated_token_ids_hex": _token_ids_hex(generated_ids),
                    "tokenizer_source": tokenizer_source,
                }
            )

        yield f"data: {json.dumps(done_payload)}\n\n"

    if stream:
        return Response(generate_stream(), mimetype="text/event-stream")
    else:
        token_ids: list[int] = []
        for token_id in model.generate(
            input_ids,
            max_tokens,
            temperature,
            top_p,
            stop_token_ids=stop_token_ids,
        ):
            token_ids.append(int(token_id))
        generated_text = _decode_token_sequence(token_ids)
        completion = prompt + generated_text if echo_prompt else generated_text
        stop_token_id = int(token_ids[-1]) if token_ids else None
        stop_reason = (
            "stop_token" if stop_token_id is not None and stop_token_id in stop_token_ids else "max_tokens"
        )
        payload: dict[str, Any] = {
            "completion": completion,
            "tokens_generated": len(token_ids),
            "stop_reason": stop_reason,
            "stop_token_id": stop_token_id,
        }
        if debug_tokens:
            payload.update(
                {
                    "prompt_token_ids": prompt_token_ids,
                    "prompt_token_ids_hex": _token_ids_hex(prompt_token_ids),
                    "generated_token_ids": token_ids,
                    "generated_token_ids_hex": _token_ids_hex(token_ids),
                    "tokenizer_source": tokenizer_source,
                    "generated_text_only": generated_text,
                }
            )
        return jsonify(payload)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    stats = {}
    if model and model.expert_store:
        stats = model.expert_store.get_stats()

    return jsonify(
        {
            "status": "healthy" if model else "initializing",
            "model_loaded": model is not None,
            "tokenizer_loaded": tokenizer is not None,
            "tokenizer_source": tokenizer_source,
            "stop_token_ids": stop_token_ids,
            "expert_cache_stats": stats,
        }
    )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="mlx-od-moe server")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--expert-dir", help="Converted expert directory")
    source_group.add_argument(
        "--gguf-experts",
        help="Use GGUF file directly as expert source (avoids expert conversion output)",
    )
    parser.add_argument("--base-weights", required=True, help="Base model weights")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path or HF model ID")
    parser.add_argument("--cache-size-gb", type=int, default=48, help="Cache size in GB")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")

    args = parser.parse_args()

    initialize_model(
        expert_dir=args.expert_dir,
        base_weights=args.base_weights,
        gguf_experts=args.gguf_experts,
        tokenizer_path=args.tokenizer,
        cache_size_gb=args.cache_size_gb,
    )

    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

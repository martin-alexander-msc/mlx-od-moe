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
from typing import Optional

from .model import KimiODMoEModel, ODMoEConfig
from .gguf_expert_store import infer_gguf_moe_metadata
from .weight_loader import (
    load_base_weight_items,
    infer_config_overrides_from_base_shapes,
    inspect_base_weight_shapes,
    validate_expert_conversion,
    infer_num_local_experts,
)


app = Flask(__name__)
model: Optional[KimiODMoEModel] = None
tokenizer = None


def _load_tokenizer(tokenizer_path: str):
    """Load tokenizer from model directory."""
    global tokenizer
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path}")
    except Exception as e:
        print(f"Warning: Could not load tokenizer ({e}). Using fallback encoding.")
        tokenizer = None


def _tokenize(text: str) -> mx.array:
    """Tokenize text to token IDs."""
    if tokenizer is not None:
        ids = tokenizer.encode(text, return_tensors=None)
        return mx.array([ids])
    # Fallback: encode as UTF-8 byte values (limited but functional)
    ids = list(text.encode("utf-8"))
    return mx.array([ids])


def _detokenize(token_id: int) -> str:
    """Convert token ID back to text."""
    if tokenizer is not None:
        return tokenizer.decode([token_id])
    return chr(token_id) if 32 <= token_id < 127 else ""


def _collect_model_weight_shapes(model: KimiODMoEModel) -> dict[str, tuple[int, ...]]:
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


def initialize_model(
    expert_dir: Optional[str],
    base_weights: str,
    gguf_experts: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    cache_size_gb: int = 48,
):
    """Initialize model with OD-MoE."""
    global model

    print("Initializing OD-MoE model...")
    if gguf_experts is None:
        if not expert_dir:
            raise RuntimeError("expert_dir is required unless --gguf-experts is provided")
        try:
            validate_expert_conversion(expert_dir)
        except Exception as e:
            raise RuntimeError(f"Invalid expert conversion in {expert_dir}: {e}")

    try:
        overrides = infer_config_overrides_from_base_shapes(base_weights)
        if gguf_experts:
            gguf_meta = infer_gguf_moe_metadata(gguf_experts)
            for key in (
                "num_local_experts",
                "num_hidden_layers",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
            ):
                if key in gguf_meta:
                    overrides[key] = gguf_meta[key]
        else:
            inferred_experts = infer_num_local_experts(expert_dir)
            if inferred_experts is not None:
                overrides["num_local_experts"] = inferred_experts
    except Exception as e:
        raise RuntimeError(f"Failed to infer model config from converted weights: {e}")

    # Reconcile attention dimensions with actual converted base tensor shapes.
    try:
        shapes = inspect_base_weight_shapes(base_weights)
        q_shape = shapes.get("blk.0.attn_q.weight") or shapes.get("layers.0.attention.q_proj.weight")
        k_shape = shapes.get("blk.0.attn_k.weight") or shapes.get("layers.0.attention.k_proj.weight")
        v_shape = shapes.get("blk.0.attn_v.weight") or shapes.get("layers.0.attention.v_proj.weight")

        hidden = int(overrides["hidden_size"])

        def _proj_out(shape):
            if not shape or len(shape) != 2:
                return None
            a, b = int(shape[0]), int(shape[1])
            if a == hidden and b != hidden:
                return b
            if b == hidden and a != hidden:
                return a
            return max(a, b)

        q_out = _proj_out(q_shape)
        k_out = _proj_out(k_shape)
        v_out = _proj_out(v_shape)
        kv_proj_out = k_out or v_out

        if q_out and kv_proj_out:
            head_dim = int(overrides["head_dim"]) if "head_dim" in overrides else None
            q_heads = (
                int(overrides["num_attention_heads"])
                if "num_attention_heads" in overrides
                else None
            )
            kv_heads = (
                int(overrides["num_key_value_heads"])
                if "num_key_value_heads" in overrides
                else None
            )

            # If rope/head_dim metadata is available, projection tensor widths are the source of truth.
            # Some GGUFs expose head counts that do not match projection packing for this runtime.
            if head_dim is not None and q_out % head_dim == 0 and kv_proj_out % head_dim == 0:
                q_heads = q_out // head_dim
                kv_heads = kv_proj_out // head_dim
                overrides["num_attention_heads"] = q_heads
                overrides["num_key_value_heads"] = kv_heads

            if q_heads is not None and kv_heads is not None:
                if q_out % q_heads != 0 or kv_proj_out % kv_heads != 0:
                    raise RuntimeError(
                        f"Head counts do not match base tensor projection sizes: "
                        f"q_out={q_out}, kv_proj_out={kv_proj_out}, num_attention_heads={q_heads}, "
                        f"num_key_value_heads={kv_heads}"
                    )
                q_head_dim = q_out // q_heads
                kv_head_dim = kv_proj_out // kv_heads
                if q_head_dim != kv_head_dim:
                    raise RuntimeError(
                        f"Inconsistent inferred head dims from base tensors: "
                        f"q_head_dim={q_head_dim}, kv_head_dim={kv_head_dim}"
                    )
                overrides["head_dim"] = q_head_dim
                if q_heads % kv_heads != 0:
                    raise RuntimeError(
                        f"Invalid attention grouping inferred from base tensors: "
                        f"num_attention_heads={q_heads}, num_key_value_heads={kv_heads}"
                    )
                if v_out:
                    if v_out % kv_heads != 0:
                        raise RuntimeError(
                            f"Value projection width is not divisible by kv heads: "
                            f"v_out={v_out}, num_key_value_heads={kv_heads}"
                        )
                    overrides["value_head_dim"] = v_out // kv_heads
                print(
                    "Resolved attention projection dims: "
                    f"q_out={q_out}, k_out={k_out}, v_out={v_out}, "
                    f"q_heads={q_heads}, kv_heads={kv_heads}, "
                    f"head_dim={overrides.get('head_dim')}, "
                    f"value_head_dim={overrides.get('value_head_dim')}"
                )
            elif head_dim is None:
                raise RuntimeError(
                    "Could not determine attention head configuration. "
                    "GGUF metadata must provide head_count/head_count_kv or rope.dimension_count."
                )
    except Exception as e:
        raise RuntimeError(f"Failed to reconcile attention dimensions: {e}")

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
        weight_items, load_stats = load_base_weight_items(base_weights, mx.load, expected_shapes)
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

    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    input_ids = _tokenize(prompt)

    start_time = time.time()
    tokens_generated = 0

    def generate_stream():
        nonlocal tokens_generated

        for token_id in model.generate(input_ids, max_tokens, temperature, top_p):
            tokens_generated += 1
            token_text = _detokenize(token_id)

            yield f"data: {json.dumps({'token': token_text, 'token_id': token_id})}\n\n"

        # Send completion stats
        total_time = time.time() - start_time
        stats = model.expert_store.get_stats() if model.expert_store else {}

        yield f"data: {json.dumps({'done': True, 'tokens_generated': tokens_generated, 'time_seconds': round(total_time, 3), 'tokens_per_second': round(tokens_generated / total_time, 1) if total_time > 0 else 0, 'cache_stats': stats})}\n\n"

    if stream:
        return Response(generate_stream(), mimetype="text/event-stream")
    else:
        tokens = []
        for token_id in model.generate(input_ids, max_tokens, temperature, top_p):
            tokens.append(_detokenize(token_id))
        return jsonify(
            {"completion": "".join(tokens), "tokens_generated": len(tokens)}
        )


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

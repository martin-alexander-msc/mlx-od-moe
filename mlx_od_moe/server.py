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

from .model import KimiODMoEModel, KimiODMoEConfig


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


def initialize_model(
    expert_dir: str,
    base_weights: str,
    tokenizer_path: Optional[str] = None,
    cache_size_gb: int = 48,
):
    """Initialize model with OD-MoE."""
    global model

    print("Initializing Kimi OD-MoE model...")
    config = KimiODMoEConfig()
    model = KimiODMoEModel(config)

    # Load base weights
    print(f"Loading base weights from {base_weights}...")
    try:
        weights = mx.load(base_weights)
        model.load_weights(list(weights.items()))
    except FileNotFoundError:
        raise FileNotFoundError(f"Base weights not found: {base_weights}")
    except Exception as e:
        raise RuntimeError(f"Failed to load base weights from {base_weights}: {e}")

    # Setup OD-MoE
    try:
        model.setup_od_moe(expert_dir, cache_size_gb=cache_size_gb)
    except FileNotFoundError:
        raise FileNotFoundError(f"Expert directory not found: {expert_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to setup OD-MoE from {expert_dir}: {e}")

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
    parser.add_argument("--expert-dir", required=True, help="Expert directory")
    parser.add_argument("--base-weights", required=True, help="Base model weights")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path or HF model ID")
    parser.add_argument("--cache-size-gb", type=int, default=48, help="Cache size in GB")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")

    args = parser.parse_args()

    initialize_model(args.expert_dir, args.base_weights, args.tokenizer, args.cache_size_gb)

    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

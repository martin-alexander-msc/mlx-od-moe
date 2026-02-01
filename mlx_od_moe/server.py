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


def initialize_model(expert_dir: str, base_weights: str, cache_size_gb: int = 48):
    """Initialize model with OD-MoE"""
    global model
    
    print("Initializing Kimi OD-MoE model...")
    config = KimiODMoEConfig()
    model = KimiODMoEModel(config)
    
    # TODO: Load base weights
    # base_weights_dict = mx.load(base_weights)
    # model.load_weights(base_weights_dict)
    
    # Setup OD-MoE
    model.setup_od_moe(expert_dir, cache_size_gb=cache_size_gb)
    
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
    stream = data.get("stream", True)
    
    if not prompt:
        return jsonify({"error": "prompt required"}), 400
    
    # TODO: Tokenize prompt
    # For now, dummy input
    input_ids = mx.array([[1, 2, 3]])
    
    start_time = time.time()
    tokens_generated = 0
    
    def generate_stream():
        nonlocal tokens_generated
        
        for token_id in model.generate(input_ids, max_tokens, temperature):
            tokens_generated += 1
            # TODO: Detokenize
            token_text = f"token_{token_id}"
            
            yield f"data: {json.dumps({'token': token_text})}\n\n"
        
        # Send completion stats
        total_time = time.time() - start_time
        stats = model.expert_store.get_stats() if model.expert_store else {}
        
        yield f"data: {json.dumps({
            'done': True,
            'tokens_generated': tokens_generated,
            'time_seconds': total_time,
            'tokens_per_second': tokens_generated / total_time if total_time > 0 else 0,
            'cache_stats': stats
        })}\n\n"
    
    if stream:
        return Response(generate_stream(), mimetype="text/event-stream")
    else:
        # Non-streaming: collect all tokens
        tokens = list(model.generate(input_ids, max_tokens, temperature))
        return jsonify({
            "completion": " ".join(f"token_{t}" for t in tokens),
            "tokens_generated": len(tokens)
        })


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    
    Response:
    {
        "status": "healthy",
        "model_loaded": true,
        "expert_cache_stats": {
            "cache_hits": 1250,
            "cache_misses": 87,
            "hit_rate": 0.935,
            "working_set_experts": 224
        }
    }
    """
    stats = {}
    if model and model.expert_store:
        stats = model.expert_store.get_stats()
    
    return jsonify({
        "status": "healthy" if model else "initializing",
        "model_loaded": model is not None,
        "expert_cache_stats": stats
    })


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="mlx-od-moe server")
    parser.add_argument("--expert-dir", required=True, help="Expert directory")
    parser.add_argument("--base-weights", required=True, help="Base model weights")
    parser.add_argument("--cache-size-gb", type=int, default=48, help="Cache size in GB")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    
    args = parser.parse_args()
    
    # Initialize model
    initialize_model(args.expert_dir, args.base_weights, args.cache_size_gb)
    
    # Start server
    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

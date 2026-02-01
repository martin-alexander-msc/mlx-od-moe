"""
Shadow Model - Lightweight expert predictor for prefetch

Predicts which experts will be needed 4 layers ahead based on current hidden states.
Enables overlapping SSD I/O with computation to hide latency.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Dict, Optional
import threading


class ExpertPredictor(nn.Module):
    """
    Lightweight MLP that predicts which experts will be needed for upcoming layers.

    Input: Hidden state from layer t (before MoE)
    Output: Top-k expert indices for layers t+1, t+2, t+3, t+4

    Target latency: <1ms on M4 Max
    """

    def __init__(
        self,
        hidden_dim: int = 4096,       # Kimi-K2.5 hidden size
        num_experts: int = 384,        # Total experts per layer
        num_layers_ahead: int = 4,     # Lookahead window
        predictor_dim: int = 1024      # Compressed representation
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_layers_ahead = num_layers_ahead

        # Encoder: compress hidden states to smaller representation
        # Removed Dropout for faster inference
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, predictor_dim),
            nn.LayerNorm(predictor_dim),
            nn.SiLU()
        )

        # Separate prediction head for each lookahead step
        # Simplified to single linear layer for speed
        self.prediction_heads = [
            nn.Linear(predictor_dim, num_experts)
            for _ in range(num_layers_ahead)
        ]

    def __call__(self, hidden_states: mx.array) -> List[mx.array]:
        """
        Predict expert indices for upcoming layers.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            List of 4 tensors, each (batch, top_k) of expert indices
        """
        # Pool over sequence length (mean pooling)
        # Shape: (batch, hidden_dim)
        pooled = mx.mean(hidden_states, axis=1)

        # Encode to compressed representation
        # Shape: (batch, predictor_dim)
        encoded = self.encoder(pooled)

        # Predict for each lookahead step
        predictions = []
        for head in self.prediction_heads:
            # Get logits over all experts
            # Shape: (batch, num_experts)
            logits = head(encoded)

            # Select top-k experts
            # For batch size 1 (inference), just use argmax
            if logits.shape[0] == 1:
                # Shape: (top_k,)
                top_k_indices = mx.argsort(logits[0], axis=-1)[-8:]
            else:
                # Shape: (batch, top_k)
                top_k_indices = mx.argsort(logits, axis=-1)[:, -8:]

            predictions.append(top_k_indices)

        return predictions

    def save_weights(self, path):
        """Save model weights to safetensors file"""
        from safetensors.mlx import save_file
        from pathlib import Path

        path = Path(path)
        params = {}

        # Encoder parameters
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                params[f'encoder_{i}_weight'] = layer.weight
                params[f'encoder_{i}_bias'] = layer.bias
            elif isinstance(layer, nn.LayerNorm):
                params[f'encoder_{i}_weight'] = layer.weight
                params[f'encoder_{i}_bias'] = layer.bias

        # Prediction head parameters
        for head_idx, head in enumerate(self.prediction_heads):
            params[f'head_{head_idx}_weight'] = head.weight
            params[f'head_{head_idx}_bias'] = head.bias

        save_file(params, str(path))

    def load_weights(self, path):
        """Load model weights from safetensors file"""
        from safetensors.mlx import load_file
        from pathlib import Path

        path = Path(path)
        params = load_file(str(path))

        # Load encoder parameters
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                layer.weight = params[f'encoder_{i}_weight']
                layer.bias = params[f'encoder_{i}_bias']
            elif isinstance(layer, nn.LayerNorm):
                layer.weight = params[f'encoder_{i}_weight']
                layer.bias = params[f'encoder_{i}_bias']

        # Load prediction head parameters
        for head_idx, head in enumerate(self.prediction_heads):
            head.weight = params[f'head_{head_idx}_weight']
            head.bias = params[f'head_{head_idx}_bias']


class ShadowRunner:
    """
    Manages shadow model execution alongside main model.

    Runs predictions and queues results for OD-MoE layers to consume.
    Uses MLX async execution for non-blocking predictions.
    """

    def __init__(self, predictor_path: Optional[str] = None):
        self.predictor = ExpertPredictor()

        if predictor_path:
            from pathlib import Path
            self.predictor.load_weights(Path(predictor_path))

        # Prediction queue: stores recent predictions
        self.prediction_queue: List[Dict] = []
        self.lock = threading.Lock()

    def predict_async(
        self,
        hidden_states: mx.array,
        current_layer: int
    ) -> List[mx.array]:
        """
        Queue prediction for async processing.

        MLX uses lazy computation - builds graph without executing.
        mx.async_eval schedules async execution in background.

        Returns immediately with predictions (computed in background).
        """
        # Build computation graph (not executed yet)
        predictions = self.predictor(hidden_states)

        # Schedule async execution (non-blocking)
        mx.async_eval(predictions)

        # Store in queue for retrieval
        with self.lock:
            self.prediction_queue.append({
                'layer': current_layer,
                'predictions': predictions
            })

            # Keep queue bounded (last 10 predictions)
            if len(self.prediction_queue) > 10:
                self.prediction_queue.pop(0)

        return predictions

    def get_predictions_for_layer(self, layer_idx: int) -> List[int]:
        """
        Retrieve predicted experts for a specific layer.

        Called by OD-MoE layer before computation to trigger prefetch.

        Args:
            layer_idx: Layer that needs expert predictions

        Returns:
            List of expert indices (flattened and deduplicated)
        """
        with self.lock:
            # Find predictions made from appropriate layer
            # Predictions from layer L cover L+1, L+2, L+3, L+4
            for item in self.prediction_queue:
                source_layer = item['layer']
                lookahead = layer_idx - source_layer - 1

                # Check if this prediction covers requested layer
                if 0 <= lookahead < 4:
                    predictions = item['predictions']

                    # Ensure computation is complete
                    mx.eval(predictions)

                    # Extract experts for this lookahead step
                    pred = predictions[lookahead]

                    # Flatten and deduplicate
                    if pred.ndim == 1:
                        experts = pred.tolist()
                    else:
                        experts = pred.flatten().tolist()

                    return list(set(map(int, experts)))

        # Default fallback: first 8 experts if no prediction available
        return list(range(8))

    def clear_queue(self):
        """Clear prediction queue (useful for testing)"""
        with self.lock:
            self.prediction_queue.clear()

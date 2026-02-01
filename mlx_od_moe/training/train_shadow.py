"""
Train Shadow Model predictor.

Learns to predict which experts will be needed based on hidden states.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from mlx_od_moe.shadow_model import ExpertPredictor


def compute_top_k_accuracy(
    predictions: mx.array,
    targets: mx.array,
    k: int
) -> float:
    """
    Compute top-k accuracy: what % of target experts are in predicted top-k.

    Args:
        predictions: (batch, num_experts) logits
        targets: (batch, 8) ground truth expert indices

    Returns:
        Accuracy in [0, 1]
    """
    # Get top-k predictions
    top_k_pred = mx.argsort(predictions, axis=-1)[:, -k:]

    # For each sample, check how many targets are in predictions
    batch_size = targets.shape[0]
    matches = 0
    total = 0

    for i in range(batch_size):
        target_set = set(targets[i].tolist())
        pred_set = set(top_k_pred[i].tolist())
        matches += len(target_set & pred_set)
        total += len(target_set)

    return matches / total if total > 0 else 0.0


def train_shadow_model(
    training_data_path: Path,
    output_path: Path,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    layer_idx: int = 0
) -> Dict:
    """
    Train shadow model predictor.

    Args:
        training_data_path: Path to .npz file with training data
        output_path: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        layer_idx: Which layer to train predictor for

    Returns:
        Dictionary of training metrics
    """
    # Load training data
    print(f"Loading training data from {training_data_path}...")
    data = np.load(training_data_path)

    hidden_states = mx.array(data[f'hidden_states_{layer_idx}'])
    expert_choices = mx.array(data[f'expert_choices_{layer_idx}'])

    num_samples = hidden_states.shape[0]
    print(f"Loaded {num_samples} training samples")

    # Initialize model and optimizer
    model = ExpertPredictor()
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Track metrics
    metrics = {
        'initial_loss': None,
        'final_loss': None,
        'top1_accuracy': 0.0,
        'top4_accuracy': 0.0,
        'top8_accuracy': 0.0,
    }

    def loss_fn(model, hidden, targets):
        """Compute cross-entropy loss for all 4 prediction heads"""
        # Add sequence dimension (we pooled during collection)
        hidden = mx.expand_dims(hidden, axis=1)

        predictions = model(hidden)

        total_loss = 0.0

        # Loss for each lookahead step
        for step in range(4):
            # Convert to multi-hot encoding
            batch_size = targets.shape[0]
            target_multi_hot = mx.zeros((batch_size, 384))

            for batch_idx in range(batch_size):
                for expert_idx in targets[batch_idx, step, :]:
                    target_multi_hot[batch_idx, int(expert_idx)] = 1.0

            # Binary cross-entropy loss
            logits = predictions[step]
            probs = mx.sigmoid(logits)

            # Binary cross-entropy
            bce = -(target_multi_hot * mx.log(probs + 1e-8) +
                    (1 - target_multi_hot) * mx.log(1 - probs + 1e-8))
            loss = mx.mean(bce)

            total_loss += loss

        return total_loss / 4.0

    # Training loop
    num_batches = (num_samples + batch_size - 1) // batch_size

    for epoch in range(epochs):
        epoch_loss = 0.0

        # Shuffle data
        indices = mx.random.permutation(num_samples)
        hidden_shuffled = hidden_states[indices]
        targets_shuffled = expert_choices[indices]

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)

            batch_hidden = hidden_shuffled[start:end]
            batch_targets = targets_shuffled[start:end]

            # Forward and backward
            loss, grads = mx.value_and_grad(loss_fn)(
                model, batch_hidden, batch_targets
            )

            # Update
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")

        if epoch == 0:
            metrics['initial_loss'] = avg_loss
        if epoch == epochs - 1:
            metrics['final_loss'] = avg_loss

    # Evaluate on training set
    print("\nEvaluating final model...")

    eval_size = min(1000, num_samples)
    eval_hidden = hidden_states[:eval_size]
    eval_targets = expert_choices[:eval_size]

    eval_hidden = mx.expand_dims(eval_hidden, axis=1)
    predictions = model(eval_hidden)

    # Compute top-k accuracies for first prediction head (t+1)
    metrics['top1_accuracy'] = compute_top_k_accuracy(
        predictions[0], eval_targets[:, 0, :], k=1
    )
    metrics['top4_accuracy'] = compute_top_k_accuracy(
        predictions[0], eval_targets[:, 0, :], k=4
    )
    metrics['top8_accuracy'] = compute_top_k_accuracy(
        predictions[0], eval_targets[:, 0, :], k=8
    )

    print(f"Top-1 accuracy: {metrics['top1_accuracy']:.1%}")
    print(f"Top-4 accuracy: {metrics['top4_accuracy']:.1%}")
    print(f"Top-8 accuracy: {metrics['top8_accuracy']:.1%}")

    # Save model
    model.save_weights(output_path)

    # Check model size
    size_mb = output_path.stat().st_size / 1e6
    print(f"Model size: {size_mb:.1f}MB")

    if size_mb > 500:
        print(f"WARNING: Model size {size_mb:.1f}MB exceeds 500MB target")

    return metrics

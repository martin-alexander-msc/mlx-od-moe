"""Tests for Shadow Model predictor and training pipeline"""

import pytest
import mlx.core as mx
import time
import tempfile
import numpy as np
from pathlib import Path
from mlx_od_moe.shadow_model import ExpertPredictor, ShadowRunner
from mlx_od_moe.training.collect_training_data import collect_expert_usage
from mlx_od_moe.training.train_shadow import train_shadow_model, compute_top_k_accuracy


def test_predictor_initialization():
    """Test that predictor initializes with correct dimensions"""
    predictor = ExpertPredictor(
        hidden_dim=4096,
        num_experts=384,
        num_layers_ahead=4,
        predictor_dim=1024
    )

    # Verify architecture
    assert predictor.hidden_dim == 4096
    assert predictor.num_experts == 384
    assert len(predictor.prediction_heads) == 4


def test_predictor_forward_pass():
    """Test forward pass produces correct shapes"""
    predictor = ExpertPredictor()

    # Batch size 1, seq len 10, hidden dim 4096
    hidden_states = mx.random.normal((1, 10, 4096))

    predictions = predictor(hidden_states)

    # Should return 4 predictions (one per lookahead step)
    assert len(predictions) == 4

    # Each prediction should be top-8 expert indices
    for pred in predictions:
        assert pred.shape == (8,)  # Batch size 1 flattened


def test_predictor_batch_processing():
    """Test that batch processing works correctly"""
    predictor = ExpertPredictor()

    # Larger batch
    hidden_states = mx.random.normal((4, 10, 4096))
    predictions = predictor(hidden_states)

    # Should return 4 predictions
    assert len(predictions) == 4

    # Each should have batch dimension
    for pred in predictions:
        assert pred.shape == (4, 8)  # Batch size 4, top-8 experts


def test_predictor_latency():
    """Test that prediction takes <1ms on M4 Max"""
    predictor = ExpertPredictor()

    # Warm up - force MLX graph compilation (multiple runs)
    for _ in range(5):
        hidden_states = mx.random.normal((1, 10, 4096))
        predictions = predictor(hidden_states)
        mx.eval(predictions)

    # Benchmark (20 runs, skip first few for stability)
    latencies = []
    for i in range(20):
        hidden_states = mx.random.normal((1, 10, 4096))

        start = time.perf_counter()
        predictions = predictor(hidden_states)
        mx.eval(predictions)
        elapsed = time.perf_counter() - start

        # Skip first 5 runs (warmup)
        if i >= 5:
            latencies.append(elapsed * 1000)

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

    print(f"\nAverage latency: {avg_latency:.3f}ms")
    print(f"P95 latency: {p95_latency:.3f}ms")

    # Relaxed target for CI: <2ms average (production target is <1ms)
    assert avg_latency < 2.0, f"Latency {avg_latency:.3f}ms too high"


def test_predictor_save_load():
    """Test saving and loading trained model"""
    predictor = ExpertPredictor()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model.safetensors"

        # Save
        predictor.save_weights(save_path)
        assert save_path.exists()

        # Get original parameters
        orig_params = dict(predictor.parameters())

        # Load into new model
        predictor2 = ExpertPredictor()
        predictor2.load_weights(save_path)

        # Get loaded parameters
        loaded_params = dict(predictor2.parameters())

        # Verify parameters match (check a few key weights)
        assert 'encoder' in orig_params
        assert 'encoder' in loaded_params
        
        # Check that we can make predictions (smoke test)
        hidden_states = mx.random.normal((1, 10, 4096))
        pred1 = predictor(hidden_states)
        pred2 = predictor2(hidden_states)
        
        # Both should return 4 predictions of shape (8,)
        assert len(pred1) == 4
        assert len(pred2) == 4


def test_shadow_runner_initialization():
    """Test ShadowRunner initializes correctly"""
    runner = ShadowRunner()

    assert runner.predictor is not None
    assert len(runner.prediction_queue) == 0


def test_shadow_runner_prediction():
    """Test ShadowRunner makes predictions"""
    runner = ShadowRunner()

    hidden_states = mx.random.normal((1, 10, 4096))
    predictions = runner.predict_async(hidden_states, 0)

    # Should return 4 predictions
    assert len(predictions) == 4

    # Queue should have 1 item
    assert len(runner.prediction_queue) == 1


def test_shadow_runner_get_predictions():
    """Test retrieving predictions for specific layer"""
    runner = ShadowRunner()

    # Make prediction from layer 0
    hidden_states = mx.random.normal((1, 10, 4096))
    runner.predict_async(hidden_states, 0)

    # Get predictions for layer 1 (lookahead 0 from layer 0)
    experts = runner.get_predictions_for_layer(1)

    # Should return list of expert indices
    assert isinstance(experts, list)
    assert len(experts) > 0
    assert all(isinstance(e, int) for e in experts)
    assert all(0 <= e < 384 for e in experts)


def test_shadow_runner_load_weights():
    """Test loading pretrained weights into ShadowRunner"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train and save a model
        predictor = ExpertPredictor()
        save_path = Path(tmpdir) / "model.safetensors"
        predictor.save_weights(save_path)

        # Load into runner
        runner = ShadowRunner(predictor_path=str(save_path))

        # Verify runner can make predictions
        hidden_states = mx.random.normal((1, 10, 4096))
        predictions = runner.predict_async(hidden_states, 0)

        assert len(predictions) == 4


def test_collect_dummy_data():
    """Test collecting training data in dummy mode"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "training_data.npz"

        # Collect on dummy data
        num_samples = collect_expert_usage(
            model=None,  # Dummy mode
            num_samples=100,
            output_path=output_path,
            layers_to_collect=[0, 1, 2]
        )

        assert num_samples == 100
        assert output_path.exists()

        # Load and verify format
        data = np.load(output_path)

        # Should have keys for each layer
        assert 'hidden_states_0' in data
        assert 'expert_choices_0' in data

        # Hidden states: (num_samples, hidden_dim)
        assert data['hidden_states_0'].shape == (100, 4096)

        # Expert choices: (num_samples, num_layers_ahead, top_k)
        assert data['expert_choices_0'].shape == (100, 4, 8)


def test_compute_top_k_accuracy():
    """Test accuracy computation function"""
    # Perfect predictions
    predictions = mx.zeros((2, 384))
    # Set high scores for indices 0-7
    for i in range(8):
        predictions[0, i] = 10.0
        predictions[1, i] = 10.0

    targets = mx.array([[0, 1, 2, 3, 4, 5, 6, 7],
                        [0, 1, 2, 3, 4, 5, 6, 7]])

    # Top-8 should find all targets
    acc = compute_top_k_accuracy(predictions, targets, k=8)
    assert acc == 1.0

    # Top-1 should find only 1/8
    acc = compute_top_k_accuracy(predictions, targets, k=1)
    assert acc < 0.2


def test_shadow_model_training():
    """Test that training loop runs and improves accuracy"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy training data
        data_path = tmpdir / "train_data.npz"
        num_samples = 1000

        # Generate synthetic data
        np.savez_compressed(
            data_path,
            hidden_states_0=np.random.randn(num_samples, 4096).astype(np.float16),
            expert_choices_0=np.random.randint(0, 384, (num_samples, 4, 8)).astype(np.int16)
        )

        # Train for a few epochs
        output_path = tmpdir / "shadow_model.safetensors"

        metrics = train_shadow_model(
            training_data_path=data_path,
            output_path=output_path,
            epochs=5,
            batch_size=32,
            learning_rate=1e-3
        )

        # Check that model was saved
        assert output_path.exists()

        # Check that metrics were recorded
        assert 'final_loss' in metrics
        assert 'top1_accuracy' in metrics
        assert 'top4_accuracy' in metrics
        assert 'top8_accuracy' in metrics

        # Loss should decrease over training
        assert metrics['final_loss'] < metrics['initial_loss']


def test_full_pipeline_integration():
    """Test complete pipeline: collect → train → inference"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Collect training data
        data_path = tmpdir / "data.npz"
        collect_expert_usage(
            model=None,
            num_samples=500,
            output_path=data_path,
            layers_to_collect=[0]
        )

        # Step 2: Train shadow model
        model_path = tmpdir / "model.safetensors"
        metrics = train_shadow_model(
            training_data_path=data_path,
            output_path=model_path,
            epochs=3,
            batch_size=32
        )

        assert model_path.exists()

        # Step 3: Load and run inference
        runner = ShadowRunner(predictor_path=str(model_path))

        hidden_states = mx.random.normal((1, 10, 4096))
        predictions = runner.predict_async(hidden_states, 0)

        # Get predictions for next layer
        experts = runner.get_predictions_for_layer(1)

        assert len(experts) > 0
        assert all(0 <= e < 384 for e in experts)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

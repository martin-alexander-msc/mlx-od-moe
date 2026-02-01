"""
Example: Train Shadow Model on Dummy Data

This demonstrates the full training pipeline:
1. Collect training data (dummy mode)
2. Train shadow model predictor
3. Evaluate accuracy and latency
4. Save trained model

In production, replace dummy data with real model inference.
"""

from pathlib import Path
from mlx_od_moe.training.collect_training_data import collect_expert_usage
from mlx_od_moe.training.train_shadow import train_shadow_model
from mlx_od_moe.shadow_model import ShadowRunner
import mlx.core as mx
import time


def main():
    print("=" * 70)
    print("Shadow Model Training Example")
    print("=" * 70)

    # Setup paths
    output_dir = Path("./shadow_training")
    output_dir.mkdir(exist_ok=True)

    data_path = output_dir / "training_data.npz"
    model_path = output_dir / "shadow_model.safetensors"

    # Step 1: Collect training data (dummy mode)
    print("\n[1/3] Collecting training data...")
    print("-" * 70)
    num_samples = collect_expert_usage(
        model=None,  # Dummy mode
        num_samples=5000,
        output_path=data_path,
        layers_to_collect=[0, 1, 2, 3, 4],
        sequence_length=512
    )
    print(f"✓ Collected {num_samples} samples")

    # Step 2: Train shadow model
    print("\n[2/3] Training shadow model...")
    print("-" * 70)
    metrics = train_shadow_model(
        training_data_path=data_path,
        output_path=model_path,
        epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        layer_idx=0
    )

    print(f"\n✓ Training complete!")
    print(f"  Initial loss: {metrics['initial_loss']:.4f}")
    print(f"  Final loss: {metrics['final_loss']:.4f}")
    print(f"  Loss reduction: {(1 - metrics['final_loss']/metrics['initial_loss'])*100:.1f}%")
    print()
    print(f"  Top-1 accuracy: {metrics['top1_accuracy']:.1%}")
    print(f"  Top-4 accuracy: {metrics['top4_accuracy']:.1%}")
    print(f"  Top-8 accuracy: {metrics['top8_accuracy']:.1%}")

    # Check if we hit target (note: dummy data won't hit 90%)
    if metrics['top8_accuracy'] >= 0.90:
        print(f"  ✓ Hit >90% top-8 accuracy target!")
    else:
        print(f"  ⚠ Below 90% target (expected for dummy data)")

    # Step 3: Test inference performance
    print("\n[3/3] Benchmarking inference...")
    print("-" * 70)
    runner = ShadowRunner(predictor_path=str(model_path))

    # Benchmark latency
    latencies = []
    print("Running 100 inference tests...")

    for _ in range(100):
        hidden_states = mx.random.normal((1, 10, 4096))

        start = time.perf_counter()
        predictions = runner.predict_async(hidden_states, 0)
        mx.eval(predictions)  # Force computation
        elapsed = time.perf_counter() - start

        latencies.append(elapsed * 1000)

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
    p99_latency = sorted(latencies)[int(0.99 * len(latencies))]

    print(f"\nLatency statistics:")
    print(f"  Average: {avg_latency:.3f}ms")
    print(f"  P95: {p95_latency:.3f}ms")
    print(f"  P99: {p99_latency:.3f}ms")

    if avg_latency < 1.0:
        print(f"  ✓ Hit <1ms latency target!")
    else:
        print(f"  ⚠ Above 1ms target ({avg_latency:.3f}ms)")

    # Check model size
    size_mb = model_path.stat().st_size / 1e6
    print(f"\nModel size: {size_mb:.1f}MB")

    if size_mb < 500:
        print(f"  ✓ Below 500MB size target!")
    else:
        print(f"  ⚠ Above 500MB target")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Training samples: {num_samples}")
    print(f"Top-8 accuracy: {metrics['top8_accuracy']:.1%}")
    print(f"Average latency: {avg_latency:.3f}ms")
    print(f"Model size: {size_mb:.1f}MB")
    print()
    print("Model saved to:")
    print(f"  {model_path}")
    print()
    print("Next steps for production:")
    print("  1. Collect 50K+ samples from real model")
    print("  2. Train until >90% top-8 accuracy")
    print("  3. Deploy with ShadowRunner in server")
    print("  4. Monitor cache hit rates")
    print("=" * 70)


if __name__ == "__main__":
    main()

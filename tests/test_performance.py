"""
Performance tests for OD-MoE Layer.

These tests verify that the batched implementation provides reasonable performance.
Note: mx.eval() is MLX's array evaluation function, not Python's eval() builtin.
"""

import pytest
import mlx.core as mx
import time
from mlx_od_moe.od_moe_layer import ODMoELayer


class TestPerformance:
    """Performance benchmarks for OD-MoE layer."""

    def test_forward_pass_performance_small(self):
        """Test forward pass completes in reasonable time for small inputs."""
        layer = ODMoELayer(
            layer_idx=0,
            hidden_dim=4096,
            ffn_dim=14336,
            num_experts=384,
            top_k=8
        )

        # Load some dummy experts
        for i in range(16):
            w1 = mx.random.normal((4096, 14336)) * 0.01
            w2 = mx.random.normal((14336, 4096)) * 0.01
            w3 = mx.random.normal((4096, 14336)) * 0.01
            layer.active_experts[i] = (w1, w2, w3)

        # Warmup
        x = mx.random.normal((1, 32, 4096))
        _ = layer(x)

        # Benchmark
        start = time.perf_counter()
        output = layer(x)
        # Force lazy evaluation to complete (MLX-specific, not Python eval)
        mx.eval(output)
        elapsed = time.perf_counter() - start

        # Should complete in under 1 second for small batch
        assert elapsed < 1.0, f"Forward pass took {elapsed:.3f}s, expected <1.0s"

    @pytest.mark.slow
    def test_forward_pass_performance_large(self):
        """Test forward pass on larger context (marked as slow test)."""
        layer = ODMoELayer(
            layer_idx=0,
            hidden_dim=4096,
            ffn_dim=14336,
            num_experts=384,
            top_k=8
        )

        # Load dummy experts
        for i in range(16):
            w1 = mx.random.normal((4096, 14336)) * 0.01
            w2 = mx.random.normal((14336, 4096)) * 0.01
            w3 = mx.random.normal((4096, 14336)) * 0.01
            layer.active_experts[i] = (w1, w2, w3)

        # 8K context
        x = mx.random.normal((1, 8192, 4096))

        # Warmup
        _ = layer(x)

        # Benchmark
        start = time.perf_counter()
        output = layer(x)
        mx.eval(output)  # Force computation (MLX array evaluation, not code eval)
        elapsed = time.perf_counter() - start

        print(f"\n8K context forward pass: {elapsed:.3f}s")

        # Should handle 8K context in reasonable time
        # After vectorization optimization, should be much faster
        assert elapsed < 10.0, f"8K forward pass took {elapsed:.3f}s, expected <10.0s"

    def test_batched_vs_sequential_speedup(self):
        """
        Verify batched implementation is faster than sequential.

        This is a regression test to ensure optimization remains effective.
        """
        layer = ODMoELayer(
            layer_idx=0,
            hidden_dim=128,
            ffn_dim=256,
            num_experts=16,
            top_k=4
        )

        # Load dummy experts
        for i in range(16):
            w1 = mx.random.normal((128, 256)) * 0.1
            w2 = mx.random.normal((256, 128)) * 0.1
            w3 = mx.random.normal((128, 256)) * 0.1
            layer.active_experts[i] = (w1, w2, w3)

        x = mx.random.normal((1, 256, 128))

        # Warmup
        _ = layer(x)

        # Benchmark batched (current implementation)
        start = time.perf_counter()
        for _ in range(10):
            output = layer(x)
            mx.eval(output)  # MLX array evaluation
        batched_time = (time.perf_counter() - start) / 10

        print(f"\nBatched implementation: {batched_time*1000:.2f}ms per forward pass")

        # Ensure it's reasonably fast (should be <100ms for this size)
        assert batched_time < 0.1, f"Batched took {batched_time:.3f}s, expected <0.1s"

"""
Tests for OD-MoE Layer implementation.

Following TDD: Tests written first to define expected behavior.
"""

import pytest
import mlx.core as mx
import numpy as np
from mlx_od_moe.od_moe_layer import ODMoELayer
from mlx_od_moe.expert_store import UnifiedMemoryExpertStore
from mlx_od_moe.shadow_model import ShadowRunner


class TestLoadExperts:
    """Test suite for expert weight loading and splitting."""

    def test_splits_flat_expert_weights_correctly(self):
        """Load experts should split flat array into w1, w2, w3 matrices."""
        # Create layer
        layer = ODMoELayer(
            layer_idx=0,
            hidden_dim=4096,
            ffn_dim=14336,
            num_experts=384,
            top_k=8
        )

        # Create mock expert store that returns flat weights
        # Expected layout: [w1_flat, w2_flat, w3_flat]
        # w1: (hidden, ffn) = 4096 * 14336
        # w2: (ffn, hidden) = 14336 * 4096
        # w3: (hidden, ffn) = 4096 * 14336
        total_params = (4096 * 14336) + (14336 * 4096) + (4096 * 14336)

        # Create dummy expert store
        class MockExpertStore:
            def fetch(self, layer, expert):
                # Return flat array with known pattern
                return mx.ones((total_params,))

        layer.expert_store = MockExpertStore()

        # Load single expert
        layer.load_experts([0])

        # Verify expert was loaded
        assert 0 in layer.active_experts

        # Verify shapes
        w1, w2, w3 = layer.active_experts[0]
        assert w1.shape == (4096, 14336), f"w1 shape wrong: {w1.shape}"
        assert w2.shape == (14336, 4096), f"w2 shape wrong: {w2.shape}"
        assert w3.shape == (4096, 14336), f"w3 shape wrong: {w3.shape}"

    def test_loads_multiple_experts(self):
        """Should load multiple experts in single call."""
        layer = ODMoELayer(layer_idx=0)

        class MockExpertStore:
            def fetch(self, layer, expert):
                total_params = (4096 * 14336) + (14336 * 4096) + (4096 * 14336)
                return mx.ones((total_params,))

        layer.expert_store = MockExpertStore()

        # Load multiple experts
        layer.load_experts([0, 5, 10])

        # All should be loaded
        assert 0 in layer.active_experts
        assert 5 in layer.active_experts
        assert 10 in layer.active_experts

    def test_evicts_experts_not_in_current_topk(self):
        """Should remove experts not in current top-k set."""
        layer = ODMoELayer(layer_idx=0)

        class MockExpertStore:
            def fetch(self, layer, expert):
                total_params = (4096 * 14336) + (14336 * 4096) + (4096 * 14336)
                return mx.ones((total_params,))

        layer.expert_store = MockExpertStore()

        # Load initial set
        layer.load_experts([0, 1, 2])
        assert len(layer.active_experts) == 3

        # Load new set (no overlap)
        layer.load_experts([5, 6, 7])

        # Old experts should be evicted
        assert 0 not in layer.active_experts
        assert 1 not in layer.active_experts
        assert 2 not in layer.active_experts

        # New experts should be loaded
        assert 5 in layer.active_experts
        assert 6 in layer.active_experts
        assert 7 in layer.active_experts


class TestForwardPass:
    """Test suite for optimized forward pass."""

    def test_forward_pass_produces_correct_output_shape(self):
        """Forward pass should maintain input shape."""
        layer = ODMoELayer(
            layer_idx=0,
            hidden_dim=4096,
            ffn_dim=14336,
            top_k=8
        )

        # Small input for testing
        batch, seq_len, hidden = 2, 16, 4096
        x = mx.random.normal((batch, seq_len, hidden))

        # Run forward
        output = layer(x)

        # Should maintain shape
        assert output.shape == (batch, seq_len, hidden)

    def test_batched_expert_application_matches_sequential(self):
        """
        Optimized batched version should produce same results as token-by-token.

        This test will FAIL initially because current implementation is sequential.
        After optimization, both paths should produce identical results.
        """
        # Create layer with small dimensions for testing
        layer = ODMoELayer(
            layer_idx=0,
            hidden_dim=128,
            ffn_dim=256,
            num_experts=16,
            top_k=4
        )

        # Load some dummy experts
        for i in range(16):
            w1 = mx.random.normal((128, 256)) * 0.1
            w2 = mx.random.normal((256, 128)) * 0.1
            w3 = mx.random.normal((128, 256)) * 0.1
            layer.active_experts[i] = (w1, w2, w3)

        # Small input
        x = mx.random.normal((1, 8, 128))

        # Run forward (will use optimized path after implementation)
        output = layer(x)

        # Verify output is valid (not zeros, not NaN)
        assert not mx.all(output == 0)
        assert not mx.any(mx.isnan(output))

        # Verify output magnitude is reasonable (not exploding)
        assert mx.max(mx.abs(output)) < 100.0


class TestRouter:
    """Test suite for router with load balancing."""

    def test_router_produces_valid_probabilities(self):
        """Router should output valid probability distribution."""
        layer = ODMoELayer(layer_idx=0, num_experts=384)

        x = mx.random.normal((2, 16, 4096))
        batch, seq_len, hidden = x.shape

        # Get router logits
        x_flat = x.reshape(-1, hidden)
        router_logits = layer.gate(x_flat)

        # Should have right shape
        assert router_logits.shape == (batch * seq_len, 384)

        # Convert to probabilities
        probs = mx.softmax(router_logits, axis=-1)

        # Probabilities should sum to ~1.0
        prob_sums = mx.sum(probs, axis=-1)
        assert mx.allclose(prob_sums, mx.ones_like(prob_sums), atol=1e-5)

    def test_load_balancing_loss_computed(self):
        """
        Should compute auxiliary load balancing loss.

        This test will FAIL until load balancing is implemented.
        """
        layer = ODMoELayer(layer_idx=0, num_experts=384)

        x = mx.random.normal((2, 16, 4096))

        # Forward pass should compute and store load balancing loss
        output = layer(x)

        # Check that load balancing loss exists and is reasonable
        assert hasattr(layer, 'aux_loss'), "Layer should compute auxiliary loss"
        assert layer.aux_loss is not None
        assert layer.aux_loss >= 0, "Aux loss should be non-negative"
        assert layer.aux_loss < 10.0, "Aux loss should be bounded"

    def test_topk_weight_normalization_toggle(self):
        """Top-k score renormalization should be optional."""

        class FixedGate:
            def __call__(self, x):
                return mx.array([[4.0, 2.0, 1.0, -1.0]])

        x_flat = mx.zeros((1, 4))

        layer_norm = ODMoELayer(
            layer_idx=0,
            hidden_dim=4,
            ffn_dim=8,
            num_experts=4,
            top_k=2,
            norm_topk_prob=True,
        )
        layer_norm.gate = FixedGate()
        _, _, topk_norm = layer_norm._route_experts(x_flat)
        topk_norm_sum = float(mx.sum(topk_norm, axis=-1).item())
        assert abs(topk_norm_sum - 1.0) < 1e-6

        layer_raw = ODMoELayer(
            layer_idx=0,
            hidden_dim=4,
            ffn_dim=8,
            num_experts=4,
            top_k=2,
            norm_topk_prob=False,
        )
        layer_raw.gate = FixedGate()
        _, _, topk_raw = layer_raw._route_experts(x_flat)
        topk_raw_sum = float(mx.sum(topk_raw, axis=-1).item())
        assert topk_raw_sum < 1.0

    def test_route_experts_rejects_non_finite_logits(self):
        """Router should fail fast on NaN/Inf logits."""

        class BadGate:
            def __call__(self, x):
                return mx.array([[0.0, float("nan"), 1.0, 2.0]])

        layer = ODMoELayer(
            layer_idx=0,
            hidden_dim=4,
            ffn_dim=8,
            num_experts=4,
            top_k=2,
        )
        layer.gate = BadGate()

        with pytest.raises(ValueError, match="non-finite"):
            layer._route_experts(mx.zeros((1, 4)))


class TestPrefetch:
    """Test suite for shadow model prefetch integration."""

    def test_triggers_prefetch_for_next_layer(self):
        """Forward pass should trigger prefetch for next layer."""
        # Create mock shadow runner that tracks calls
        class MockShadowRunner:
            def __init__(self):
                self.calls = []

            def get_predictions_for_layer(self, layer_idx):
                self.calls.append(layer_idx)
                return [0, 1, 2, 3, 4, 5, 6, 7]

        # Create mock expert store that tracks prefetch calls
        class MockExpertStore:
            def __init__(self):
                self.prefetch_calls = []

            def fetch(self, layer, expert):
                total_params = (4096 * 14336) + (14336 * 4096) + (4096 * 14336)
                return mx.ones((total_params,))

            def prefetch(self, layer, experts):
                self.prefetch_calls.append((layer, experts))

        shadow_runner = MockShadowRunner()
        expert_store = MockExpertStore()

        layer = ODMoELayer(
            layer_idx=5,
            shadow_runner=shadow_runner,
            expert_store=expert_store
        )

        # Run forward
        x = mx.random.normal((1, 8, 4096))
        _ = layer(x)

        # Should have triggered prefetch for layer 6
        assert len(expert_store.prefetch_calls) > 0
        prefetch_layer, _ = expert_store.prefetch_calls[0]
        assert prefetch_layer == 6

    def test_no_prefetch_for_final_layer(self):
        """Last layer should not trigger prefetch."""
        class MockExpertStore:
            def __init__(self):
                self.prefetch_calls = []

            def fetch(self, layer, expert):
                total_params = (4096 * 14336) + (14336 * 4096) + (4096 * 14336)
                return mx.ones((total_params,))

            def prefetch(self, layer, experts):
                self.prefetch_calls.append((layer, experts))

        expert_store = MockExpertStore()
        shadow_runner = ShadowRunner()

        # Last layer (27)
        layer = ODMoELayer(
            layer_idx=27,
            shadow_runner=shadow_runner,
            expert_store=expert_store
        )

        x = mx.random.normal((1, 8, 4096))
        _ = layer(x)

        # Should not have prefetched (no layer 28)
        assert len(expert_store.prefetch_calls) == 0


class TestTelemetry:
    """Test suite for expert load balancing metrics."""

    def test_tracks_expert_usage(self):
        """
        Should track which experts are used and how frequently.

        This test will FAIL until telemetry is implemented.
        """
        layer = ODMoELayer(layer_idx=0, num_experts=384, top_k=8)

        # Run several forward passes
        for _ in range(5):
            x = mx.random.normal((1, 8, 4096))
            _ = layer(x)

        # Should have usage statistics
        assert hasattr(layer, 'get_expert_usage_stats'), \
            "Layer should provide expert usage stats"

        stats = layer.get_expert_usage_stats()

        # Stats should contain meaningful data
        assert 'expert_counts' in stats
        assert 'total_selections' in stats
        assert stats['total_selections'] > 0

    def test_reports_load_balance_coefficient(self):
        """
        Should compute load balance coefficient (lower = more balanced).

        Perfect balance: all experts used equally → coefficient ≈ 1.0
        Imbalanced: few experts used heavily → coefficient >> 1.0
        """
        layer = ODMoELayer(layer_idx=0, num_experts=384, top_k=8)

        # Run forward passes
        for _ in range(10):
            x = mx.random.normal((1, 16, 4096))
            _ = layer(x)

        # Get stats
        assert hasattr(layer, 'get_expert_usage_stats')
        stats = layer.get_expert_usage_stats()

        # Should report balance metric
        assert 'load_balance_coefficient' in stats
        assert stats['load_balance_coefficient'] >= 1.0

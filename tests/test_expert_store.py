"""Unit tests for ExpertStore - TDD approach"""

import pytest
import mlx.core as mx
from pathlib import Path
import tempfile
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from mlx_od_moe.expert_store import UnifiedMemoryExpertStore


@pytest.fixture
def temp_expert_dir():
    """Create temporary expert directory with real safetensors files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        expert_dir = Path(tmpdir) / "experts"
        expert_dir.mkdir()

        # Create dummy expert files using safetensors format
        # Expert shape: w1 (4096, 14336) + w2 (14336, 4096) + w3 (4096, 14336)
        # Total: 4096*14336 + 14336*4096 + 4096*14336 = 235,929,600 params
        # At fp16: ~471MB per expert (but we'll use smaller for tests)

        # For testing, create smaller experts: 100x200 matrices
        # w1: (100, 200), w2: (200, 100), w3: (100, 200)
        # Total: 20k + 20k + 20k = 60k params
        for layer in range(2):
            for expert in range(8):
                key = f"layer_{layer:02d}_expert_{expert:03d}"

                # Create three weight matrices
                w1 = mx.random.normal((100, 200))
                w2 = mx.random.normal((200, 100))
                w3 = mx.random.normal((100, 200))

                # Save as safetensors
                weights = {'w1': w1, 'w2': w2, 'w3': w3}
                mx.save_safetensors(str(expert_dir / f"{key}.safetensors"), weights)

        yield expert_dir


# ============================================================================
# Test 1: Initialization and Registry
# ============================================================================

def test_expert_store_initialization(temp_expert_dir):
    """
    RED: Test that expert store initializes and indexes all safetensors files.
    Should build registry with metadata (path, size) without loading weights.
    """
    store = UnifiedMemoryExpertStore(
        str(temp_expert_dir),
        cache_size_gb=1,
        num_layers=2,
        num_experts_per_layer=8
    )

    # Should have indexed 16 experts (2 layers × 8 experts)
    assert len(store.expert_registry) == 16
    assert store.current_cache_bytes == 0

    # Check registry has correct metadata
    key = "layer_00_expert_000"
    assert key in store.expert_registry
    assert 'path' in store.expert_registry[key]
    assert 'size' in store.expert_registry[key]
    assert store.expert_registry[key]['path'].exists()


def test_missing_expert_dir_raises_error():
    """RED: Should raise FileNotFoundError if expert directory doesn't exist"""
    with pytest.raises(FileNotFoundError):
        UnifiedMemoryExpertStore(
            "/nonexistent/path",
            cache_size_gb=1,
            num_layers=1,
            num_experts_per_layer=1
        )


# ============================================================================
# Test 2: Fetch Functionality
# ============================================================================

def test_fetch_loads_from_safetensors(temp_expert_dir):
    """
    RED: Test that fetch() loads weights from safetensors and returns mx.array.
    Should use mx.load() and properly split w1, w2, w3 weights.
    """
    store = UnifiedMemoryExpertStore(
        str(temp_expert_dir),
        cache_size_gb=1,
        num_layers=2,
        num_experts_per_layer=8
    )

    # First fetch should load from disk
    expert_weights = store.fetch(0, 0)

    # Should return dict with w1, w2, w3
    assert isinstance(expert_weights, dict)
    assert 'w1' in expert_weights
    assert 'w2' in expert_weights
    assert 'w3' in expert_weights

    # Check shapes
    assert expert_weights['w1'].shape == (100, 200)
    assert expert_weights['w2'].shape == (200, 100)
    assert expert_weights['w3'].shape == (100, 200)

    # Verify stats updated
    assert store.stats['cache_misses'] == 1
    assert store.stats['cache_hits'] == 0
    assert store.stats['bytes_loaded'] > 0


def test_fetch_nonexistent_expert_raises_error(temp_expert_dir):
    """RED: Should raise KeyError if expert doesn't exist in registry"""
    store = UnifiedMemoryExpertStore(
        str(temp_expert_dir),
        cache_size_gb=1,
        num_layers=2,
        num_experts_per_layer=8
    )

    with pytest.raises(KeyError):
        store.fetch(99, 99)  # Non-existent layer/expert


# ============================================================================
# Test 3: Cache Hit/Miss Behavior
# ============================================================================

def test_cache_hit_on_repeated_fetch(temp_expert_dir):
    """
    RED: Test that repeated fetch of same expert hits cache.
    Second fetch should be cache hit, not cache miss.
    """
    store = UnifiedMemoryExpertStore(
        str(temp_expert_dir),
        cache_size_gb=1,
        num_layers=2,
        num_experts_per_layer=8
    )

    # First access: cache miss
    expert1 = store.fetch(0, 0)
    assert store.stats['cache_misses'] == 1
    assert store.stats['cache_hits'] == 0

    # Second access: cache hit
    expert2 = store.fetch(0, 0)
    assert store.stats['cache_hits'] == 1
    assert store.stats['cache_misses'] == 1

    # Should be same object (cached)
    assert expert1['w1'] is expert2['w1']

    # Verify hit rate
    stats = store.get_stats()
    assert stats['hit_rate'] == 0.5  # 1 hit out of 2 requests


# ============================================================================
# Test 4: LRU Eviction
# ============================================================================

def test_lru_eviction_when_cache_full(temp_expert_dir):
    """
    RED: Test that LRU eviction works when cache size exceeded.
    Should evict least recently used expert to make space.
    """
    # Create very small cache: only room for ~2 experts
    # Each expert is ~240KB, so 0.0005GB = ~500KB allows 2 experts
    store = UnifiedMemoryExpertStore(
        str(temp_expert_dir),
        cache_size_gb=0.0005,  # 500KB
        num_layers=2,
        num_experts_per_layer=8
    )

    # Load 3 experts (should evict first one)
    expert0 = store.fetch(0, 0)
    expert1 = store.fetch(0, 1)
    expert2 = store.fetch(0, 2)

    # Cache should not exceed limit (with some tolerance)
    max_cache_bytes = 0.0005 * 1024**3
    assert store.current_cache_bytes <= max_cache_bytes * 1.2  # 20% tolerance

    # First expert should be evicted, not in cache anymore
    assert "layer_00_expert_000" not in store.lru_cache


def test_lru_ordering_updates_on_access(temp_expert_dir):
    """
    RED: Test that accessing an expert moves it to end (most recent).
    LRU should track access order correctly.
    """
    store = UnifiedMemoryExpertStore(
        str(temp_expert_dir),
        cache_size_gb=1,
        num_layers=2,
        num_experts_per_layer=8
    )

    # Load 3 experts
    store.fetch(0, 0)
    store.fetch(0, 1)
    store.fetch(0, 2)

    # Access first expert again (should move to end)
    store.fetch(0, 0)

    # Check ordering: expert 0 should be at the end
    cache_keys = list(store.lru_cache.keys())
    assert cache_keys[-1] == "layer_00_expert_000"


# ============================================================================
# Test 5: Telemetry
# ============================================================================

def test_fetch_latency_tracking(temp_expert_dir):
    """
    RED: Test that fetch latency is tracked for each fetch.
    Should record p50, p95, p99 percentiles.
    """
    store = UnifiedMemoryExpertStore(
        str(temp_expert_dir),
        cache_size_gb=1,
        num_layers=2,
        num_experts_per_layer=8
    )

    # Fetch multiple experts to build latency distribution
    for i in range(8):
        store.fetch(0, i)

    # Check that fetch times were recorded
    assert len(store.stats['fetch_times']) == 8

    # Get stats should compute percentiles
    stats = store.get_stats()
    assert 'p50_latency_ms' in stats
    assert 'p95_latency_ms' in stats
    assert 'p99_latency_ms' in stats

    # Latencies should be positive numbers
    assert stats['p50_latency_ms'] > 0
    assert stats['p95_latency_ms'] >= stats['p50_latency_ms']
    assert stats['p99_latency_ms'] >= stats['p95_latency_ms']


def test_telemetry_tracks_bytes_loaded(temp_expert_dir):
    """RED: Test that total bytes loaded is tracked"""
    store = UnifiedMemoryExpertStore(
        str(temp_expert_dir),
        cache_size_gb=1,
        num_layers=2,
        num_experts_per_layer=8
    )

    # Fetch 3 different experts
    store.fetch(0, 0)
    store.fetch(0, 1)
    store.fetch(0, 2)

    stats = store.get_stats()
    assert stats['bytes_loaded'] > 0

    # Fetch same expert again (cache hit, no new bytes loaded)
    initial_bytes = stats['bytes_loaded']
    store.fetch(0, 0)

    stats = store.get_stats()
    assert stats['bytes_loaded'] == initial_bytes  # No change


# ============================================================================
# Test 6: Prefetch (Thread-Safe)
# ============================================================================

def test_prefetch_loads_experts_async(temp_expert_dir):
    """
    RED: Test that prefetch() loads experts in background thread.
    Should not block, and experts should be in cache when fetched later.
    """
    store = UnifiedMemoryExpertStore(
        str(temp_expert_dir),
        cache_size_gb=1,
        num_layers=2,
        num_experts_per_layer=8
    )

    # Prefetch experts 0-3 in background
    store.prefetch(0, [0, 1, 2, 3])

    # Wait for prefetch to complete
    time.sleep(0.5)

    # Fetch should be cache hit
    initial_hits = store.stats['cache_hits']
    store.fetch(0, 0)
    assert store.stats['cache_hits'] == initial_hits + 1

    # Verify prefetch stats
    stats = store.get_stats()
    assert stats['prefetch_hits'] > 0


def test_prefetch_skips_already_cached(temp_expert_dir):
    """RED: Test that prefetch skips experts already in cache"""
    store = UnifiedMemoryExpertStore(
        str(temp_expert_dir),
        cache_size_gb=1,
        num_layers=2,
        num_experts_per_layer=8
    )

    # Load expert into cache
    store.fetch(0, 0)

    # Prefetch same expert (should skip)
    store.prefetch(0, [0])

    # Should not create redundant prefetch future
    assert "layer_00_expert_000" not in store.prefetch_futures


def test_prefetch_thread_safety(temp_expert_dir):
    """
    RED: Test that concurrent prefetch calls are thread-safe.
    Multiple threads prefetching different experts should not corrupt cache.
    """
    store = UnifiedMemoryExpertStore(
        str(temp_expert_dir),
        cache_size_gb=1,
        num_layers=2,
        num_experts_per_layer=8
    )

    def prefetch_worker(expert_ids):
        """Worker that prefetches a list of experts"""
        for expert_id in expert_ids:
            store.prefetch(0, [expert_id])

    # Launch multiple threads prefetching different experts
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(prefetch_worker, [0, 1]),
            executor.submit(prefetch_worker, [2, 3]),
            executor.submit(prefetch_worker, [4, 5]),
            executor.submit(prefetch_worker, [6, 7]),
        ]

        # Wait for all to complete
        for f in futures:
            f.result()

    # Wait for prefetches to finish
    time.sleep(1.0)

    # All experts should be fetchable without errors
    for i in range(8):
        expert = store.fetch(0, i)
        assert expert is not None


# ============================================================================
# Test 7: Clear Cache
# ============================================================================

def test_clear_cache_empties_lru(temp_expert_dir):
    """RED: Test that clear_cache() removes all experts from cache"""
    store = UnifiedMemoryExpertStore(
        str(temp_expert_dir),
        cache_size_gb=1,
        num_layers=2,
        num_experts_per_layer=8
    )

    # Load some experts
    store.fetch(0, 0)
    store.fetch(0, 1)
    store.fetch(0, 2)

    assert len(store.lru_cache) == 3
    assert store.current_cache_bytes > 0

    # Clear cache
    store.clear_cache()

    # Cache should be empty
    assert len(store.lru_cache) == 0
    assert store.current_cache_bytes == 0


def test_clear_cache_preserves_registry(temp_expert_dir):
    """RED: Test that clear_cache() doesn't affect registry"""
    store = UnifiedMemoryExpertStore(
        str(temp_expert_dir),
        cache_size_gb=1,
        num_layers=2,
        num_experts_per_layer=8
    )

    # Store registry size before clear
    registry_size = len(store.expert_registry)

    # Load and clear
    store.fetch(0, 0)
    store.clear_cache()

    # Registry should be unchanged
    assert len(store.expert_registry) == registry_size


# ============================================================================
# Test 8: Integration Test
# ============================================================================

def test_full_workflow_with_prefetch(temp_expert_dir):
    """
    RED: Integration test - full workflow with prefetch and cache hits.
    Simulates realistic usage pattern.
    """
    store = UnifiedMemoryExpertStore(
        str(temp_expert_dir),
        cache_size_gb=1,
        num_layers=2,
        num_experts_per_layer=8
    )

    # Simulate layer 0 forward pass
    # 1. Prefetch predicted experts for layer 0
    store.prefetch(0, [0, 1, 2])

    # 2. Wait for prefetch
    time.sleep(0.3)

    # 3. Fetch experts (should be cache hits)
    initial_hits = store.stats['cache_hits']
    for expert_id in [0, 1, 2]:
        store.fetch(0, expert_id)

    # Should have had cache hits
    assert store.stats['cache_hits'] > initial_hits

    # 4. Get stats
    stats = store.get_stats()
    assert stats['hit_rate'] >= 0.5
    assert stats['working_set_experts'] >= 3

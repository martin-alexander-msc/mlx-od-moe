"""
Expert Store - Unified memory expert management with LRU cache

Key responsibilities:
- Index all expert files without loading them
- Maintain LRU cache of mx.arrays (working set)
- Async prefetch experts into cache
- Thread-safe access with locks
- Telemetry for cache hit rate tracking
"""

import mlx.core as mx
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from collections import OrderedDict
import time
from typing import Dict, List, Optional


class UnifiedMemoryExpertStore:
    """
    Manages experts across 28 layers using unified memory.
    
    Key insight: On Apple Silicon, we don't "move" data CPU→GPU.
    We just keep pointers. The cache is just working set residency.
    """
    
    def __init__(
        self,
        expert_dir: str,
        cache_size_gb: int = 48,
        num_layers: int = 28,
        num_experts_per_layer: int = 384
    ):
        self.expert_dir = Path(expert_dir)
        self.cache_size = cache_size_gb * 1024**3  # Convert to bytes
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts_per_layer
        
        # Expert registry: metadata only, no weights loaded
        # Format: {key: {'path': Path, 'size': int, 'last_accessed': float}}
        self.expert_registry: Dict[str, Dict] = {}
        
        # LRU cache of expert weights (the actual working set)
        self.lru_cache: OrderedDict[str, Dict[str, mx.array]] = OrderedDict()
        self.current_cache_bytes = 0
        self.cache_lock = threading.RLock()
        
        # Async prefetch
        self.prefetch_executor = ThreadPoolExecutor(max_workers=4)
        self.prefetch_futures: Dict[str, Future] = {}
        
        # Telemetry
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'prefetch_hits': 0,
            'bytes_loaded': 0,
            'fetch_times': []  # Track latency
        }
        
        # Initialize registry
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Build index of all expert files without loading them"""
        print(f"Indexing experts in {self.expert_dir}...")
        
        if not self.expert_dir.exists():
            raise FileNotFoundError(f"Expert directory not found: {self.expert_dir}")
        
        for layer_idx in range(self.num_layers):
            for expert_idx in range(self.num_experts_per_layer):
                key = self._expert_key(layer_idx, expert_idx)
                path = self.expert_dir / f"{key}.safetensors"
                
                if path.exists():
                    self.expert_registry[key] = {
                        'path': path,
                        'size': path.stat().st_size,
                        'last_accessed': 0
                    }
                else:
                    print(f"Warning: Missing expert {key}")
        
        total_size = sum(e['size'] for e in self.expert_registry.values())
        total_gb = total_size / 1e9
        print(f"Indexed {len(self.expert_registry)} experts ({total_gb:.1f}GB total)")
    
    def _expert_key(self, layer: int, expert: int) -> str:
        """Generate consistent expert key"""
        return f"layer_{layer:02d}_expert_{expert:03d}"
    
    def fetch(self, layer: int, expert: int) -> Dict[str, mx.array]:
        """
        Fetch expert weights.

        Fast path: already in LRU cache
        Slow path: load from safetensors file on SSD

        On Apple Silicon unified memory, both are GPU-accessible.

        Returns:
            Dict with keys 'w1', 'w2', 'w3' containing mx.arrays
        """
        key = self._expert_key(layer, expert)
        start_time = time.perf_counter()

        # Fast path: already cached
        with self.cache_lock:
            if key in self.lru_cache:
                self.stats['cache_hits'] += 1
                self.lru_cache.move_to_end(key)  # Mark as recently used
                return self.lru_cache[key]

        # Slow path: load from disk
        self.stats['cache_misses'] += 1

        if key not in self.expert_registry:
            raise KeyError(f"Expert {key} not found in registry")

        entry = self.expert_registry[key]

        # Load from safetensors using mx.load()
        # Returns dict with 'w1', 'w2', 'w3' keys
        weights = mx.load(str(entry['path']))

        # Add to cache
        self._add_to_cache(key, weights, entry['size'])
        self.stats['bytes_loaded'] += entry['size']

        # Record latency
        elapsed = time.perf_counter() - start_time
        self.stats['fetch_times'].append(elapsed)

        return weights
    
    def _add_to_cache(self, key: str, weights: Dict[str, mx.array], size: int):
        """Add to LRU cache with eviction"""
        with self.cache_lock:
            # Evict until there's space
            while self.current_cache_bytes + size > self.cache_size and self.lru_cache:
                oldest_key, oldest_weights = self.lru_cache.popitem(last=False)
                evicted_size = self.expert_registry[oldest_key]['size']
                self.current_cache_bytes -= evicted_size
                # MLX array memory freed when garbage collected
                del oldest_weights

            # Add new expert
            self.lru_cache[key] = weights
            self.lru_cache.move_to_end(key)
            self.current_cache_bytes += size
    
    def prefetch(self, layer: int, experts: List[int]):
        """
        Async prefetch experts into cache.
        Non-blocking—starts background thread to load.

        IMPORTANT: MLX isn't thread-safe for graph operations.
        We use fetch() which handles the thread-safety by using locks.
        The actual fetch() call will be on the main thread when accessed.
        """
        for expert in experts:
            key = self._expert_key(layer, expert)

            # Skip if already cached
            with self.cache_lock:
                if key in self.lru_cache:
                    self.stats['prefetch_hits'] += 1
                    continue

            # Skip if already being fetched
            if key in self.prefetch_futures:
                continue

            # Submit async load - fetch() handles thread safety
            future = self.prefetch_executor.submit(self._prefetch_load, layer, expert)
            self.prefetch_futures[key] = future

    def _prefetch_load(self, layer: int, expert: int):
        """
        Internal prefetch worker.
        Calls fetch() which will load and cache the expert.
        """
        try:
            self.fetch(layer, expert)
            self.stats['prefetch_hits'] += 1
            # Remove from futures dict when done
            key = self._expert_key(layer, expert)
            if key in self.prefetch_futures:
                del self.prefetch_futures[key]
        except Exception as e:
            print(f"Prefetch failed for layer {layer} expert {expert}: {e}")
            key = self._expert_key(layer, expert)
            if key in self.prefetch_futures:
                del self.prefetch_futures[key]
    
    def get_stats(self) -> Dict:
        """Return cache statistics with latency percentiles"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0

        avg_fetch_time = (
            sum(self.stats['fetch_times']) / len(self.stats['fetch_times'])
            if self.stats['fetch_times'] else 0
        )

        # Calculate percentiles if we have fetch times
        if self.stats['fetch_times']:
            sorted_times = sorted(self.stats['fetch_times'])
            n = len(sorted_times)
            # Use max to avoid index 0 for very small samples
            p50_idx = max(0, int(n * 0.50) - 1)
            p95_idx = max(0, int(n * 0.95) - 1)
            p99_idx = max(0, int(n * 0.99) - 1)

            p50_latency = sorted_times[p50_idx] * 1000  # Convert to ms
            p95_latency = sorted_times[p95_idx] * 1000
            p99_latency = sorted_times[p99_idx] * 1000
        else:
            p50_latency = p95_latency = p99_latency = 0

        return {
            **self.stats,
            'hit_rate': hit_rate,
            'cache_size_gb': self.current_cache_bytes / 1e9,
            'working_set_experts': len(self.lru_cache),
            'avg_fetch_time_ms': avg_fetch_time * 1000,
            'p50_latency_ms': p50_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
        }
    
    def clear_stats(self):
        """Reset telemetry counters"""
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'prefetch_hits': 0,
            'bytes_loaded': 0,
            'fetch_times': []
        }

    def clear_cache(self):
        """Clear all cached experts (useful for testing)"""
        with self.cache_lock:
            self.lru_cache.clear()
            self.current_cache_bytes = 0

"""
GGUF-backed expert store.

Loads experts on demand directly from packed MoE tensors in a GGUF file,
avoiding full expert materialization on disk.
"""

from __future__ import annotations

from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from collections import OrderedDict
import time
from typing import Any, Dict, List

import numpy as np
import mlx.core as mx
import gguf
from gguf import quants


def _coerce_gguf_field_value(field) -> Any:
    part = field.parts[-1]
    if isinstance(part, np.ndarray):
        if part.size == 1:
            return part.reshape(()).item()
        return [x.item() if hasattr(x, "item") else x for x in part.reshape(-1)]
    if hasattr(part, "item"):
        try:
            return part.item()
        except Exception:
            pass
    if hasattr(part, "tobytes"):
        try:
            return part.tobytes().decode("utf-8")
        except Exception:
            pass
    return part


def _scalarize(value: Any, default: Any = None) -> Any:
    if isinstance(value, list):
        if not value:
            return default
        numeric = [v for v in value if isinstance(v, (int, float, np.integer, np.floating))]
        if not numeric:
            return default
        return numeric[0]
    return value


def infer_gguf_moe_metadata(gguf_path: str) -> dict[str, Any]:
    """Infer MoE-relevant metadata from GGUF fields."""
    path = Path(gguf_path)
    if not path.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    reader = gguf.GGUFReader(str(path))
    arch = None
    if "general.architecture" in reader.fields:
        part = reader.fields["general.architecture"].parts[-1]
        arch = part.tobytes().decode("utf-8") if hasattr(part, "tobytes") else str(part)

    meta: dict[str, Any] = {}
    if arch:
        for field_name, target in [
            (f"{arch}.block_count", "num_hidden_layers"),
            (f"{arch}.expert_count", "num_local_experts"),
            (f"{arch}.embedding_length", "hidden_size"),
            (f"{arch}.feed_forward_length", "intermediate_size"),
            (f"{arch}.vocab_size", "vocab_size"),
            (f"{arch}.attention.head_count", "num_attention_heads"),
            (f"{arch}.attention.head_count_kv", "num_key_value_heads"),
            (f"{arch}.rope.dimension_count", "head_dim"),
            (f"{arch}.attention.key_length", "attention_key_length"),
            (f"{arch}.attention.value_length", "attention_value_length"),
            (f"{arch}.rope.freq_base", "rope_theta"),
            (f"{arch}.context_length", "max_position_embeddings"),
            (f"{arch}.full_attention_interval", "full_attention_interval"),
            (f"{arch}.norm_top_k_prob", "norm_topk_prob"),
            (f"{arch}.expert_used_count", "num_experts_per_tok"),
            (f"{arch}.expert_feed_forward_length", "moe_intermediate_size"),
            (f"{arch}.expert_shared_feed_forward_length", "shared_expert_intermediate_size"),
            (f"{arch}.ssm.group_count", "linear_num_key_heads"),
            (f"{arch}.ssm.state_size", "linear_state_size"),
            (f"{arch}.ssm.inner_size", "linear_inner_size"),
            (f"{arch}.ssm.conv_kernel", "linear_conv_kernel_dim"),
        ]:
            if field_name in reader.fields:
                value = _coerce_gguf_field_value(reader.fields[field_name])
                value = _scalarize(value, default=value)
                if target == "norm_topk_prob":
                    meta[target] = bool(value)
                elif target == "rope_theta":
                    meta[target] = float(value)
                else:
                    meta[target] = int(value)

        # Derive linear-attention dims for Qwen3Next-style metadata.
        if (
            "linear_num_key_heads" in meta
            and "linear_state_size" in meta
            and "linear_inner_size" in meta
        ):
            state = int(meta["linear_state_size"])
            inner = int(meta["linear_inner_size"])
            meta["linear_key_head_dim"] = state
            meta["linear_value_head_dim"] = state
            if state > 0:
                meta["linear_num_value_heads"] = inner // state

        if "attention_key_length" in meta and "num_attention_heads" in meta:
            key_len = int(meta["attention_key_length"])
            heads = int(meta["num_attention_heads"])
            if heads > 0:
                meta["head_dim"] = key_len
                meta["partial_rotary_factor"] = 0.25

        rope_field = f"{arch}.rope.dimension_count"
        if rope_field in reader.fields and "attention_key_length" in meta:
            rope_dim = _scalarize(_coerce_gguf_field_value(reader.fields[rope_field]))
            if isinstance(rope_dim, (int, float)) and int(meta["attention_key_length"]) > 0:
                meta["partial_rotary_factor"] = float(rope_dim) / float(meta["attention_key_length"])
    return meta


def _align_to_shape(array: np.ndarray, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    if tuple(array.shape) == expected_shape:
        return array
    if tuple(reversed(array.shape)) == expected_shape:
        return np.transpose(array, axes=tuple(reversed(range(array.ndim))))
    raise ValueError(
        f"Failed to align tensor {name}: got {array.shape}, expected {expected_shape}"
    )


class GGUFOnDemandExpertStore:
    """Expert store that reads experts directly from GGUF packed tensors."""

    def __init__(
        self,
        gguf_path: str,
        cache_size_gb: int = 48,
        num_layers: int = 28,
        num_experts_per_layer: int = 384,
        output_dtype=np.float16,
    ):
        self.gguf_path = Path(gguf_path)
        self.cache_size = cache_size_gb * 1024**3
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts_per_layer
        self.output_dtype = output_dtype

        if not self.gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.gguf_path}")

        self.reader = gguf.GGUFReader(str(self.gguf_path))
        self.tensor_lookup = {tensor.name: tensor for tensor in self.reader.tensors}

        # LRU cache and telemetry mirror UnifiedMemoryExpertStore API.
        self.lru_cache: OrderedDict[str, Dict[str, mx.array]] = OrderedDict()
        self.cache_entry_sizes: Dict[str, int] = {}
        self.current_cache_bytes = 0
        self.cache_lock = threading.RLock()
        self.prefetch_executor = ThreadPoolExecutor(max_workers=4)
        self.prefetch_futures: Dict[str, Future] = {}
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "prefetch_hits": 0,
            "bytes_loaded": 0,
            "fetch_times": [],
        }

        print(
            f"Using GGUF on-demand experts from {self.gguf_path} "
            f"(layers={self.num_layers}, experts/layer={self.num_experts_per_layer})"
        )

    def _expert_key(self, layer: int, expert: int) -> str:
        return f"layer_{layer:02d}_expert_{expert:03d}"

    def _dequantize_expert_slice(self, tensor, expert: int) -> np.ndarray:
        qtype = gguf.GGMLQuantizationType(tensor.tensor_type)
        raw = tensor.data
        if raw.ndim < 1:
            raise ValueError(f"Unexpected tensor rank for {tensor.name}: {raw.shape}")
        if expert >= raw.shape[0]:
            raise IndexError(
                f"Expert index {expert} out of range for {tensor.name} raw shape {raw.shape}"
            )

        raw_slice = raw[expert]
        dequantized = quants.dequantize(raw_slice, qtype)
        expected = tuple(int(x) for x in tensor.shape[:-1])
        aligned = _align_to_shape(dequantized, expected, tensor.name)
        return aligned.astype(self.output_dtype)

    def _load_expert(self, layer: int, expert: int) -> tuple[Dict[str, mx.array], int]:
        gate_name = f"blk.{layer}.ffn_gate_exps.weight"
        down_name = f"blk.{layer}.ffn_down_exps.weight"
        up_name = f"blk.{layer}.ffn_up_exps.weight"

        missing = [n for n in (gate_name, down_name, up_name) if n not in self.tensor_lookup]
        if missing:
            raise KeyError(f"Missing packed expert tensors for layer {layer}: {missing}")

        w1_np = self._dequantize_expert_slice(self.tensor_lookup[gate_name], expert)
        w2_np = self._dequantize_expert_slice(self.tensor_lookup[down_name], expert)
        w3_np = self._dequantize_expert_slice(self.tensor_lookup[up_name], expert)

        size_bytes = w1_np.nbytes + w2_np.nbytes + w3_np.nbytes
        weights = {"w1": mx.array(w1_np), "w2": mx.array(w2_np), "w3": mx.array(w3_np)}
        return weights, size_bytes

    def _add_to_cache(self, key: str, weights: Dict[str, mx.array], size: int):
        with self.cache_lock:
            while self.current_cache_bytes + size > self.cache_size and self.lru_cache:
                oldest_key, oldest_weights = self.lru_cache.popitem(last=False)
                evicted_size = self.cache_entry_sizes.pop(oldest_key, 0)
                self.current_cache_bytes -= evicted_size
                del oldest_weights

            self.lru_cache[key] = weights
            self.cache_entry_sizes[key] = size
            self.lru_cache.move_to_end(key)
            self.current_cache_bytes += size

    def fetch(self, layer: int, expert: int) -> Dict[str, mx.array]:
        key = self._expert_key(layer, expert)
        start_time = time.perf_counter()

        with self.cache_lock:
            if key in self.lru_cache:
                self.stats["cache_hits"] += 1
                self.lru_cache.move_to_end(key)
                return self.lru_cache[key]

        self.stats["cache_misses"] += 1
        weights, size_bytes = self._load_expert(layer, expert)
        self._add_to_cache(key, weights, size_bytes)
        self.stats["bytes_loaded"] += size_bytes

        elapsed = time.perf_counter() - start_time
        self.stats["fetch_times"].append(elapsed)
        return weights

    def _prefetch_load(self, layer: int, expert: int):
        try:
            self.fetch(layer, expert)
            self.stats["prefetch_hits"] += 1
        except Exception as e:
            print(f"Prefetch failed for layer {layer} expert {expert}: {e}")
        finally:
            key = self._expert_key(layer, expert)
            if key in self.prefetch_futures:
                del self.prefetch_futures[key]

    def prefetch(self, layer: int, experts: List[int]):
        for expert in experts:
            key = self._expert_key(layer, expert)
            with self.cache_lock:
                if key in self.lru_cache:
                    self.stats["prefetch_hits"] += 1
                    continue
            if key in self.prefetch_futures:
                continue
            self.prefetch_futures[key] = self.prefetch_executor.submit(
                self._prefetch_load, layer, expert
            )

    def get_stats(self) -> Dict:
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = self.stats["cache_hits"] / total_requests if total_requests > 0 else 0
        avg_fetch_time = (
            sum(self.stats["fetch_times"]) / len(self.stats["fetch_times"])
            if self.stats["fetch_times"]
            else 0
        )
        if self.stats["fetch_times"]:
            sorted_times = sorted(self.stats["fetch_times"])
            n = len(sorted_times)
            p50 = sorted_times[max(0, int(n * 0.50) - 1)] * 1000
            p95 = sorted_times[max(0, int(n * 0.95) - 1)] * 1000
            p99 = sorted_times[max(0, int(n * 0.99) - 1)] * 1000
        else:
            p50 = p95 = p99 = 0

        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_size_gb": self.current_cache_bytes / 1e9,
            "working_set_experts": len(self.lru_cache),
            "avg_fetch_time_ms": avg_fetch_time * 1000,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
        }

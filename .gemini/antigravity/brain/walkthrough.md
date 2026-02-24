# Walkthrough

1. Traced runtime memory growth to long-lived `ODMoELayer.active_experts` references and always-on prefetch behavior.
2. Updated `ODMoELayer` to:
   - evict stale active experts before loading new ones,
   - clear active expert references after each forward pass by default when using an expert store.
3. Added strict low-memory cache semantics in both expert stores:
   - `cache_size_gb <= 0` disables LRU retention,
   - skip caching experts larger than configured cache budget.
4. Made prefetch explicit:
   - server exposes `--enable-prefetch` and `--predictor-path`,
   - model setup now disables prefetch by default and only builds `ShadowRunner` when enabled.
5. Aligned shadow predictor construction to runtime model dimensions (hidden size / experts / top-k).
6. Updated README with new defaults and recommended low-memory commands.

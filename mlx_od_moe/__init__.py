"""mlx-od-moe: On-Demand Mixture of Experts for Apple Silicon"""

__version__ = "0.1.0"

from .expert_store import UnifiedMemoryExpertStore
from .gguf_expert_store import GGUFOnDemandExpertStore
from .shadow_model import ExpertPredictor, ShadowRunner
from .od_moe_layer import ODMoELayer
from .model import ODMoEConfig, KimiODMoEModel, KimiODMoEConfig, KVCache, Attention, TransformerBlock
from .qwen3_next_od_model import Qwen3NextODConfig, Qwen3NextODMoEModel

__all__ = [
    "UnifiedMemoryExpertStore",
    "GGUFOnDemandExpertStore",
    "ExpertPredictor",
    "ShadowRunner",
    "ODMoELayer",
    "ODMoEConfig",
    "KimiODMoEModel",
    "KimiODMoEConfig",
    "Qwen3NextODConfig",
    "Qwen3NextODMoEModel",
    "KVCache",
    "Attention",
    "TransformerBlock",
]

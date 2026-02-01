"""
Collect training data for Shadow Model.

Runs inference on a pretrained model and records:
- Hidden states before each MoE layer
- Which experts were actually selected by the router

This creates supervised training data for the predictor.
"""

import mlx.core as mx
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm


def collect_expert_usage(
    model,
    num_samples: int = 10000,
    output_path: Path = Path("training_data.npz"),
    layers_to_collect: Optional[List[int]] = None,
    sequence_length: int = 512
) -> int:
    """
    Collect training data by running inference and recording expert usage.

    Args:
        model: Pretrained model with OD-MoE layers (or None for dummy data)
        num_samples: Number of samples to collect
        output_path: Where to save the collected data
        layers_to_collect: Which layers to collect data for (None = all 28)
        sequence_length: Length of input sequences

    Returns:
        Number of samples collected
    """
    if layers_to_collect is None:
        layers_to_collect = list(range(28))

    # Storage for collected data
    # Format: {layer_idx: {'hidden_states': [], 'expert_choices': []}}
    collected_data: Dict[int, Dict[str, List]] = {
        layer: {'hidden_states': [], 'expert_choices': []}
        for layer in layers_to_collect
    }

    print(f"Collecting {num_samples} samples from {len(layers_to_collect)} layers...")

    for sample_idx in tqdm(range(num_samples), desc="Collecting samples"):
        # Generate random input (in production, use real prompts)
        input_ids = mx.random.randint(0, 50000, (1, sequence_length))

        if model is None:
            # Dummy mode for testing
            for layer in layers_to_collect:
                # Fake hidden state (mean pooled from sequence)
                hidden_state = np.random.randn(4096).astype(np.float16)

                # Fake expert choices for next 4 layers
                # Shape: (4, 8) - 4 lookahead layers, top-8 experts each
                expert_choices = np.random.randint(0, 384, (4, 8), dtype=np.int16)

                collected_data[layer]['hidden_states'].append(hidden_state)
                collected_data[layer]['expert_choices'].append(expert_choices)
        else:
            # Real collection from model
            # TODO: Implement actual model forward pass with hooks
            # This would involve:
            # 1. Register forward hooks on each OD-MoE layer
            # 2. Run model.forward(input_ids)
            # 3. Hooks capture:
            #    - hidden_states before MoE layer
            #    - expert indices selected by router
            # 4. Store for next 4 layers
            raise NotImplementedError("Real model collection not yet implemented")

    # Convert lists to numpy arrays and save
    save_dict = {}
    for layer in layers_to_collect:
        save_dict[f'hidden_states_{layer}'] = np.stack(
            collected_data[layer]['hidden_states']
        )
        save_dict[f'expert_choices_{layer}'] = np.stack(
            collected_data[layer]['expert_choices']
        )

    np.savez_compressed(output_path, **save_dict)
    print(f"Saved {num_samples} samples to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e6:.1f}MB")

    return num_samples


def collect_from_pretrained_model(
    model_path: str,
    expert_dir: str,
    num_samples: int = 10000,
    output_path: Path = Path("training_data.npz")
):
    """
    Collect training data from a pretrained Kimi-K2.5 model.

    This is the production version that actually loads the model.

    Args:
        model_path: Path to base model weights
        expert_dir: Path to expert directory
        num_samples: Number of samples to collect
        output_path: Where to save training data
    """
    # TODO: Implement when full model loading is ready
    # from mlx_od_moe.model import KimiODMoEModel
    # model = KimiODMoEModel.from_pretrained(model_path, expert_dir)
    # collect_expert_usage(model, num_samples, output_path)

    raise NotImplementedError("Pretrained model loading not yet implemented")

"""
Custom collate functions for DataLoader to handle text alignment.
"""

import torch
from typing import List, Dict, Any


def custom_collate_fn(batch):
    """
    Custom collate function to ensure text labels are correctly aligned with trajectories.
    
    The default PyTorch collate function doesn't handle lists of strings well,
    which can cause text labels to be misaligned or lost. This function:
    1. Uses default collate for tensors (obs, action)
    2. Explicitly preserves text as a list of lists, maintaining order
    3. Ensures batch[i] gets text[i] from the dataset
    
    This is critical for maintaining correct text-to-trajectory alignment.
    """
    # Separate tensor fields from text field
    tensor_fields = {}
    text_list = []
    
    # Extract fields from each batch item
    for item in batch:
        for key, value in item.items():
            if key == "text":
                # Preserve text as-is (list of strings per trajectory)
                text_list.append(value)
            else:
                # Collect tensor fields
                if key not in tensor_fields:
                    tensor_fields[key] = []
                tensor_fields[key].append(value)
    
    # Use default collate for tensor fields (obs, action)
    # Stack tensors manually (equivalent to default_collate)
    collated = {}
    for key, values in tensor_fields.items():
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], dict):
            # Handle nested dicts (e.g., obs = {"image": ..., "qpos": ..., "text": ...})
            collated[key] = {}
            for sub_key in values[0].keys():
                if sub_key == "text":
                    # Text is handled separately at top level
                    continue
                sub_values = [v[sub_key] for v in values]
                if isinstance(sub_values[0], torch.Tensor):
                    collated[key][sub_key] = torch.stack(sub_values, dim=0)
                else:
                    collated[key][sub_key] = sub_values
        else:
            # Fallback: try to convert to tensor and stack
            collated[key] = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in values], dim=0)
    
    # Preserve text as list of lists - CRITICAL: maintain order!
    # batch[i] should correspond to text_list[i]
    if text_list:
        collated["text"] = text_list  # List of B lists, each list has T strings
    else:
        collated["text"] = None
    
    return collated


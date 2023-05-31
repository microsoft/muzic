import torch
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
InputDataClass = NewType("InputDataClass", Any)

def default_data_collator(features: List[InputDataClass], return_tensors="pt") -> Dict[str, Any]:
    return torch_default_data_collator(features)

def torch_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}
    batch['labels'] = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "labels" in first and first["labels"] is not None:
        for k in first['labels'].keys():
            label = first["labels"][k].item() if isinstance(first["labels"][k], torch.Tensor) else first["labels"][k]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch['labels'][k] = torch.tensor([f["labels"][k] for f in features], dtype=dtype)
    
    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids", "labels") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    
    return batch
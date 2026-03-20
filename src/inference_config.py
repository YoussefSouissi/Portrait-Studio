from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import dataclasses


@dataclass
class InferenceConfig:
    guidance_scale: float = 9.0
    num_inference_steps: int = 50
    use_random_seed: bool = True
    seed: Optional[int] = None
    height: int = 1024
    width: int = 1024

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

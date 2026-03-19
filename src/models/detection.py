import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Detection:
    """Represents a single detection or annotation in an image."""
    
    label: str
    bbox: Tuple[float, float, float, float] # (x_min, y_min, x_max, y_max) in absolute pixel coordinates
    confidence: Optional[float] = None

@dataclass
class ImageData:
    """Represents an image tile"""
    image_tag: str
    image: np.ndarray
    detections: list[Detection]

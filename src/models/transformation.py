from dataclasses import dataclass
from typing import Tuple

@dataclass
class TransformationMap:
    crop_y_min: int
    crop_y_max: int
    tile_x_min: int
    tile_x_max: int
    x_compression: float
    y_compression: float
    y_padding: int
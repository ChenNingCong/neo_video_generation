
from dataclasses import dataclass
from typing import Tuple
@dataclass
class VideoDatasetInfo:
    is_latent : bool
    image_shape : Tuple
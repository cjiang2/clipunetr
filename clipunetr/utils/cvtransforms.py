from typing import Tuple, Union

from torchvision.transforms import Compose, ToTensor, Normalize
import cv2
import numpy as np

class Resize:
    """Wrapper for OpenCV resize operation.
    Args:
        size: (height, width) or int.
        interpolation: Interpolation Flags from OpenCV. Binary image should use cv2.INTER_LINEAR
    """
    @staticmethod
    def get_params(
        img: np.ndarray, 
        output_size: Union[int, Tuple[int, int]],
        ) -> Tuple[int, int]:
        """Get parameters for ``resizing``.
        """
        # Determine dest size
        h, w = img.shape[:2]
        if isinstance(output_size, int):
            if h > w:
                h, w = int(output_size * h / w), output_size
            else:
                h, w = output_size, int(output_size * (w / h))
        else:
            h, w = output_size[0], output_size[1]
        return h, w

    def __init__(
        self,
        size: Union[int, Tuple[int, int]], 
        interpolation: int = cv2.INTER_LINEAR,
        ):
        self.size = size
        self.interpolation = interpolation

    def __call__(
        self, 
        x: np.ndarray,
        ) -> np.ndarray:
        h, w = self.get_params(x, self.size)
        return cv2.resize(x, (w, h), interpolation=self.interpolation)

    def __repr__(self):
        interpolate_str = self.interpolation

def _transform_clipunetr(n_px):
    return Compose([
        Resize((n_px, n_px), interpolation=cv2.INTER_CUBIC),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
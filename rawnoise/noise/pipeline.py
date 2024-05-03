import torch
import os
from rawnoise.raw import RawImage

class NoisePipeline:
    """
        This class defines a synthetic noise pipeline. It takes in a raw
        image (RawImage class), and adds noise to its raw array.
        The processing is done in PyTorch, even though postprocessing
        is handled by the rawpy library, which takes in NumPy arrays.
        We use PyTorch so that we can implement the postprocessing pipeline
        in PyTorch at a later stage too. 
    """
    def __init__(self, read_noise_fn) -> None:
        self.read_noise_fn = read_noise_fn

    def apply_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.read_noise_fn(tensor)

    def __call__(self, raw: RawImage) -> RawImage:
        tensor = raw.to_tensor()

        tensor = self.apply_noise(tensor)

        return raw.update_from_tensor(tensor)
import torch
import os
from rawnoise.raw import RawImage
from rawnoise.noise import ReadNoise
from copy import deepcopy

class NoisePipeline(torch.nn.Module):
    """
        This class defines a synthetic noise pipeline. It takes in a raw
        image (RawImage class), and adds noise to its raw array.
        The processing is done in PyTorch, even though postprocessing
        is handled by the rawpy library, which takes in NumPy arrays.
        We use PyTorch so that we can implement the postprocessing pipeline
        in PyTorch at a later stage too. 
    """
    def __init__(self, read_noise) -> None:
        super().__init__()
        self.read_noise = read_noise

    def apply_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.read_noise(tensor)

    def __call__(self, raw: RawImage, copy: bool = False) -> RawImage:
        if copy:
            # if the original object can't be modified, set this to true
            raw = deepcopy(raw)
        tensor = raw.to_tensor()

        tensor = self.apply_noise(tensor)

        return raw.update_from_tensor(tensor)
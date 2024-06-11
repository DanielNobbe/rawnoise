import torch
import os
from rawnoise.raw import RawImage
from copy import deepcopy
from .params import GainParams
from .shot_noise import GainSampler, ShotNoise

class NoisePipeline(torch.nn.Module):
    """
        This class defines a synthetic noise pipeline. It takes in a raw
        image (RawImage class), and adds noise to its raw array.
        The processing is done in PyTorch, even though postprocessing
        is handled by the rawpy library, which takes in NumPy arrays.
        We use PyTorch so that we can implement the postprocessing pipeline
        in PyTorch at a later stage too. 

        Here we should sample the K parameter, which is central to
        multiple types of noise.
    """
    def __init__(self, read_noise=None, shot_noise: ShotNoise = None, gain_sampler: GainSampler = None) -> None:
        super().__init__()
        self.read_noise = read_noise
        self.shot_noise: ShotNoise = shot_noise
        self.gain_sampler: GainSampler = gain_sampler

    def apply_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        # gain = self.gain_sampler.sample()
    
        # for testing shot noise
        return self.shot_noise(tensor)


        # return self.read_noise(tensor)

    def __call__(self, raw: RawImage, copy: bool = False) -> RawImage:
        if copy:
            # if the original object can't be modified, set this to true
            raw = deepcopy(raw)
        tensor = raw.to_tensor()

        tensor = self.apply_noise(tensor)

        raw.update_from_tensor(tensor)
        return raw
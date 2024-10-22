import torch
import os
from rawnoise.raw import RawImage
from copy import deepcopy
from .params import GainParams, TukeyParams
from .shot_noise import GainSampler, ShotNoise
from .read_noise import TukeyReadNoise, TukeySampler

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
    def __init__(self, read_noise=None, shot_noise: ShotNoise = None, gain_sampler: GainSampler = None, tukey_sampler: TukeySampler = None) -> None:
        super().__init__()
        self.read_noise = read_noise
        self.shot_noise: ShotNoise = shot_noise
        self.gain_sampler: GainSampler = gain_sampler
        self.tukey_sampler: TukeySampler = tukey_sampler

    def apply_noise(self, tensor: torch.Tensor, ratio: float = 1.0) -> torch.Tensor:
        gain = self.gain_sampler.sample()
        
        # shot noise is a transformation applied to tensor
        shot_noise = self.shot_noise(tensor, K=gain, ratio=ratio)

        sigma, lamda = self.tukey_sampler.sample(gain)

        # read noise is also a transformation on the tensor
        # its additivity is handled internally
        read_noise = self.read_noise(shot_noise, sigma=sigma, lamda=lamda)

        return read_noise

    def __call__(self, raw: RawImage, copy: bool = False) -> RawImage:
        if copy:
            # if the original object shouldn't be modified, set this to true
            raw = deepcopy(raw)
        tensor = raw.to_tensor()

        tensor = self.apply_noise(tensor)

        raw.update_from_tensor(tensor)
        return raw
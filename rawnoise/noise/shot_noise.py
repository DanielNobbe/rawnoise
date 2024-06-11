import torch
from torch.distributions import Uniform, Poisson
from rawnoise.noise.params import GainParams

class GainSampler:
    """
        Class used to sample the gain (K), based on params given by ELD.
        Note that it lives on CPU, if necessary it is possible to 
        bring it to GPU by moving self.distr.low/high to cuda.
    """
    def __init__(self, params: GainParams, saturation_level: int = 2**14-1) -> None:
        self.params = params
        self.distr = Uniform(params.Kmin, params.Kmax)
        self.saturation_level = saturation_level

    def sample(self) -> float:
        K_log = self.distr.sample()
        return torch.exp(K_log) / self.saturation_level
    


class ShotNoise(torch.nn.Module):
    """
        This class defines a function that adds shot noise to an image.
        Shot noise is Poisson distributed, and depends on the image gain.

        For now, use a pre-defined K value (gain).
        Next, implement sampling of K at every forward call, or using a
        given K.
    """
    def __init__(self, K: float, ratio: float) -> None:
        super().__init__()
        self.K = K
        self.rng = torch.Generator(device='cpu')
        self.ratio = ratio  # used to scale the magnitude of the noise

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        rate = tensor / self.ratio / self.K

        # Results are now Poisson distributed,
        # expected value is same as rate
        result = torch.poisson(rate, generator=self.rng)

        result = result * self.K * self.ratio

        return result


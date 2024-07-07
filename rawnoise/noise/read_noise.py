import torch
from torch.distributions import Normal
from rawnoise.noise.params import ScaleParams, TukeyParams

class GaussianReadNoise(torch.nn.Module):
    """
        This class defines a function that adds read noise to an image.
        Read noise is Tukey-Lambda distributed, but we will first 
        implement it as Gaussian noise.
    """
    def __init__(self, ratio: float = 200.) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, tensor: torch.Tensor, sigma: float = 10.) -> torch.Tensor:
        # noise mean is zero
        # noise std dev is sigma
        # mean is zero, no bias in noise
        distr = torch.distributions.Normal(0, sigma)

        noise = distr.sample(tensor.shape)

        tensor /= self.ratio

        result = tensor + noise

        result *= self.ratio

        result = torch.clamp(result, 0, 1)

        return result
    

class ScaleSampler:
    """
        Class used to sample a scale parameter based on
        slope, sigma and bias parameters. See ScaleParams 
        class.

        We will finally sample the TL scale parameter from a Gaussian.
        The sampling works as follows:
            1. Gaussian mean: slope * log(K) + bias
            2. Gaussian scale: sigma
        --> since K can be different between each image, we cannot initialise the full distribution.
        But we can keep a standard-normal and reparametrise it when sampling
    """
    def __init__(self, params: ScaleParams) -> None:
        self.params = params

        self.distr = Normal(0.0, 1.0)

    def sample(self, gain: float = 1.0) -> float:
        
        mean = self.params.slope * torch.log(gain) + self.params.bias

        output = self.distr.sample() * self.params.sigma + mean

        return output


class TukeySampler:
    """
        Class used to sample the Tukey-Lambda scale
        and shape parameters.
        It uses the ScaleSampler to sample the scale
        TODO: Include colour bias too
    """
    def __init__(self, params: TukeyParams, saturation_level: float = 2**14-1) -> None:
        self.params = params
        self.scale_sampler = ScaleSampler(params.scale)
        self.saturation_level = saturation_level

    def sample(self, gain: float = 1.0) -> float:
        scale = self.scale_sampler.sample(gain)
        shape = self.params.shape[torch.randint(0, len(self.params.shape), (1,))]
        # TODO: Use generator for this sample
        return scale / self.saturation_level, shape


class TukeyReadNoise(torch.nn.Module):
    """
        This class defines a function that adds read noise to an image.
        Read noise is Tukey-Lambda distributed.

        NOTE: We are not using the colour bias at this moment.
    """
    def __init__(self, ratio: float = 1., eps=1e-9) -> None:
        super().__init__()
        self.ratio = ratio

        self.base_distr = torch.distributions.Uniform(eps, 1.0)

    
    def sample_TL(self, sample_shape=torch.Size, lamda: float = 1.0, sigma: float = 2.794, eps=1e-9) -> torch.Tensor:
        U = self.base_distr.sample(sample_shape)
        Q = (1/(lamda + eps)) * (U**lamda - (1 - U)**lamda)
        return Q * sigma


    def forward(self, tensor: torch.Tensor, lamda: float = 1.0, sigma: float = 2.794) -> torch.Tensor:
        """
        We can sample from the Tukey-Lambda distribution using inverse
        transform sampling, since the cumulative distribution function
        is known.

        - needs the inverse of the CDF, i.e. the quantile function Q
        - Q = (1/lambda) * (p^lambda - (1- p)^lambda) if lambda !=0
        - Q = ln(p / (1-p)) if lambda = 0

        steps:
        1. Sample from U[0,1] (for each pixel)
        2. Insert U as p into Q(p, lambda) and the output will have the right distr

        https://github.com/Srameo/LED/blob/main/led/data/noise_utils/common.py use a similar method
        They use an epsilon for the uniform though, not sure why?

        This is with a bias of zero, to deal with the bias we need to use the raw_colors in the rawpy object, but we can apply straight to the input/output tensor
        """

        noise = self.sample_TL(tensor.shape, lamda, sigma)

        tensor /= self.ratio

        result = tensor + noise

        result *= self.ratio

        result = torch.clamp(result, 0, 1)

        return result
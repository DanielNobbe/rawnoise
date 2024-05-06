import torch


class ReadNoise(torch.nn.Module):
    """
        This class defines a function that adds read noise to an image.
        Read noise is Tukey-Lambda distributed, but we will first 
        implement it as Gaussian noise.
    """
    def __init__(self, sigma: float = 10.) -> None:
        super().__init__()
        self.sigma = sigma
        
        # good default value for sigma is 10, see unpack-eld-params script

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # TODO: implement as gaussian noise
        # noise mean is zero
        # noise std dev is sigma
        # mean is zero, no bias in noise
        distr = torch.distributions.Normal(0, self.sigma)
        # we define the distr here so that we can sample the params
        # later, and we will sample a lot anyhow
        noise = distr.sample(tensor.shape)
        # we don't need to learn the distr params, so don't use rsample

        result = tensor + noise

        return result
import torch


class ReadNoise(torch.nn.Module):
    """
        This class defines a function that adds read noise to an image.
        Read noise is Tukey-Lambda distributed, but we will first 
        implement it as Gaussian noise.
    """
    def __init__(self, sigma: float = 10., ratio: float = 200., K: float = 0.1) -> None:
        super().__init__()
        self.sigma = sigma
        self.ratio = ratio
        self.K = K

        print(f"Read noise sigma: {sigma}")
        
        # good default value for sigma is 10, see unpack-eld-params script
        # this probably is in the range of 16bit uint though?
        # So divide by 65k? 

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

        tensor /= self.ratio

        result = tensor + noise

        result *= self.ratio

        ## how do we get the right value? Should be careful to not exceed 1
        # anyway
        # TODO: Clip to 1.0

        result = torch.clamp(result, 0, 1)

        return result
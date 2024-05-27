import torch


class GaussianReadNoise(torch.nn.Module):
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
    


class TukeyReadNoise(torch.nn.Module):
    """
        This class defines a function that adds read noise to an image.
        Read noise is Tukey-Lambda distributed.
    """
    def __init__(self, lamda: float = 1.0, sigma: float = 2.794, ratio: float = 200., K: float = 0.1) -> None:
        super().__init__()
        self.lamda = lamda
        self.sigma = sigma
        self.ratio = ratio
        self.K = K

        self.distr = torch.distributions.Uniform(0, 1.0)

        print(f"Read noise sigma: {sigma}")
        
        # good default value for sigma is 10, see unpack-eld-params script
        # this probably is in the range of 16bit uint though?
        # So divide by 65k? 

    
    def sample_TL(self, sample_shape=torch.Size, eps=1e-9) -> torch.Tensor:
        U = self.distr.sample(sample_shape)
        Q = (1/(self.lamda + eps)) * (U**self.lamda - (1 - U)**self.lamda)
        return Q * self.sigma


    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
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
        
        # we define the distr here so that we can sample the params
        # later, and we will sample a lot anyhow
        noise = self.sample_TL(tensor.shape)

        # we don't need to learn the distr params, so don't use rsample

        tensor /= self.ratio

        result = tensor + noise

        result *= self.ratio

        ## how do we get the right value? Should be careful to not exceed 1
        # anyway
        # TODO: Clip to 1.0

        result = torch.clamp(result, 0, 1)

        return result
import numpy as np
import os
import json

def main():

    # npy_path = '/Users/danielnobbe/projects/rawnoise/eld-params/NikonD850_params.npy'
    # npy_path = '/Users/danielnobbe/projects/rawnoise/eld-params/CanonEOS70D_params.npy'
    # npy_path = '/Users/danielnobbe/projects/rawnoise/eld-params/CanonEOS700D_params.npy'
    npy_path = '/Users/danielnobbe/projects/rawnoise/eld-params/SonyA7S2_params.npy'

    with open(npy_path, 'rb') as f:
        params = np.load(f, allow_pickle=True).item()

    """
    Params contains a dict with keys:
        1. G_shape
            Probably has to do with the tukey-lambda distribution params
            Has 18 values in CanonEOS5D4 file
        2. Profile-1
            Camera profile, see below.
        3. Kmin
        4. Kmax
            K will be sampled from a uniform distr between these values. 
            It is a scaling factor for the Poisson distr. for shot noise
        5. color_bias
            Colour bias, not sure how it's used.

        Camera profile (i.e. 'Profile-1'):
            1. G_scale:
                Probably parameters for tukey-lambda distribution for read
                noise
                - bias
                - sigma
                - slope
            2. R_scale:
                Row noise params. Gaussian, so I wonder what the slope 
                param is.
                - bias
                - sigma
                - slope
            3. g_scale:
                Params for Gaussian model for read noise
                Scale for the Gaussian distr is drawn from a normal distr
                with some modifications, and it is dependent on the 
                shot noise K parameter. (which probably has to do with
                photon sensitivity)
                - bias
                - sigma
                - slope
        Probably for a baseline value of the standard deviation for
        Gaussian read noise, we can take
        g_scale_bias + g_scale_slope * (Kmax - Kmin)/2

        For SonyA7S2: 2.817967097156868
        For CanonEOS5D4: 6.166059442821648
        For CanonEOS70D: 9.29072231063719
        For CanonEOS700D: 11.279734826362809
        For NikonD850: 8.165955499878708



    """
    g_scale_sigma = params['Profile-1']['g_scale']['sigma']
    g_scale_bias = params['Profile-1']['g_scale']['bias']
    g_scale_slope = params['Profile-1']['g_scale']['slope']
    Kmax = params['Kmax']
    Kmin = params['Kmin']

    # they multiply sigma by log_K, which is found as
    # log_K = np.random.uniform(low=np.log(1e-1), high=np.log(30))
    # np.log(1e-1) = -2.3025
    # np.log(30) = 3.401
    # E =~ 1.0

    sigma = g_scale_bias + g_scale_slope * 1.0

    print(sigma)
    breakpoint()

    """Tukey lambda has three params:
    - shape (lambda)
    - scale (sigma)
    - colour bias (mu)

    Now, we have for G_scale three values:
    - sigma
    - bias
    - slope
    --> these are probably used to sample a value for the scale of G, analogously to the Guassian scale

    Outside of the `Profile-1`, there is a `G_shape` and `color_bias`, these probably refer to the shape factor (lambda) and the mu.
    Why does G_shape have 18 values?
    And why is the shape of color_bias 18x4?

    Colour bias is four colour pixels:
    R, G1, B, G2

    That explains the shape of colour_bias

    They could not find a statistical model describing the lambda and mu values,
    so these should just be sampled from the empirical distribution, which
    apparently consists of 18 values.
    Note that for most parameters, they depend on K, which is the selected system gain, which correlates to ISO. As such, we can find a value for
    G_scale based on K, but not G_shape. It's odd, since the average G_shape is around 0, and the same goes for the colour_bias. 
    Note that this is for the SAME camera, under different 
    gain values.. So should we not select the values with gain closest?
    Seems like they didn't leave every stone unturned -- why is the colour
    bias and shape different every time?
    The scale decides the magnitude of the noise, in addition to the gain, so obviously the _amount_ of noise increases as the gain increases.
    But the distribution of the noise itself is different every time..

    For now, calculate what the scale should be, and we just randomly pick one of the bias/shape pairs:
     [ 1.1709546 ,  1.0815442 ,  1.1473132 ,  1.0724843 ], -0.1428

    They give a method for sampling the scale value, but let's
    keep it constant for now, just taking the mean they get there

    log(sigma) = sigma_slope*log(K) + sigma_bias
    we set log(K) to 1, since that came out of the uniform distr above

    """

    # G_scale entries are for Tukey Lambda?
    G_scale_sigma = params['Profile-1']['G_scale']['sigma']
    G_scale_bias = params['Profile-1']['G_scale']['bias']
    G_scale_slope = params['Profile-1']['G_scale']['slope']

    sigma = G_scale_bias + G_scale_slope * 1.0
    
    

if __name__ == '__main__':
    main()
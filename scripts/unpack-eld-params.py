import numpy as np
import os

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

    g_scale_bias = params['Profile-1']['g_scale']['bias']
    g_scale_slope = params['Profile-1']['g_scale']['slope']
    Kmax = params['Kmax']
    Kmin = params['Kmin']

    sigma = g_scale_bias + g_scale_slope * (Kmax - Kmin)/2

    print(sigma)


if __name__ == '__main__':
    main()
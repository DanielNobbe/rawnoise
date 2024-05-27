from rawnoise.noise import NoisePipeline, TukeyReadNoise, GaussianReadNoise
from rawnoise.raw import RawImage

def main():
    raw = RawImage('inputs/DSC00866.ARW')

    raw.postprocess(save=True)

    uint14_max = 16384  # values are 14bit in raw? TODO: CHeck
    
    sigma = 2.8 / uint14_max
    ratio = 100

    pipeline = NoisePipeline(
        read_noise=TukeyReadNoise(
            lamda=-0.1428,
            sigma=sigma,
            ratio=ratio,
            K = 0.1  # from ELD it's ~0.09
        )
    )

    # right, in the sampling code in ELD they take the log of this value

    tukey_raw = pipeline(raw, copy=True)

    # raw.update_from_tensor(new_tensor)

    tukey_raw.postprocess(save='outputs/DSC00866_tukey.tiff')

    sigma = 5.2 / uint14_max  # this gives about the same as 2.8 for tukey
    # guassian default is 1.7
    gauss_pipeline = NoisePipeline(
        read_noise=GaussianReadNoise(
            sigma=sigma,
            ratio=ratio,
            K = 0.1  # from ELD it's ~0.09
        )
    )

    gauss_raw = gauss_pipeline(raw, copy=True)

    gauss_raw.postprocess(save='outputs/DSC00866_gauss.tiff')



if __name__ == '__main__':
    main()
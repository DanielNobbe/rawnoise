from rawnoise.noise import NoisePipeline, TukeyReadNoise, GaussianReadNoise, load_params_from_json, ShotNoise
from rawnoise.raw import RawImage

def main():
    raw = RawImage('inputs/DSC00866.ARW')

    raw.postprocess(save=True)

    uint14_max = 16384  # values are 14bit in raw? TODO: CHeck
    
    sigma = 2.8 / uint14_max
    ratio = 100

    params = load_params_from_json('eld-params/SonyA7S2_params.json')


    K = 50. / uint14_max  # uint14_max is saturation level

    # --> this means that the shot noise variation will be stronger with
    # strong amplification. Which makes sense, the number of photons stay 
    # the same, and actually go down to reach a certain intensity level
    # and the shot noise is a kind of quantisation noise, 
    # noticable mostly when total quantities are low.
    # feels like this is slightly different than my original understanding
    # though.. When there are fewer total photons, the noise should be
    # more noticeable. But fewer photons means either:
    # 1. lower exposure (in which case the absolute difference is about the same, potentially even smaller)
    # 2. more gain, in which case the absolute difference is amplified
    # don't know if more photons would mean if it's more likely that the value is close to the expected value

    pipeline = NoisePipeline(
        # read_noise=TukeyReadNoise(
        #     lamda=-0.1428,
        #     sigma=sigma,
        #     ratio=ratio,
        #     K = 0.1  # from ELD it's ~0.09
        # )
        shot_noise=ShotNoise(
            K=K,
            ratio=1.0
        )
    )

    # K = 0.1 is much too high? should it be divided by the saturation value somewhere? Yeah probably we need to get the number of actual photons

    # right, in the sampling code in ELD they take the log of this value

    tukey_raw = pipeline(raw, copy=True)

    # raw.update_from_tensor(new_tensor)

    tukey_raw.postprocess(save='outputs/DSC00866_shot.tiff')

    sigma = 5.2 / uint14_max  # this gives about the same as 2.8 for tukey
    # guassian default is 1.7
    # gauss_pipeline = NoisePipeline(
    #     read_noise=GaussianReadNoise(
    #         sigma=sigma,
    #         ratio=ratio,
    #         K = 0.1  # from ELD it's ~0.09
    #     )
    # )

    # gauss_raw = gauss_pipeline(raw, copy=True)

    # gauss_raw.postprocess(save='outputs/DSC00866_gauss.tiff')



if __name__ == '__main__':
    main()
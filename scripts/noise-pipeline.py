from rawnoise.noise import NoisePipeline, TukeyReadNoise, GaussianReadNoise, load_params_from_json, ShotNoise, GainSampler, TukeySampler
from rawnoise.raw import RawImage
from rawnoise.noise.params import GainParams

def main():
    raw = RawImage('inputs/DSC00866.ARW')

    raw.postprocess(save=True)

    params = load_params_from_json('eld-params/SonyA7S2_params.json')

    pipeline = NoisePipeline(
        gain_sampler=GainSampler(params=params['gain']),
        shot_noise=ShotNoise(),
        tukey_sampler=TukeySampler(params=params['tukey']),
        read_noise=TukeyReadNoise(),
    )

    noised_raw = pipeline(raw, copy=True)

    noised_raw.postprocess(save='outputs/DSC00866_noised.tiff')


if __name__ == '__main__':
    main()
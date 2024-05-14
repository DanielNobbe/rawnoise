from rawnoise.noise import NoisePipeline, ReadNoise
from rawnoise.raw import RawImage

def main():
    raw = RawImage('inputs/DSC00866.ARW')

    raw.postprocess(save=True)

    uint14_max = 16384  # values are 14bit in raw? TODO: CHeck
    
    sigma = 1.7 / uint14_max
    ratio = 100

    pipeline = NoisePipeline(
        read_noise=ReadNoise(\
            sigma=sigma,
            ratio=ratio,
            K = 0.1  # from ELD it's ~0.09
        )
    )

    # right, in the sampling code in ELD they take the log of this value

    new_raw = pipeline(raw)

    # raw.update_from_tensor(new_tensor)

    raw.postprocess(save='outputs/DSC00866_noised.tiff')



if __name__ == '__main__':
    main()
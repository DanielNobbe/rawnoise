from rawnoise.noise import NoisePipeline, ReadNoise
from rawnoise.raw import RawImage

def main():
    raw = RawImage('inputs/DSC00866.ARW')

    raw.postprocess(save=True)

    uint14_max = 16384  # values are 14bit in raw? TODO: CHeck
    
    sigma = 16. / uint14_max
    # saturation_level = (16383 - 800) / uint14_max
    ratio = 100

    pipeline = NoisePipeline(
        read_noise=ReadNoise(\
            sigma=sigma,
            ratio=ratio
        )
    )

    # sigma value default is not correct -- what should it be?
    # something missing from the values is the ISO value relation to the 
    # sigma value.. Although that might have to do with K?
    # use default sigma value
    # raw_tensor = raw.to_tensor()

    # right, in the sampling code in ELD they take the log of this value

    new_raw = pipeline(raw)

    # raw.update_from_tensor(new_tensor)

    raw.postprocess(save='outputs/DSC00866_noised.tiff')



if __name__ == '__main__':
    main()
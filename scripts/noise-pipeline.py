from rawnoise.noise import NoisePipeline, ReadNoise
from rawnoise.raw import RawImage

def main():
    raw = RawImage('inputs/DSC00866.ARW')

    raw.postprocess(save=True)

    pipeline = NoisePipeline(
        read_noise=ReadNoise(sigma=10.)
    )
    # use default sigma value
    # raw_tensor = raw.to_tensor()

    new_raw = pipeline(raw)

    # raw.update_from_tensor(new_tensor)

    raw.postprocess(save='outputs/DSC00866_noised.tiff')



if __name__ == '__main__':
    main()
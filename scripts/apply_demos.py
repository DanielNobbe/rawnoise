import os
import rawpy
from rawnoise.raw import RawImage

def main():
    raw_input = 'inputs/DSC00866.ARW'

    with rawpy.imread(raw_input) as raw:
        print(raw)

        rgb = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  use_camera_wb=False)
        # numpy array of demosaiced image

        print(rgb.shape)

        mat = raw.raw_image  # this gives a copy of the raw Bayer array, which we can apply the noise to
        col = raw.raw_colors  # this gives the colour index at each point in the raw_image array, use to deal with colour-dependent noise
        # they're both numpy

        print(raw.raw_image.max())
        
        mat = mat / 2

        print(raw.raw_image.max())

        raw.raw_image[:] = mat[:]  # we can't overwrite this attribute, but we can modify the array

        print(raw.raw_image.max())
        # not possible to overwrite the raw_image attribute, maybe this is why ELD guy made a custom rawpy?
        # it'll be super annoying to work with files, although we can just write it to a buffer too, still
        # not ideal
        rgb2 = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  use_camera_wb=False)
        print(f"rgb mean {rgb.mean()}, rgb2 mean {rgb2.mean()}")
        # --> indeed the mean gets halved. Max is 255 in both cases, not sure why


        # we need a way to process the raw image, adding noise to it, then still being able to do postprocessing
        # This could be done by inheriting a class from the rawpy class.
        # But then if we ever implement demosaicing in pytorch it's no longer necessary, since we can just keep the operation independent of the rawpy class. There are benefits to having a class in general though, to keep track of camera/whitebalance info that is independent of the raw info.
        # We can just remove the inheritance later on.

        breakpoint()


        

        

if __name__ == '__main__':
    main()
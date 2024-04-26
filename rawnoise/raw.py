import rawpy
import os
import torch
import numpy as np
from numpy.typing import NDArray

class RawImage:
    """
        This class wraps the RawPy class. Our pipeline passes this one around, potentially adding noise to the raw matrix.
        Required functions:
        - to_array: get the raw image as a numpy array (just returns the raw_image attribute)
        - to_tensor: convert the raw image to a tensor
        - update_from_array: set the raw image from a numpy array
        - update_from_tensor: set the raw image from a tensor
    """

    def __init__(self, path: str) -> None:
        self.path = path
        with rawpy.imread(self.path) as raw:
            self.raw_image = raw.raw_image
        # it's a bit annoying, since rawpy needs closing,
        # it probably keeps the file open the entire time, which isn't really
        # necessary.
        # would be better to just keep a reference to the file path,
        # and load the file when necessary for postprocessing, otherwise just keep
        # the raw array

    def to_array(self) -> None:
        return self.raw_image
    
    def to_tensor(self) -> None:
        return torch.from_numpy(self.raw_image)
    
    def update_from_array(self, array: NDArray) -> None:
        # first verify that the size of the new array is the same
        # note: array could be type annotated with dtype, not sure what that is in rawpy though

        if array.shape != self.raw_image.shape:
            raise ValueError(f"New array size ({array.shape}) doesn't match the raw image size ({self.raw_image.shape})")
        
        # assign values to raw_image array (could also replace attr)
        self.raw_image[:] = array[:]

    def update_from_tensor(self, tensor: torch.Tensor) -> None:
        array = tensor.numpy()

        self.update_from_array(array)

    @staticmethod
    def update_raw(raw, array: NDArray) -> None:
        raw.raw_image[:] = array[:]
    
    def postprocess(self) -> NDArray:
        with rawpy.imread(self.path) as raw:
            self.update_raw(raw, self.raw_image)
            result = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  use_camera_wb=False)
        return result
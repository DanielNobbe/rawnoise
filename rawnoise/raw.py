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
        self.rawpy = rawpy.imread(path)

    def to_array(self) -> None:
        return self.rawpy.raw_image
    
    def to_tensor(self) -> None:
        return torch.from_numpy(self.rawpy.raw_image)
    
    def update_from_array(self, array: NDArray) -> None:
        # first verify that the size of the new array is the same
        # note: array could be type annotated with dtype, not sure what that is in rawpy though
        raw_height = self.rawpy.sizes.raw_height
        raw_width = self.rawpy.sizes.raw_width

        array_height = array.shape[0]
        array_width = array.shape[1]

        if raw_height != array_height or raw_width != array_width:
            raise ValueError(f"New array size ({array_height}, {array_width}) doesn't match the raw image size ({raw_height}, {raw_width})")
        
        # assign values to raw_image array, we can't assign a new object as raw_image
        self.rawpy.raw_image[:] = array[:]

    def update_from_tensor(self, tensor: torch.Tensor) -> None:
        array = tensor.numpy()

        self.update_from_array(array)
    
    def postprocess(self) -> NDArray:
        return self.rawpy.postprocess()
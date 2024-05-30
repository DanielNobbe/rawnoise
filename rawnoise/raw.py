import rawpy
import os
import torch
import numpy as np
from numpy.typing import NDArray
from copy import deepcopy
import imageio
from torchvision.transforms import ConvertImageDtype

class RawImage:
    """
        This class wraps the RawPy class. Our pipeline passes this one around, potentially adding noise to the raw matrix.
        Required functions:
        - to_array: get the raw image as a numpy array (just returns the raw_image attribute)
        - to_tensor: convert the raw image to a tensor
        - update_from_array: set the raw image from a numpy array
        - update_from_tensor: set the raw image from a tensor
    """

    def __init__(self, path: str, saturation_level: float = 2**14 - 1) -> None:
        self.path = path
        self.filename = os.path.splitext(os.path.basename(path))[0]
        self.tensor = None
        self.saturation_level = saturation_level
        # NOTE: Need to check how cuda is handled here, since
        # the dtype contains cuda info
        with rawpy.imread(self.path) as raw:
            self.raw_image = raw.raw_image.copy()
            # copy so it's detached from the raw object,
            # they do some weird hacks in rawpy

    def to_array(self, copy=False) -> None:
        print(f"NOTE: The to_array method does not convert the raw image to float values ranged [0,1], but returns the raw integer values.")
        if not copy:
            return self.raw_image
        else:
            print(f"Returning copy")
            return self.raw_image.copy()
        
    @staticmethod
    def convert_to_float(tensor: torch.Tensor, saturation_level: float = 2**14 - 1) -> torch.Tensor:
        # takes a uint16 tensor, converts to float,
        # then scales to range 0-1 (using 14bit max value)
        if tensor.dtype != torch.uint16:
            raise ValueError("tensor must be uint16")

        tensor = tensor.to(dtype=torch.float32)
        tensor = tensor / saturation_level

        return tensor
    
    @staticmethod
    def convert_to_uint(tensor: torch.Tensor, eps=1e-6, saturation_level: float = 2**14 - 1) -> torch.Tensor:
        # takes a float tensor, converts to uint16
        # uses 14bit max value

        if tensor.dtype != torch.float32:
            raise ValueError("tensor must be float32")

        tensor = tensor * (saturation_level + 1.0 - eps)  # stolen from torchvis impl
        # (https://github.com/pytorch/vision/blob/f766d7ac01a4fc6a98b43047e351e51ad7329f5e/torchvision/transforms/_functional_tensor.py#L64)
        tensor = tensor.to(dtype=torch.uint16)

        return tensor
    
    def to_tensor(self) -> None:
        if self.tensor is None:
            tensor = torch.from_numpy(self.raw_image)
            # TODO: Sensors are only 12-14bit, so noise should be applied
            # in that precision too.. Use 32bit for now though
            tensor = self.convert_to_float(tensor, saturation_level=self.saturation_level)

            self.tensor = tensor
        
        return self.tensor
    
    def update_from_array(self, array: NDArray) -> None:
        # first verify that the size of the new array is the same
        # note: array could be type annotated with dtype, not sure what that is in rawpy though

        if array.shape != self.raw_image.shape:
            raise ValueError(f"New array size ({array.shape}) doesn't match the raw image size ({self.raw_image.shape})")
        
        # assign values to raw_image array (could also replace attr)
        self.raw_image = array

    def update_from_tensor(self, tensor: torch.Tensor | None = None) -> None:
        if tensor is None:
            tensor = self.tensor
        tensor = self.convert_to_uint(tensor)
        array = tensor.numpy()

        self.update_from_array(array)

    @staticmethod
    def update_raw(raw, array: NDArray) -> None:
        raw.raw_image[:] = array[:]
    
    @staticmethod
    def save(raw_array: NDArray, path: str) -> None:
        assert os.path.splitext(path)[1] == '.tiff', "Only tiff files are supported"
        imageio.imwrite(path, raw_array)

    def postprocess(self, save: bool | str = False) -> NDArray:
        with rawpy.imread(self.path) as raw:
            self.update_raw(raw, self.raw_image)
            result = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  use_camera_wb=False)

        if save:
            if isinstance(save, str):
                # path specified through save arg
                self.save(result, save)
            else:
                os.makedirs('outputs', exist_ok=True)
                self.save(result, f"outputs/{self.filename}.tiff")

        return result
    
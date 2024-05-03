import numpy as np
from rawnoise.raw import RawImage
import torch

"""
Tests to add:
    - update raw image array from new array then do postprocess
    - update raw image array from new tensor then do postprocess
    - insert new raw array
    - get raw array out
    - get raw tensor out
    - test postprocess (that the right algorithm is used)

"""

def test_postprocess():
    """
        We can't include the postprocessed array in this repo, since it 
        will be too large. 
        We can, however, include the mean and std, which will be 
        close enough to verify that the postprocessing is working 
        correctly.
        TODO: Add jpg of the postprocessed image to compare
    """

    expected_mean = 65.33916398411215
    expected_std = 54.31737883016544
    expected_max = 255
    expected_min = 0

    raw = RawImage('inputs/DSC00866.ARW')

    result = raw.postprocess()

    assert result.mean() == expected_mean
    assert result.std() == expected_std
    assert result.max() == expected_max
    assert result.min() == expected_min



def test_raw_update_from_array():
    # TODO: Update with known values to verify
    raw = RawImage('inputs/DSC00866.ARW')

    initial_array = raw.to_array()

    new_array = initial_array / 2

    result_before = raw.postprocess()

    raw.update_from_array(new_array)

    result_after = raw.postprocess()

    # should not be the same
    assert not np.allclose(result_before, result_after)


def test_raw_update_from_tensor():
    raw = RawImage('inputs/DSC00866.ARW')

    initial_array = raw.to_array()

    new_tensor = torch.ones(initial_array.shape)

    result_before = raw.postprocess()

    raw.update_from_tensor(new_tensor)

    result_after = raw.postprocess()

    # this assert uses the fact that if the entire bayer matrix is 1,
    # the demosaicing algorithm AHD apparently returns a fully-zero image
    assert not result_after.any()
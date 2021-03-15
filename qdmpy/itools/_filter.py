# -*- coding: utf-8 -*-
"""
This module holds functions for filtering images (2D numpy arrays).

Functions
---------
 - `qdmpy.itools._filter.get_im_filtered`
 - `qdmpy.itools._filter.get_im_filtered_gaussian`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.itools._filter.get_im_filtered": True,
    "qdmpy.itools._filter.get_im_filtered_gaussian": True,
}

# ============================================================================

import numpy as np
import scipy.ndimage

# ============================================================================

# hands off to other filters
def get_im_filtered(image, filter_type, **kwargs):
    """Wrapped over other filters defined in `qdmpy.itools._filter'.
    Current filters defined:
        - filter_type = gaussian, `qdmpy.itools._filter.get_im_filtered_gaussian`
    """
    if type(image) != np.ma.core.MaskedArray:
        image = np.ma.masked_array(image)
    filter_fns = {
        "gaussian": get_im_filtered_gaussian,
    }
    # filter not so great with nans (and mask doesn't work w filters) -> set to nanmean
    if "upper_threshold" in kwargs and kwargs["upper_threshold"]:
        image[image > kwargs["upper_threshold"]] = np.nan
    if "lower_threshold" in kwargs and kwargs["lower_threshold"]:
        image[image < kwargs["lower_threshold"]] = np.nan
    image[np.logical_or(np.isnan(image), image.mask)] = np.nanmean(image)

    return filter_fns[filter_type](image, **kwargs)


# ============================================================================


def get_im_filtered_gaussian(image, sigma):
    """Returns image filtered through scipy.ndimage.gaussian_filter with
    parameter 'sigma'."""
    # f_image = np.fft.fft2(image)
    return scipy.ndimage.gaussian_filter(
        image,
        sigma=sigma,
    )
    # return np.fft.ifft2(result).real


# ============================================================================

# -*- coding: utf-8 -*-
"""
This module holds ...

Functions
---------
 - `QDMPy.itools._filter.`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.itools._filter.": True,
}

# ============================================================================

import numpy as np
import scipy.ndimage

# ============================================================================

# hands off to other filters
def get_im_filtered(image, filter_type, **kwargs):
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
    return scipy.ndimage.gaussian_filter(image, sigma=sigma)


# ============================================================================

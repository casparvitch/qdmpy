# -*- coding: utf-8 -*-
"""
This module holds ...

Functions
---------
 - `QDMPy.itools._mask.`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.itools._mask.": True,
}

# ============================================================================

import numpy as np

# ============================================================================

import QDMPy.itools._polygon

# ============================================================================


def mask_polygons(image, polygons=None):
    image = np.array(image)
    if polygons is None:
        return np.ma.masked_array(image, mask=np.zeros(image.shape))
    if len(image.shape) != 2:
        raise ValueError("image is not a 2D array")

    if type(polygons) != list or type(polygons[0]) != QDMPy.itools._polygon.Polygon:
        raise TypeError("polygons were not None, a list or a list of Polygon objects")

    ylen, xlen = image.shape
    masked_area = np.full(image.shape, True)  # all masked to start with

    # coordinate grid for all coordinates TODO check the 'ij' indexing here... (I think xy...)
    grid_y, grid_x = np.meshgrid(range(ylen), range(xlen), indexing="ij")

    for p in polygons:
        in_or_out = p.is_inside(grid_y, grid_x)  # NOTE ensure polygon has same indexing convention
        # mask all vals that are not background
        m = np.ma.masked_greater_equal(in_or_out, 0).mask
        masked_area = np.logical_and(masked_area, ~m)

    return np.ma.masked_array(image, mask=masked_area)


# ============================================================================

# TODO add circle, elliptical mask -> not sure how to define though
# -- define once at PL stage perhaps (separate tool)
#       as they're surely going to be due to laser profile
# -- yeah separate pysimplegui that can be called, and then dragged to resize + move???

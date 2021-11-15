# -*- coding: utf-8 -*-
"""
Shared methods that I couldn't find a nice place to put.

Must be at bottom of dependency tree
"""

# ============================================================================

import numpy as np
import warnings

# ============================================================================


def warn(msg):
    warnings.warn(msg, QDMPYWarning)


# allows us to separate qdmpy warnings from those in other packages.
class QDMPYWarning(Warning):
    pass


# ============================================================================


def define_aois(options):
    """
        Defines areas of interest (aois).

        Returns list of aois that can be used to directly index into image array, e.g.:
        sig_aoi = sig[:, aoi[0], aoi[1]].
    z
        Arguments
        ---------
        options : dict
            Generic options dict holding all the user options.

        Returns
        -------
        aois : list
            List of aoi regions. Much like roi object, these are a length-2 list of np meshgrids
            that can be used to directly index into image to provide a view into just the aoi
            part of the image. E.g. sig_aoi = sig[:, aoi[0], aoi[1]]. Returns a list as in
            general we have more than one area of interest.
            I.e. sig_aoi_1 = sig[:, aois[1][0], aois[1][1]]
    """
    aois = []

    i = 0
    while True:
        i += 1
        try:
            start = options["AOI_" + str(i) + "_start"]
            end = options["AOI_" + str(i) + "_end"]

            if start is None or end is None:
                continue
            aois.append(_define_area_roi(*start, *end))
        except KeyError:
            break
    return aois


# ============================================================================


def define_roi(options, full_size_h, full_size_w):
    """
    Defines meshgrids that can be used to slice image into smaller region of interest (roi).

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    full_size_w : int
        Width of image (after rebin, before roi cut).
    full_size_h : int
        Height of image (after rebin, before roi cut).

    Returns
    -------
    roi : length 2 list of np meshgrids
        Defines an roi that can be applied to the 3D image through direct indexing.
        E.g. sig_roi = sig[:, roi[0], roi[1]]
    """

    if options["ROI"] == "Full":
        roi = _define_area_roi(0, 0, full_size_w - 1, full_size_h - 1)
    elif options["ROI"] == "Rectangle":
        start_x, start_y = options["ROI_start"]
        end_x, end_y = options["ROI_end"]
        roi = _define_area_roi(start_x, start_y, end_x, end_y)

    return roi


# ============================================================================


def _define_area_roi(start_x, start_y, end_x, end_y):
    """
    Makes a list with of meshgrids that defines the roi/aoi. Rectangular.

    Arguments
    ---------
    start_x, start_y, end_x, end_y : int
        Positions of roi vertices as coordinates of indices.
        Defines a rectangle between (start_x, start_y) and (end_x, end_y).

    Returns
    -------
    roi : length 2 list of np meshgrids
        Defines an roi that can be applied to the 3D image through direct indexing.
        E.g. sig_roi = sig[:, roi[0], roi[1]]
    """
    x = np.linspace(start_x, end_x, end_x - start_x + 1, dtype=int)
    y = np.linspace(start_y, end_y, end_y - start_y + 1, dtype=int)
    xv, yv = np.meshgrid(x, y)
    return [
        yv,
        xv,
    ]  # arrays are indexed in image convention, e.g. sig[sweep_param, y, x]


# ============================================================================

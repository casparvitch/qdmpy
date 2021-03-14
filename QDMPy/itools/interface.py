# -*- coding: utf-8 -*-
"""
This module holds ...

Functions
---------
 - `QDMPy.itools.interface.`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.itools.interface.": True,
}

# ============================================================================

import numpy as np

# ============================================================================

import QDMPy.itools._bground
import QDMPy.itools._mask
import QDMPy.itools._polygon

# ============================================================================


def get_background(image, method, polygons=None, **method_params_dict):
    """Returns a background for given image, via chosen method.

    Methods available:
        - "fix_zero"
            - Fix background to be a constant offset (z value)
            - params required in method_params_dict:
                "zero" an int/float, defining the constant offset of the background
        - "three_point"
            - Calculate plane background with linear algebra from three [x,y] lateral positions
              given
            - params required in method_params_dict:
                - "points" a len-3 iterable containing [x, y] points
        - "mean"
            - background calculated from mean of image
            - no params required
        - "poly"
            - background calculated from polynomial fit to image.
            - params required in method_params_dict:
                - "order": an int, the 'order' polynomial to fit. (e.g. 1 = plane).
        - "gaussian"
            - background calculated from gaussian fit to image.
            - no params required
        - "interpolate"
            - Background defined by the dataset smoothed via a sigma-gaussian filtering,
                and method-interpolation over masked (polygon) regions.
            - params required in method_params_dict:
                - "interp_method": nearest, linear, cubic.
                - "sigma": sigma passed to gaussian filter (see scipy.ndimage.gaussian_filter)
                    which is utilized on the background before interpolating
        - "gaussian_filter"
            - background calculated from image filtered with a gaussian filter.
            - params required in method_params_dict:
                - "sigma": sigma passed to gaussian filter (see scipy.ndimage.gaussian_filter)

    polygon utilization:
        - if method is not interpolate, the image is masked where the polygons are
          and the background is calculated without these regions
        - if the method is interpolate, these regions are interpolated over (and the rest
          of the image, gaussian smoothed, is 'background').

    Arguments
    ---------
    image : 2D array-like
        image to get backgrond of
    method : str
        Method to use, available options above
    **method_params_dict : dict
        Key-value pairs passed onto each background backend. Required params
        given above.
    polygons : list, optional
        list of `QDMPy.itools._polygon.Polygon` objects.
        (the default is None, in which case the polygon feature is not used)

    Returns
    -------
    im_bground : ndarray
        2D numpy array, representing the 'background' of image.
    """
    # Discuss masking -> if polygons provided, background is calculated but with
    # polygon regions masked -> background calculated without these.
    # also used for interpolation method.
    method_required_settings = {
        "fix_zero": ["zero"],
        "three_point": ["points"],
        "mean": [],
        "poly": ["order"],
        "gaussian": [],
        "interpolate": ["interp_method", "sigma"],
        "gaussian_filter": ["sigma"],
    }
    method_fns = {
        "fix_zero": QDMPy.itools._bground.zero_background,
        "three_point": QDMPy.itools._bground.three_point_background,
        "mean": QDMPy.itools._bground.mean_background,
        "poly": QDMPy.itools._bground.poly_background,
        "gaussian": QDMPy.itools._bground.gaussian_background,
        "interpolate": QDMPy.itools._bground.interpolated_background,
        "gaussian_filter": QDMPy.itools._bground.filtered_background,
    }
    image = np.array(image)
    if len(image.shape) != 2:
        raise ValueError("image is not a 2D array")
    if type(method) != str:
        raise TypeError("method must be a string.")
    if type(method_params_dict) != dict:
        raise TypeError("method_params_dict must be a dict.")
    if method not in method_required_settings:
        raise ValueError(
            "'method' argument to get_background not in implemented_methods: "
            + f"{method_required_settings.keys()}"
        )
    for setting in method_required_settings[method]:
        if setting not in method_params_dict:
            raise ValueError(f"{setting} key missing from method_params_dict for method: {method}")

    if method != "interpolate":
        # can't mask it for interpolate as we need that info!
        image = QDMPy.itools._mask.mask_polygons(image, polygons)

    if method == "gaussian_filter":
        method_params_dict["filter_type"] = "gaussian"

    if method == "interpolate":
        method_params_dict["polygons"] = polygons

    return method_fns[method](image, **method_params_dict)


# ============================================================================


def get_im_filtered(*args, **kwargs):
    """transparent wrapper for `QDMPy.itools._filter.get_im_filtered`"""
    return QDMPy.itools._filter.get_im_filtered(*args, **kwargs)


# ============================================================================


def polygon_gui(image=None):
    """transparent wrapper for `QDMPy.itools._polygon.polygon_gui(image)`"""
    return QDMPy.itools._polygon.polygon_gui(image)

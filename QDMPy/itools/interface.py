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

# ============================================================================


def get_background(image, method, polygons=None, **method_params_dict):
    method_required_settings = {
        "fix_zero": ["zero"],
        "three_point": ["points"],
        "mean": [],
        "poly": ["order"],
        "gaussian": [],
        "interpolate": ["method", "sigma"],
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

    image = QDMPy.itools._mask.mask_polygons(image, polygons)

    if method == "gaussian_filter":
        method_params_dict["filter_type"] = "gaussian"

    if method == "interpolate":
        method_params_dict["polygons"] = polygons

    return method_fns[method](image, **method_params_dict)


# ============================================================================


def im_filter(*args, **kwargs):
    return QDMPy.itools._filter.im_filter(*args, **kwargs)

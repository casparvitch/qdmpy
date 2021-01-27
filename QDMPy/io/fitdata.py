# -*- coding: utf-8 -*-
"""
This module holds the tools for loading fit results.

Functions
---------
 - `QDMPy.io.fitdata.load_prev_fit_results`
 - `QDMPy.io.fitdata.load_fit_param`
 - `QDMPy.io.fitdata.save_pixel_fit_results`

"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.io.fitdata.load_prev_fit_results": True,
    "QDMPy.io.fitdata.load_fit_param": True,
    "QDMPy.io.fitdata.save_pixel_fit_results": True,
}

# ============================================================================

import numpy as np

# ============================================================================

import QDMPy.io.rawdata
import QDMPy.fit._models as fit_models

# ============================================================================


def load_prev_fit_results(options):
    """Load (all) parameter fit results from previous processing."""

    prev_options = QDMPy.io.rawdata._get_prev_options(options)

    fit_param_res_dict = {}

    for fn_type, num in prev_options["fit_functions"].items():
        for param_name in fit_models.AVAILABLE_FNS[fn_type].param_defn:
            for n in range(num):
                param_key = param_name + "_" + str(n)
                fit_param_res_dict[param_key] = load_fit_param(options, param_key)
    return fit_param_res_dict


# ============================================================================


def load_fit_param(options, param_key):
    """Load a previously fit param, of name 'param_key'."""
    return np.loadtxt(options["data_dir"] / (param_key + ".txt"))


# ============================================================================


def save_pixel_fit_results(options, pixel_fit_params):
    if pixel_fit_params is not None:
        for param_key, result in enumerate(pixel_fit_params):
            np.savetxt(options["data_dir"] / f"{param_key}.txt", result)


# ============================================================================

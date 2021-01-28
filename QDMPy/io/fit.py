# -*- coding: utf-8 -*-
"""
This module holds the tools for loading/saving fit results.

Functions
---------
 - `QDMPy.io.fit.load_prev_fit_results`
 - `QDMPy.io.fit.load_fit_param`
 - `QDMPy.io.fit.save_pixel_fit_results`

"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.io.fit.load_prev_fit_results": True,
    "QDMPy.io.fit.load_fit_param": True,
    "QDMPy.io.fit.save_pixel_fit_results": True,
}

# ============================================================================

import numpy as np
import warnings
import os

# ============================================================================

import QDMPy.io.raw
import QDMPy.constants

# ============================================================================


def load_prev_fit_results(options):
    """Load (all) parameter fit results from previous processing."""

    prev_options = QDMPy.io.raw._get_prev_options(options)

    fit_param_res_dict = {}

    for fn_type, num in prev_options["fit_functions"].items():
        for param_name in QDMPy.constants.AVAILABLE_FNS[fn_type].param_defn:
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
    """
    Saves pixel fit results to disk.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    fit_result_dict : OrderedDict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    """
    if pixel_fit_params is not None:
        for param_key, result in pixel_fit_params.items():
            np.savetxt(options["data_dir"] / f"{param_key}.txt", result)


# ============================================================================


def load_reference_experiment_fit_results(options, ref_options=None, ref_options_dir=None):
    """
    ref_options dict -> pixel_fit_params dict.

    Provide one of ref_options and ref_options_dir. If both are None, returns None (with a
    warning). If both are supplied, ref_options takes precedence.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options (for the main experiment).

    ref_options : dict, default=None
        Generic options dict holding all the user options (for the reference experiment).

    ref_options_dir : str or path object, default=None
        Path to read reference options from, i.e. will read 'ref_options_dir / saved_options.json'.

    Returns
    -------
    fit_result_dict : OrderedDict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.

        If no reference experiment is given (i.e. ref_options and ref_options_dir are None) then
        returns None
    """
    if ref_options is None and ref_options_dir is None:
        warnings.warn(
            "No reference experiment options dict provided, continuing without reference."
        )
        return None

    if ref_options_dir is not None:
        ref_options_path = os.path.join(ref_options_dir, "saved_options.json")
    else:
        ref_options_path = None

    ref_options = QDMPy.io.raw.load_options(
        options_dict=ref_options, options_path=ref_options_path, check_for_prev_result=True
    )

    # ok now have ref_options dict, time to load params
    if ref_options["found_prev_result"]:
        ref_fit_result_dict = load_prev_fit_results(ref_options)
        return ref_fit_result_dict
    else:
        raise RuntimeError("Didn't find reference experiment fit results?")


# ============================================================================

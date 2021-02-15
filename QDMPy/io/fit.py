# -*- coding: utf-8 -*-
"""
This module holds the tools for loading/saving fit results.

Functions
---------
 - `QDMPy.io.fit.load_prev_fit_results`
 - `QDMPy.io.fit.load_fit_param`
 - `QDMPy.io.fit.save_pixel_fit_results`
 - `QDMPy.io.fit.load_reference_experiment_fit_results`

"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.io.fit.load_prev_fit_results": True,
    "QDMPy.io.fit.load_fit_param": True,
    "QDMPy.io.fit.save_pixel_fit_results": True,
    "QDMPy.io.fit.load_reference_experiment_fit_results": True,
}

# ============================================================================

import numpy as np
import warnings
import os
from pathlib import Path

# ============================================================================

import QDMPy.io.raw

# ============================================================================


def load_prev_fit_results(options):
    """Load (all) parameter fit results from previous processing."""

    prev_options = QDMPy.io.raw._get_prev_options(options)

    fit_param_res_dict = {}

    from QDMPy.constants import AVAILABLE_FNS as FN_SELECTOR

    for fn_type, num in prev_options["fit_functions"].items():
        for param_name in FN_SELECTOR[fn_type].param_defn:
            for n in range(num):
                param_key = param_name + "_" + str(n)
                fit_param_res_dict[param_key] = load_fit_param(options, param_key)
    fit_param_res_dict["residual_0"] = load_fit_param(options, "residual_0")
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
        ref_name = "no"
        options["sub_ref_dir"] = options["output_dir"].joinpath(f"sub_{ref_name}_Bnv")
        options["sub_ref_data_dir"] = options["sub_ref_dir"].joinpath("data")
        if not os.path.isdir(options["sub_ref_dir"]):
            os.mkdir(options["sub_ref_dir"])
        if not os.path.isdir(options["sub_ref_data_dir"]):
            os.mkdir(options["sub_ref_data_dir"])
        return None

    if ref_options_dir is not None:
        ref_options_path = os.path.join(ref_options_dir, "saved_options.json")
    else:
        ref_options_path = None

    ref_options = QDMPy.io.raw.load_options(
        options_dict=ref_options,
        options_path=ref_options_path,
        check_for_prev_result=True
    )

    ref_name = Path(ref_options["filepath"]).stem
    # first make a sub ref output folder
    options["sub_ref_dir"] = options["output_dir"].joinpath(f"sub_{ref_name}_Bnv")
    options["sub_ref_data_dir"] = options["sub_ref_dir"].joinpath("data")
    if not os.path.isdir(options["sub_ref_dir"]):
        os.mkdir(options["sub_ref_dir"])
    if not os.path.isdir(options["sub_ref_data_dir"]):
        os.mkdir(options["sub_ref_data_dir"])

    # ok now have ref_options dict, time to load params
    if ref_options["found_prev_result"]:
        ref_fit_result_dict = load_prev_fit_results(ref_options)
        return ref_fit_result_dict
    else:
        from QDMPy.io.json2dict import dict_to_json_str
        print(dict_to_json_str(ref_options))
        raise RuntimeError("Didn't find reference experiment fit results?")


# ============================================================================


def load_ref_options(sig_options, ref_options_dict=None, ref_options_path=None):
    """
    Load and process options (from json file or dict) into generic options dict used everywhere.

    Also handles directory creation etc. to put results in.
    Provide either options_dict OR options_path (must provide one!).

    Note the system default options are loaded in, so you only need to set the things you need.
    In particular, filepath, fit_functions and system_name must be set

    Optional Arguments
    -------------------
    options_dict : dict
        Directly pass in a dictionary of options.
        Default: None

    path : string
        Path to fit options .json file. Can be absolute, or from QDMPy.
        Default: None

    check_for_prev_result : bool
        Check to see if there's a previous fit result for these options.
        Default: False

    Returns
    -------
    options : dict
        Generic options dict holding all the user options.
    """

    # pass options_dict and/or options_path
    if options_dict is not None and options_path is not None:
        # options_dict takes precedence
        options_path = None
    if options_dict is None and options_path is None:
        raise RuntimeError("pass at least one of options_dict and options_path to load_options")

    if options_path is not None:
        prelim_options = QDMPy.io.json2dict.json_to_dict(options_path)
    else:
        prelim_options = OrderedDict(options_dict)  # unnescessary py3.7+, leave to describe intent

    required_options = ["filepath", "fit_functions"]
    for key in required_options:
        if key not in prelim_options:
            raise RuntimeError(f"Must provide these options: {required_options}")

    from QDMPy.constants import choose_system # avoid cyclic imports

    chosen_system = choose_system(prelim_options["system_name"])

    chosen_system.system_specific_option_update(prelim_options)

    options = chosen_system.get_default_options()  # first load in default options
    # now update with what has been decided upon by user
    options = _recursive_dict_update(options, prelim_options)

    chosen_system.determine_binning(options)

    chosen_system.system_specific_option_update(
        options
    )  # do this again on full options to be sure

    options["system"] = chosen_system

    systems.clean_options(options)  # check all the options make sense

    # create output directories
    _define_output_dir(options)

    if not os.path.isdir(options["output_dir"]):
        os.mkdir(options["output_dir"])
    if not os.path.isdir(options["data_dir"]):
        os.mkdir(options["data_dir"])

    _check_if_ref_fit(options)

    return options

# ============================================================================


def _check_if_ref_fit(ref_options, sig_options):
    """
    Looks for previous fit result.

    If previous fit result exists, checks for compatibility between option choices.

    Returns nothing.
    """
    ref_options["found_prev_result"] = (
        QDMPy.io.raw._options_compatible(ref_options, sig_options) # need different method...
        and QDMPy.io.raw._prev_pixel_results_exist(ref_options, sig_options)
    )
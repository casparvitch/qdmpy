# -*- coding: utf-8 -*-
"""
This module holds the tools for loading/saving fit results.

Functions
---------
 - `QDMPy.io.fit.load_prev_fit_results`
 - `QDMPy.io.fit.load_prev_fit_sigmas`
 - `QDMPy.io.fit.load_fit_param`
 - `QDMPy.io.fit.load_fit_sigma`
 - `QDMPy.io.fit.save_pixel_fit_results`
 - `QDMPy.io.fit.save_pixel_fit_sigmas`
 - `QDMPy.io.fit.load_reference_experiment_fit_results`

"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.io.fit.load_prev_fit_results": True,
    "QDMPy.io.fit.load_prev_fit_sigmas": True,
    "QDMPy.io.fit.load_fit_param": True,
    "QDMPy.io.fit.load_fit_sigma": True,
    "QDMPy.io.fit.save_pixel_fit_results": True,
    "QDMPy.io.fit.save_pixel_fit_sigmas": True,
    "QDMPy.io.fit.load_reference_experiment_fit_results": True,
    "QDMPy.io.fit._check_if_already_fit": True,
    "QDMPy.io.fit._prev_options_exist": True,
    "QDMPy.io.fit._options_compatible": True,
    "QDMPy.io.fit._prev_pixel_results_exist": True,
    "QDMPy.io.raw._get_prev_options": True,
}

# ============================================================================

import numpy as np
import warnings
import os
from pathlib import Path

# ============================================================================

import QDMPy.io.raw
import QDMPY.fit as Qfit

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


def load_prev_fit_sigmas(options):
    """Load (all) parameter fit results from previous processing."""

    prev_options = QDMPy.io.raw._get_prev_options(options)

    sigmas = {}

    from QDMPy.constants import AVAILABLE_FNS as FN_SELECTOR

    for fn_type, num in prev_options["fit_functions"].items():
        for param_name in FN_SELECTOR[fn_type].param_defn:
            for n in range(num):
                key = param_name + "_" + str(n)
                sigmas[key] = load_fit_sigma(options, key)
    return sigmas


# ============================================================================


def load_fit_sigma(options, key):
    return np.loadtxt(options["data_dir"] / (key + "_sigma.txt"))


# ============================================================================


def load_fit_param(options, param_key):
    """Load a previously fit param, of name 'param_key'."""
    return np.loadtxt(options["data_dir"] / (param_key + ".txt"))


# ============================================================================


def save_pixel_fit_sigmas(options, sigmas):
    if sigmas is not None:
        for key, result in sigmas.items():
            np.savetxt(options["data_dir"] / f"{key}_sigma.txt", result)


# ============================================================================


def save_pixel_fit_results(options, pixel_fit_params):
    """
    Saves pixel fit results to disk.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    pixel_fit_params : OrderedDict
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
        Generic options dict holding all the user options (for the main/signal experiment).

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
        options["field_dir"] = options["output_dir"].joinpath("field")
        options["field_sig_dir"] = options["field_dir"].joinpath("sig")
        options["field_ref_dir"] = options["field_dir"].joinpath("ref_nothing")

        if not os.path.isdir(options["field_dir"]):
            os.mkdir(options["field_dir"])
        if not os.path.isdir(options["field_sig_dir"]):
            os.mkdir(options["field_sig_dir"])
        if not os.path.isdir(options["field_ref_dir"]):
            os.mkdir(options["field_ref_dir"])
        return None, None

    if ref_options_dir is not None:
        ref_options_path = os.path.join(ref_options_dir, "saved_options.json")
    else:
        ref_options_path = None

    ref_options = QDMPy.io.raw.load_options(
        options_dict=ref_options,
        options_path=ref_options_path,
        check_for_prev_result=True,
        loading_ref=True,
    )

    ref_name = Path(ref_options["filepath"]).stem
    options["field_dir"] = options["output_dir"].joinpath("field")
    options["field_sig_dir"] = options["field_dir"].joinpath("sig")
    options["field_ref_dir"] = options["field_dir"].joinpath(f"ref_{ref_name}")
    if not os.path.isdir(options["field_dir"]):
        os.mkdir(options["field_dir"])
    if not os.path.isdir(options["field_sig_dir"]):
        os.mkdir(options["field_sig_dir"])
    if not os.path.isdir(options["field_ref_dir"]):
        os.mkdir(options["field_ref_dir"])

    # ok now have ref_options dict, time to load params
    if ref_options["found_prev_result"]:
        ref_fit_result_dict = load_prev_fit_results(ref_options)
        ref_sigmas = load_prev_fit_sigmas(ref_options)
        return ref_fit_result_dict, ref_sigmas
    else:
        warnings.warn(
            "Didn't find reference experiment fit results? Reason:\n"
            + ref_options["found_prev_result_reason"]
        )
        return None, None


# ============================================================================


def _check_if_already_fit(options, loading_ref=False):
    """
    Looks for previous fit result.

    If previous fit result exists, checks for compatibility between option choices.

    loading_ref (bool): skip checks for force_fit etc. and just see if prev pixel results exist.

    Returns nothing.
    """
    if not loading_ref:
        if not options["force_fit"]:
            if not _prev_options_exist(options):  # i.e. look for saved options in output dir
                options["found_prev_result_reason"] = "couldn't find previous options"
                options["found_prev_result"] = False
            elif not (res := _options_compatible(options, _get_prev_options(options)))[0]:
                options["found_prev_result_reason"] = "option not compatible:\n" + res[1]
                options["found_prev_result"] = False
            elif not (res2 := _prev_pixel_results_exist(options))[0]:
                options["found_prev_result_reason"] = (
                    "couldn't find prev pixel results:\n" + res2[1]
                )
                options["found_prev_result"] = False
            else:
                options["found_prev_result_reason"] = "found prev result :)"
                options["found_prev_result"] = True
        else:
            options["found_prev_result_reason"] = "option 'force_fit' was True"
            options["found_prev_result"] = False
    elif not (res3 := _prev_pixel_results_exist(options))[0]:
        options["found_prev_result_reason"] = "couldn't find prev pixel results:\n" + res3[1]
        options["found_prev_result"] = False
    else:
        options["found_prev_result_reason"] = "found prev result :)"
        options["found_prev_result"] = True


# ============================================================================


def _prev_options_exist(options):
    """
    Checks if options file from previous result can be found in default location, returns Bool.
    """
    prev_opt_path = os.path.normpath(options["output_dir"] / "saved_options.json")
    return os.path.isfile(prev_opt_path)


# ============================================================================


def _options_compatible(options, prev_options):
    """
    Checks if option choices are compatible with previously fit options

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    prev_options : dict
        Generic options dict from previous fit result.

    Returns
    -------
    _options_compatible : bool
        Whether or not options are compatible.
    """

    if not (
        options["additional_bins"] == prev_options["additional_bins"]
        or (options["additional_bins"] in [0, 1] and prev_options["additional_bins"] in [0, 1])
    ):
        return False, "different binning"
    for option_name in [
        "normalisation",
        "fit_backend",
        "fit_functions",
        "ROI",
        "ignore_ref",
        "system_name",
        "remove_start_sweep",
        "remove_end_sweep",
        "use_ROI_avg_fit_res_for_all_pixels",
    ]:
        if options[option_name] != prev_options[option_name]:
            return False, f"different option: {option_name}"
    # check relevant roi params
    if options["ROI"] == "Rectangle" and (
        options["ROI_start"] != prev_options["ROI_start"]
        and options["ROI_end"] != prev_options["ROI_end"]
    ):
        return False, "different ROI rectangle bounds"

    # check relevant param guesses/bounds

    # check relevant fit params
    if options["fit_backend"] == "scipyfit":
        for fit_opt_name in [
            "scipyfit_method",
            "scipyfit_use_analytic_jac",
            "scipyfit_fit_jac_acc",
            "scipyfit_fit_gtol",
            "scipyfit_fit_xtol",
            "scipyfit_fit_ftol",
            "scipyfit_scale_x",
            "scipyfit_loss_fn",
        ]:
            if (
                fit_opt_name not in options
                or fit_opt_name not in prev_options
                or options[fit_opt_name] != prev_options[fit_opt_name]
            ):
                return False, f"scipyfit option different: {fit_opt_name}"
    elif options["fit_backend"] == "gpufit":
        for fit_opt_name in ["gpufit_tolerance", "gpufit_max_iterations", "gpufit_estimator_id"]:
            if options[fit_opt_name] != prev_options[fit_opt_name]:
                return False, f"gpufit option different: {fit_opt_name}"

    # ok now the trickiest one, check parameter guesses & bounds
    unique_params = set(Qfit.get_param_defn(Qfit.FitModel(options["fit_functions"])))

    for param_name in unique_params:
        if options[param_name + "_guess"] != prev_options[param_name + "_guess"]:
            return False, f"param guess different: {param_name}"

        range_opt = param_name + "_range"
        if range_opt in options and range_opt in prev_options:
            if options[range_opt] != prev_options[range_opt]:
                return False, f"different range options: {param_name}"
            else:
                continue  # this param all g, check others
        # ok range takes precedence over bounds
        if range_opt in options and range_opt not in prev_options:
            return False, f"different range/bound options: {param_name}"
        if range_opt not in options and range_opt in prev_options:
            return False, f"different range/bound options: {param_name}"
        # finally check bounds
        if options[param_name + "_bounds"] != prev_options[param_name + "_bounds"]:
            return False, f"param range different: {param_name}"

    # if all that was ok, return True
    return True, "all g"


# ============================================================================


def _prev_pixel_results_exist(options):
    """
    Check if the actual fit result files exists.

    Arguments
    ---------
    options : dict
        Generic options dict from (either prev. or current, should be the equiv.) fit result.

    Returns
    -------
    pixels_results_exist : bool
        Whether or not previous pixel result files exist.
    """

    # avoid cyclic imports
    from QDMPy.constants import AVAILABLE_FNS as FN_SELECTOR

    for fn_type, num in options["fit_functions"].items():
        for param_name in FN_SELECTOR[fn_type].param_defn:
            for n in range(num):
                param_key = param_name + "_" + str(n)
                if not os.path.isfile(options["data_dir"] / (param_key + ".txt")):
                    return False, f"couldn't find previous param: {param_key}"

    if not os.path.isfile(options["data_dir"] / "residual_0.txt"):
        return False, "couldn't find previous residual"
    return True, "found all prev pixel results :)"


# ============================================================================


def _prev_sigma_results_exist(prev_options):
    """ as `QDMPy.io.raw._prev_pixel_results_exist` but for sigmas """
    # avoid cyclic imports
    from QDMPy.constants import AVAILABLE_FNS as FN_SELECTOR

    for fn_type, num in prev_options["fit_functions"].items():
        for param_name in FN_SELECTOR[fn_type].param_defn:
            for n in range(num):
                param_key = param_name + "_" + str(n)
                if not os.path.isfile(prev_options["data_dir"] / (param_key + "_sigma.txt")):
                    return False

    return True


# ============================================================================


def _get_prev_options(options):
    """
    Reads options file from previous fit result (.json), returns a dictionary.
    """
    return QDMPy.io.json_to_dict(options["output_dir"] / "saved_options.json")


# ============================================================================

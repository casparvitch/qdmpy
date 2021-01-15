# -*- coding: utf-8 -*-
"""
This module holds the general interface tools for fitting raw data, independent of fit backend
(e.g. scipy/gpufit etc.).

Functions
---------
 - `QDMPy.fit_interface.get_pixel_fitting_results`
 - `QDMPy.fit_interface.load_prev_fit_results`
 - `QDMPy.fit_interface.load_fit_param`
 - `QDMPy.fit_interface.define_fit_model`
 - `QDMPy.fit_interface.prep_fit_backends`
 - `QDMPy.fit_interface.fit_ROI_avg`
 - `QDMPy.fit_interface.fit_AOIs`
 - `QDMPy.fit_interface.fit_pixels`
"""

# ============================================================================

__author__ = "Sam Scholten"

# ============================================================================

import numpy as np

# ============================================================================

import QDMPy.fit_models as fit_models
import QDMPy.data_loading as data_loading

# ============================================================================


def get_pixel_fitting_results(fit_model, fit_results, roi_shape):
    """
    Take the fit result data from scipy/gpufit and back it down to a dictionary of arrays.

    Each array is 2D, representing the values for each parameter (specified by the dict key).


    Arguments
    ---------
    fit_model : `QDMPy.fit_models.FitModel` object.

    fit_results : list of [(x, y), result] objects
        (see `QDMPy.fit_scipy.to_squares_wrapper`, or `QDMPy.fit_gpufit.gpufit_reshape_result`)
        A list of each pixels parameter array: fit_result.x (as well as position in image
        denoted by (x, y), returned by the concurrent mapper in `fitting.fit_pixels`.

    roi_shape : iterable, length 2
        Shape of the region of interest, to allow us to generate the right shape empty arrays.
        This is probably a little useless, we could extract from any of the results.

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    """
    # initialise dictionary with key: val = param_name: param_units
    fit_image_results = fit_models.get_param_odict(fit_model)

    # override with correct size empty arrays using np.zeros
    for key in fit_image_results.keys():
        fit_image_results[key] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan

    # Fill the arrays element-wise from the results function, which returns a
    # 1D array of flattened best-fit parameters.

    for (x, y), result in fit_results:
        filled_params = {}  # keep track of index, i.e. pos_0, for this pixel
        for fn in fit_model.fn_chain:
            for param_num, param_name in enumerate(fn.param_defn):

                # keep track of what index we're up to, i.e. pos_1
                if param_name not in filled_params.keys():
                    key = param_name + "_0"
                    filled_params[param_name] = 1
                else:
                    key = param_name + "_" + str(filled_params[param_name])
                    filled_params[param_name] += 1

                fit_image_results[key][x, y] = result[fn.this_fn_param_indices[param_num]]

    return fit_image_results


# ============================================================================


def load_prev_fit_results(options):
    """Load (all) parameter fit results from previous processing."""

    prev_options = data_loading.get_prev_options(options)

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


def define_fit_model(options):
    """Define (and return) fit_model object, from options dictionary."""

    fit_functions = options["fit_functions"]

    fit_model = fit_models.FitModel(fit_functions)

    options["fit_param_defn"] = fit_models.get_param_odict(fit_model)

    prep_fit_backends(options, fit_model)

    return fit_model


# ============================================================================


def prep_fit_backends(options, fit_model):
    """
    Prepare all possible fit backends, checking that everything will work.

    Also attempts to import relevant modules into global scope.

    This is a wrapper around specific functions for each backend. All possible fit
    backends are loaded - these are decided in the config file for this system,
    i.e. system.option_choices("fit_backend")

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    fit_model : `QDMPy.fit_models.FitModel` object.

    Returns
    -------
    None

    """
    # ensure backend we want to use for pixel fittings is in comparison!
    if options["fit_backend"] not in options["fit_backend_comparison"]:
        options["fit_backend_comparison"].append(options["fit_backend"])

    for fit_backend in options["fit_backend_comparison"]:
        if fit_backend == "scipy":
            # import, but make it globally available (to module)
            global fit_scipy
            _temp = __import__("QDMPy.fit_scipy", globals(), locals())
            fit_scipy = _temp.fit_scipy

        elif fit_backend == "gpufit":
            # here we use a programmatic import as we don't want to load (and crash)
            # if user doesn't have the gpufit stuff installed
            global fit_gpufit
            _temp = __import__("QDMPy.fit_gpufit", globals(), locals())
            fit_scipy = _temp.fit_gpufit

            fit_gpufit.prep_gpufit_backend(options, fit_model)
        else:
            raise RuntimeError(
                f"No backend preparation defined for fit_backend = {options['fit_backend']}"
            )


# ============================================================================


def fit_ROI_avg(options, sig_norm, sweep_list, fit_model):
    """
    Fit the average of the measurement over the region of interest specified, with backend
    chosen via options["fit_backend_comparison"].

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    sig_norm : np array, 3D
        Normalised measurement array, shape: [sweep_list, x, y].

    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)

    fit_model : `QDMPy.fit_models.FitModel` object.

    Returns
    -------
    backend_ROI_results_lst : list
        List of `QDMPy.fitting.FitResultROIAvg` objects containing the fit result
        (see class specifics) for each fit backend selected for comparison

    """

    # list of ROI results, one element for each backend (as a (name, result) tuple)
    backend_ROI_results_lst = []
    # iterate through all possible fit backend choices
    for fit_backend in options["fit_backend_comparison"]:
        if fit_backend == "scipy":
            backend_ROI_results_lst.append(
                fit_scipy.fit_ROI_avg_scipy(options, sig_norm, sweep_list, fit_model)
            )
        elif fit_backend == "gpufit":

            backend_ROI_results_lst.append(
                fit_gpufit.fit_ROI_avg_gpufit(options, sig_norm, sweep_list, fit_model)
            )
        else:
            raise RuntimeError(
                f"No fit_ROI_avg fn defined for fit_backend = {options['fit_backend']}"
            )
    return backend_ROI_results_lst


# ============================================================================


def fit_AOIs(
    options, sig_norm, single_pixel_pl, sweep_list, fit_model, AOIs, backend_ROI_results_lst
):
    """
    Fit AOIs and single pixel with chosen backends and return fit_result_collection_lst

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    sig_norm : np array, 3D
        Normalised measurement array, shape: [sweep_list, x, y].

    single_pixel_pl : np array, 1D
        Normalised measurement array, for chosen single pixel check.

    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq).

    fit_model : `fit_models.FitModel` object.

    AOIs : list
        List of AOI specifications - each a length-2 iterable that can be used to directly index
        into sig_norm to return that AOI region, e.g. sig_norm[:, AOI[0], AOI[1]].

    roi_avg_fit_result : `QDMPy.fit_shared.ROIAvgFitResult`
        `QDMPy.fit_shared.ROIAvgFitResult` object, to pull `QDMPy.fitting.FitResultROIAvg.fit_options`
        from.

    Returns
    -------
    fit_result_collection : `QDMPy.fit_shared.FitResultCollection` object
    """

    fit_result_collection_lst = []  # list of FitResultCollection objects
    # iterate through all possible fit backend choices
    for n, fit_backend in enumerate(options["fit_backend_comparison"]):
        if fit_backend == "scipy":
            fit_result_collection_lst.append(
                fit_scipy.fit_AOIs_scipy(
                    options,
                    sig_norm,
                    single_pixel_pl,
                    sweep_list,
                    fit_model,
                    AOIs,
                    backend_ROI_results_lst[n],
                ),
            )
        elif fit_backend == "gpufit":
            fit_result_collection_lst.append(
                fit_gpufit.fit_AOIs_gpufit(
                    options,
                    sig_norm,
                    single_pixel_pl,
                    sweep_list,
                    fit_model,
                    AOIs,
                    backend_ROI_results_lst[n],
                )
            )
        else:
            raise RuntimeError(
                f"No fit_AOIs fn defined for fit_backend = {options['fit_backend']}"
            )
    return fit_result_collection_lst


# ============================================================================


def fit_pixels(options, sig_norm, sweep_list, fit_model, roi_avg_fit_result):
    """
    Fit all pixels in image with chosen fit backend.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    sig_norm : np array, 3D
        Normalised measurement array, shape: [sweep_list, x, y].

    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)

    fit_model : `fit_models.FitModel` object.

    roi_avg_fit_result : `fitting.FitResultROIAvg`
        `fitting.FitResultROIAvg` object, to pull fit_options from.

    Returns
    -------
    fit_result_dict : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    """

    # here only use only chosen backend!
    if options["fit_backend"] == "scipy":
        return fit_scipy.fit_pixels_scipy(
            options, sig_norm, sweep_list, fit_model, roi_avg_fit_result
        )
    elif options["fit_backend"] == "gpufit":
        return fit_gpufit, fit_pixels.gpufit(
            options, sig_norm, sweep_list, fit_model, roi_avg_fit_result
        )
    else:
        raise RuntimeError(f"No fit_pixels fn defined for fit_backend = {options['fit_backend']}")

# -*- coding: utf-8 -*-
"""
This module holds tools for fitting raw data via scipy. (scipy backend)

Functions
---------
 - `QDMPy.fit_scipy.prep_scipy_fit_options`
 - `QDMPy.fit_scipy.gen_scipy_init_guesses`
 - `QDMPy.fit_scipy.fit_ROI_avg_scipy`
 - `QDMPy.fit_scipy.fit_single_pixel_scipy`
 - `QDMPy.fit_scipy.fit_AOIs_scipy`
 - `QDMPy.fit_scipy.limit_cpu`
 - `QDMPy.fit_scipy.to_squares_wrapper`
 - `QDMPy.fit_scipy.fit_pixels_scipy`
"""

# ============================================================================

__author__ = "Sam Scholten"

# ==========================================================================


import numpy as np
from scipy.optimize import least_squares
from tqdm.autonotebook import tqdm  # auto detects jupyter
import psutil
import os
import concurrent.futures
from itertools import repeat
import warnings
from sys import platform

# ============================================================================

import QDMPy.systems as systems
import QDMPy.fit_models as fit_models
import QDMPy.fit_shared as fit_shared
import QDMPy.fit_interface as fit_interface

# ==========================================================================

# NOTE nee prepare_fit_options
def prep_scipy_fit_options(options, fit_model):
    """
    General options dict -> scipy_fit_options, init_param_guesses
    in format that scipy least_squares expects.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    fit_model : `fit_models.FitModel` object.

    Returns
    -------
    scipy_fit_options : dict
        Dictionary with options that scipy.optimize.least_squares expects, specific to fitting.

    init_param_guess : np array, 1D (shape: num_params)
        Array of parameter values as 'initial guess' of fit model.
    """

    # this is just constructing the initial parameter guesses and bounds in the right format
    fit_param_ar, fit_param_bound_ar = gen_scipy_init_guesses(
        options, *fit_shared.gen_init_guesses(options)
    )
    init_param_guess = fit_param_ar
    fit_bounds = (fit_param_bound_ar[:, 0], fit_param_bound_ar[:, 1])

    # see scipy.optimize.least_squares
    scipy_fit_options = {
        "method": options["fit_method"],
        "verbose": options["verbose_fitting"],
        "gtol": options["fit_gtol"],
        "xtol": options["fit_xtol"],
        "ftol": options["fit_ftol"],
        "loss": options["loss_fn"],
    }

    if options["fit_method"] != "lm":
        scipy_fit_options["bounds"] = fit_bounds
        scipy_fit_options["verbose"] = options["verbose_fitting"]

    if options["scale_x"]:
        scipy_fit_options["x_scale"] = "jac"
    else:
        options["scale_x"] = False

    # define jacobian option for least_squares fitting
    if fit_model.jacobian_scipy is None or not options["use_analytic_jac"]:
        scipy_fit_options["jac"] = options["fit_jac_acc"]
    else:
        scipy_fit_options["jac"] = fit_model.jacobian_scipy
    return scipy_fit_options, init_param_guess


# ==========================================================================


def gen_scipy_init_guesses(options, init_guesses, init_bounds):
    """
    Generate arrays of initial fit guesses and bounds in correct form for scipy least_squares.

    init_guesses and init_bounds are dictionaries up to this point, we now convert to np arrays,
    that scipy will recognise. In particular, we specificy that each of the 'num' of each 'fn_type'
    have independent parameters, so must have independent init_guesses and init_bounds when
    plugged into scipy.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    init_guesses : dict
        Dict holding guesses for each parameter, e.g. key -> list of guesses for each independent
        version of that fn_type.

    init_bounds : dict
        Dict holding guesses for each parameter, e.g. key -> list of bounds for each independent
        version of that fn_type.

    Returns
    -------
    fit_param_ar : np array, shape: num_params

    fi_param_bound_ar : np array, shape: (num_params, 2)
    """
    param_lst = []
    bound_lst = []

    for fn_type, num in options["fit_functions"].items():
        # extract a guess/bounds for each of the copies of each fn_type (e.g. 8 lorentzians)
        for n in range(num):

            for pos, key in enumerate(fit_models.AVAILABLE_FNS[fn_type].param_defn):
                # this check is to handle the edge case of guesses/bounds
                # options being provided as numbers rather than lists of numbers
                try:
                    param_lst.append(init_guesses[key][n])
                except (TypeError, KeyError):
                    param_lst.append(init_guesses[key])
                if len(np.array(init_bounds[key]).shape) == 2:
                    bound_lst.append(init_bounds[key][n])
                else:
                    bound_lst.append(init_bounds[key])

    fit_param_ar = np.array(param_lst)
    fit_param_bound_ar = np.array(bound_lst)
    return fit_param_ar, fit_param_bound_ar


# ==========================================================================


def fit_ROI_avg_scipy(options, sig_norm, sweep_list, fit_model):
    """
    Fit the average of the measurement over the region of interest specified, with scipy.

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
    `QDMPy.fitting.FitResultROIAvg` object containing the fit result (see class specifics)
    """

    systems.clean_options(options)

    # fit *all* pl data (i.e. summing over FOV)
    # collapse to just pl_ar (as function of sweep, 1D)
    pl_roi = np.nanmean(np.nanmean(sig_norm, axis=2), axis=1)

    fit_options, init_param_guess = prep_scipy_fit_options(options, fit_model)

    fitting_results = least_squares(
        fit_model.residuals_scipy, init_param_guess, args=(sweep_list, pl_roi), **fit_options
    )

    best_params = fitting_results.x
    # fit_sweep_vector = np.linspace(np.min(sweep_list), np.max(sweep_list), 10000)
    # scipy_best_fit = fit_model(best_params, fit_sweep_vector)
    # init_fit = fit_model(init_guess, fit_sweep_vector)

    # TODO how to handle this result, if we have it returned from multiple methods?
    return fit_shared.ROIAvgFitResult(
        "scipy", fit_options, fit_model, pl_roi, sweep_list, best_params, init_param_guess
    )


# ==========================================================================


def fit_single_pixel_scipy(options, pixel_pl_ar, sweep_list, fit_model, roi_avg_fit_result):
    """
    Fit Single pixel and return best_fit_result.x (i.e. the optimal fit parameters)

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    pixel_pl_ar : np array, 1D
        Normalised PL as function of sweep_list for a single pixel.

    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)

    fit_model : `QDMPy.fit_models.FitModel` object.

    roi_avg_fit_result : `QDMPy.fit_shared.ROIAvgFitResult`
        `QDMPy.fit_shared.ROIAvgFitResult` object, to pull fit_options from.

    Returns
    -------
    pixel_parameters : np array, 1D
        Best fit parameters, as determined by scipy.
    """

    fit_opts = {}
    for key, val in roi_avg_fit_result.fit_options.items():
        fit_opts[key] = val

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        init_guess_params = roi_avg_fit_result.best_params.copy()
    else:
        # this is just constructing the initial parameter guesses and bounds in the right format
        fit_param_ar, _ = gen_scipy_init_guesses(options, *fit_shared.gen_init_guesses(options))
        init_guess_params = fit_param_ar.copy()

    fitting_results = least_squares(
        fit_model.residuals_scipy, init_guess_params, args=(sweep_list, pixel_pl_ar), **fit_opts
    )
    return fitting_results.x


# ==========================================================================


def fit_AOIs_scipy(
    options, sig_norm, pixel_pl_ar, sweep_list, fit_model, AOIs, roi_avg_fit_result
):
    """
    Fit AOIs and single pixel and return list of (list of) fit results (optimal fit parameters),
    using scipy.

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

    systems.clean_options(options)

    fit_opts = {}
    for key, val in roi_avg_fit_result.fit_options.items():
        fit_opts[key] = val

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        guess_params = roi_avg_fit_result.best_params.copy()
    else:
        # this is just constructing the initial parameter guesses and bounds in the right format
        fit_param_ar, fit_param_bound_ar = gen_scipy_init_guesses(
            options, *gen_scipy_init_guesses(options)
        )
        guess_params = fit_param_ar.copy()

    single_pixel_fit_params = fit_single_pixel_scipy(
        options, pixel_pl_ar, sweep_list, fit_model, roi_avg_fit_result
    )

    AOI_avg_best_fit_results_lst = []

    for AOI in AOIs:
        aoi_sig_norm = sig_norm[:, AOI[0], AOI[1]]
        aoi_avg = np.nanmean(np.nanmean(aoi_sig_norm, axis=2), axis=1)

        fitting_results = least_squares(
            fit_model.residuals_scipy, guess_params, args=(sweep_list, aoi_avg), **fit_opts
        )
        AOI_avg_best_fit_results_lst.append(fitting_results.x)

    return fit_shared.FitResultCollection(
        "scipy", roi_avg_fit_result, single_pixel_fit_params, AOI_avg_best_fit_results_lst
    )


# ==========================================================================


def limit_cpu():
    """Called at every process start, to reduce the priority of this process"""
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use p.nice(19)
    if platform.startswith("linux"):  # linux
        p.nice(19)
    elif platform.startswith("win32"):  # windows
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    elif platform.startswith("darwin"):  # macOS
        warnings.warn("Not sure what to use for macOS... skipping cpu limitting")
    else:  # 'freebsd', 'aix', 'cygwin'...
        warnings.warn(f"Not sure what to use for your OS: {platform}... skipping cpu limitting")


# ==========================================================================


def to_squares_wrapper(fun, p0, sweep_vec, shaped_data, kwargs={}):
    """
    Simple wrapper of scipy.optimize.least_squares to allow us to keep track of which
    solution is which (or where).

    Arguments
    ---------
    fun : function
        Function object acting as residual (fit model minus pl value)

    p0 : np array
        Initial guess: array of parameters

    sweep_vec : np array
        Array (or I guess single value, anything iterable) of affine parameter (tau/freq)

    shaped_data : list (3 elements)
        array returned by `QDMPy.fit_shared.pixel_generator`: [x, y, sig_norm[:, x, y]]

    kwargs : dict
        Other options (dict) passed to least_squares, i.e. fit_options

    Returns
    -------
    (x, y), least_squares(...).x
        I.e. the position of the fit result, and then the fit result parameters array.
    """
    # shaped_data: [x, y, pl]
    # output: (x, y), result_params
    return (
        (shaped_data[0], shaped_data[1]),
        least_squares(fun, p0, args=(sweep_vec, shaped_data[2]), **kwargs).x,
    )


# ==========================================================================


def fit_pixels_scipy(options, sig_norm, sweep_list, fit_model, roi_avg_fit_result):
    """
    Fits each pixel and returns dictionary of param_name -> param_image.

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
    systems.clean_options(options)

    sweep_ar = np.array(sweep_list)
    threads = options["threads"]
    num_pixels = np.shape(sig_norm)[1] * np.shape(sig_norm)[2]

    # this makes low binning work (idk why), else do chunksize = 1
    chunksize = int(num_pixels / (threads * 100))

    # randomize order of fitting pixels (will un-scramble later) so ETA is more correct
    if options["scramble_pixels"]:
        pixel_data, unshuffler = fit_shared.shuffle_pixels(sig_norm)
    else:
        pixel_data = sig_norm

    if not chunksize:
        warnings.warn("chunksize was 0, setting to 1")
        chunksize = 1

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        init_guess_params = roi_avg_fit_result.best_params.copy()
    else:
        # this is just constructing the initial parameter guesses and bounds in the right format
        init_guess_params, _ = gen_scipy_init_guesses(
            options, *fit_shared.gen_init_guesses(options)
        )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=threads, initializer=limit_cpu
    ) as executor:
        fit_results = list(
            tqdm(
                executor.map(
                    to_squares_wrapper,
                    repeat(fit_model.residuals_scipy),
                    repeat(init_guess_params),
                    repeat(sweep_ar),
                    fit_shared.pixel_generator(pixel_data),
                    repeat(roi_avg_fit_result.fit_options.copy()),
                    chunksize=chunksize,
                ),
                ascii=True,
                mininterval=1,
                total=num_pixels,
                unit=" PX",
                disable=(not options["show_progressbar"]),
            )
        )

    roi_shape = np.shape(sig_norm)
    res = fit_interface.get_pixel_fitting_results(
        fit_model, fit_results, (roi_shape[1], roi_shape[2])
    )
    if options["scramble_pixels"]:
        return fit_shared.unshuffle_fit_results(res, unshuffler)
    else:
        return res

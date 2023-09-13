# -*- coding: utf-8 -*-
"""
This module holds tools for fitting raw data via scipy. (scipy backend)

Functions
---------
 - `qdmpy.pl.scipyfit.prep_scipyfit_options`
 - `qdmpy.pl.scipyfit.gen_scipyfit_init_guesses`
 - `qdmpy.pl.scipyfit.fit_roi_avg_pl_scipyfit`
 - `qdmpy.pl.scipyfit.fit_single_pixel_pl_scipyfit`
 - `qdmpy.pl.scipyfit.fit_aois_pl_scipyfit`
 - `qdmpy.pl.scipyfit.limit_cpu`
 - `qdmpy.pl.scipyfit.to_squares_wrapper`
 - `qdmpy.pl.scipyfit.fit_all_pixels_pl_scipyfit`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.pl.scipyfit.prep_scipyfit_options": True,
    "qdmpy.pl.scipyfit.gen_scipyfit_init_guesses": True,
    "qdmpy.pl.scipyfit.fit_roi_avg_pl_scipyfit": True,
    "qdmpy.pl.scipyfit.fit_single_pixel_pl_scipyfit": True,
    "qdmpy.pl.scipyfit.fit_aois_pl_scipyfit": True,
    "qdmpy.pl.scipyfit.limit_cpu": True,
    "qdmpy.pl.scipyfit.to_squares_wrapper": True,
    "qdmpy.pl.scipyfit.fit_all_pixels_pl_scipyfit": True,
}

# ==========================================================================

import numpy as np
from scipy.optimize import least_squares
from tqdm.autonotebook import tqdm  # auto detects jupyter
import psutil
import os
import concurrent.futures
from itertools import repeat
from sys import platform
from timeit import default_timer as timer
from datetime import timedelta

# ============================================================================

import qdmpy.pl.common
import qdmpy.pl.funcs
from qdmpy.shared.misc import warn

# ==========================================================================


def prep_scipyfit_options(options, fit_model):
    """
    General options dict -> scipyfit_options
    in format that scipy least_squares expects.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    fit_model : `qdmpy.pl.model.FitModel`
        Fit model object.

    Returns
    -------
    scipy_fit_options : dict
        Dictionary with options that scipy.optimize.least_squares expects, specific to fitting.
    """

    # this is just constructing the initial parameter guesses and bounds in the right format
    _, fit_param_bound_ar = gen_scipyfit_init_guesses(
        options, *qdmpy.pl.common.gen_init_guesses(options)
    )
    fit_bounds = (fit_param_bound_ar[:, 0], fit_param_bound_ar[:, 1])

    # see scipy.optimize.least_squares
    scipyfit_options = {
        "method": options["scipyfit_method"],
        "verbose": options["scipyfit_verbose_fitting"],
        "gtol": options["scipyfit_fit_gtol"],
        "xtol": options["scipyfit_fit_xtol"],
        "ftol": options["scipyfit_fit_ftol"],
        "loss": options["scipyfit_loss_fn"],
    }

    if options["scipyfit_method"] != "lm":
        scipyfit_options["bounds"] = fit_bounds
        scipyfit_options["verbose"] = options["scipyfit_verbose_fitting"]

    if options["scipyfit_scale_x"]:
        scipyfit_options["x_scale"] = "jac"
    else:
        options["scipyfit_scale_x"] = False

    # define jacobian option for least_squares fitting
    if not fit_model.jacobian_defined() or not options["scipyfit_use_analytic_jac"]:
        scipyfit_options["jac"] = options["scipyfit_fit_jac_acc"]
    else:
        scipyfit_options["jac"] = fit_model.jacobian_scipyfit
    return scipyfit_options


# ==========================================================================


def gen_scipyfit_init_guesses(options, init_guesses, init_bounds):
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
        The initial fit parameter guesses.
    fit_param_bound_ar : np array, shape: (num_params, 2)
        Fit parameter bounds.
    """
    param_lst = []
    bound_lst = []

    for fn_type, num in options["fit_functions"].items():
        # extract a guess/bounds for each of the copies of each fn_type (e.g. 8 lorentzians)
        for n in range(num):

            for pos, key in enumerate(qdmpy.pl.funcs.AVAILABLE_FNS[fn_type].param_defn):
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


def fit_roi_avg_pl_scipyfit(options, sig, ref, sweep_list, fit_model):
    """
    Fit the average of the measurement over the region of interest specified, with scipy.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    sig : np array, 3D
        Sig measurement array, unnormalised, shape: [sweep_list, y, x].
    ref : np array, 3D
        Ref measurement array, unnormalised, shape: [sweep_list, y, x].
    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : `qdmpy.pl.model.FitModel`
        The fit model object.

    Returns
    -------
    `qdmpy.pl.common.ROIAvgFitResult` object containing the fit result (see class specifics)
    """
    # fit *all* pl data (i.e. summing over FOV)
    # collapse to just pl_ar (as function of sweep, 1D)
    sig_mean = np.nanmean(sig, axis=(1, 2))
    ref_mean = np.nanmean(ref, axis=(1, 2))
    if not options["used_ref"]:
        roi_norm = sig_mean
    elif options["normalisation"] == "div":
        roi_norm = sig_mean / ref_mean
    elif options["normalisation"] == "sub":
        roi_norm = 1 + (sig_mean - ref_mean) / (sig_mean + ref_mean)
    elif options["normalisation"] == "true_sub":
        roi_norm = (sig_mean - ref_mean) / np.nanmax(sig_mean - ref_mean)

    fit_options = prep_scipyfit_options(options, fit_model)

    init_param_guess, init_bounds = gen_scipyfit_init_guesses(
        options, *qdmpy.pl.common.gen_init_guesses(options)
    )
    fitting_results = least_squares(
        fit_model.residuals_scipyfit,
        init_param_guess,
        args=(sweep_list, roi_norm),
        **fit_options,
    )

    best_params = fitting_results.x
    return qdmpy.pl.common.ROIAvgFitResult(
        "scipyfit",
        fit_options,
        fit_model,
        roi_norm,
        sweep_list,
        best_params,
        init_param_guess,
        init_bounds,
    )


# ==========================================================================


def fit_single_pixel_pl_scipyfit(
    options, pixel_pl_ar, sweep_list, fit_model, roi_avg_fit_result
):
    """
    Fit Single pixel and return best_fit_result.x (i.e. the optimal fit parameters)

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    pixel_pl_ar : np array, 1D
        Normalised pl as function of sweep_list for a single pixel.
    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : `qdmpy.pl.model.FitModel`
        The fit model.
    roi_avg_fit_result : `qdmpy.pl.common.ROIAvgFitResult`
        `qdmpy.pl.common.ROIAvgFitResult` object, to pull fit_options from.

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
        fit_param_ar, _ = gen_scipyfit_init_guesses(
            options, *qdmpy.pl.common.gen_init_guesses(options)
        )
        init_guess_params = fit_param_ar.copy()

    fitting_results = least_squares(
        fit_model.residuals_scipyfit,
        init_guess_params,
        args=(sweep_list, pixel_pl_ar),
        **fit_opts,
    )
    return fitting_results.x


# ==========================================================================


def fit_aois_pl_scipyfit(
    options,
    sig,
    ref,
    pixel_pl_ar,
    sweep_list,
    fit_model,
    aois,
    roi_avg_fit_result,
):
    """
    Fit AOIs and single pixel and return list of (list of) fit results (optimal fit parameters),
    using scipy.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    sig : np array, 3D
        Sig measurement array, unnormalised, shape: [sweep_list, y, x].
    ref : np array, 3D
        Ref measurement array, unnormalised, shape: [sweep_list, y, x].
    single_pixel_pl : np array, 1D
        Normalised measurement array, for chosen single pixel check.
    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq).
    fit_model : `qdmpy.pl.model.FitModel`
        The model we're fitting to.
    aois : list
        List of AOI specifications - each a length-2 iterable that can be used to directly index
        into sig_norm to return that AOI region, e.g. sig_norm[:, AOI[0], AOI[1]].
    roi_avg_fit_result : `qdmpy.pl.common.ROIAvgFitResult`
        `qdmpy.pl.common.ROIAvgFitResult` object, to pull `qdmpy.pl.common.ROIAvgFitResult.fit_options`
        from.

    Returns
    -------
    fit_result_collection : `qdmpy.pl.common.FitResultCollection`
        Collection of ROI/AOI fit results for this fit backend.
    """
    fit_opts = {}
    for key, val in roi_avg_fit_result.fit_options.items():
        fit_opts[key] = val

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        guess_params = roi_avg_fit_result.best_params.copy()
    else:
        # this is just constructing the initial parameter guesses and bounds in the right format
        fit_param_ar, _ = gen_scipyfit_init_guesses(
            options, *qdmpy.pl.common.gen_init_guesses(options)
        )
        guess_params = fit_param_ar.copy()

    single_pixel_fit_params = fit_single_pixel_pl_scipyfit(
        options, pixel_pl_ar, sweep_list, fit_model, roi_avg_fit_result
    )

    aoi_avg_best_fit_results_lst = []

    for a in aois:
        this_sig = np.nanmean(sig[:, a[0], a[1]], axis=(1, 2))
        this_ref = np.nanmean(ref[:, a[0], a[1]], axis=(1, 2))

        if not options["used_ref"]:
            this_aoi = this_sig
        elif options["normalisation"] == "div":
            this_aoi = this_sig / this_ref
        elif options["normalisation"] == "sub":
            this_aoi = 1 + (this_sig - this_ref) / (this_sig + this_ref)
        elif options["normalisation"] == "true_sub":
            this_aoi = (this_sig - this_ref) / np.nanmax(this_sig - this_ref)

        fitting_results = least_squares(
            fit_model.residuals_scipyfit,
            guess_params,
            args=(sweep_list, this_aoi),
            **fit_opts,
        )
        aoi_avg_best_fit_results_lst.append(fitting_results.x)

    return qdmpy.pl.common.FitResultCollection(
        "scipyfit",
        roi_avg_fit_result,
        single_pixel_fit_params,
        aoi_avg_best_fit_results_lst,
    )


# ==========================================================================


def limit_cpu():
    """Called at every process start, to reduce the priority of this process"""
    p = psutil.Process(os.getpid())
    # set to lowest priority
    if platform.startswith("linux"):  # linux
        p.nice(19)
    elif platform.startswith("win32"):  # windows
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    elif platform.startswith("darwin"):  # macOS
        warn("Not sure what to use for macOS... skipping cpu limitting")
    else:  # 'freebsd', 'aix', 'cygwin'...
        warn(f"Not sure what to use for your OS: {platform}... skipping cpu limitting")


# ==========================================================================


def to_squares_wrapper(fun, p0, sweep_vec, shaped_data, fit_optns):
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
        array returned by `qdmpy.pl.common.pixel_generator`: [y, x, sig_norm[:, y, x]]
    fit_optns : dict
        Other options (dict) passed to least_squares

    Returns
    -------
    wrapped_squares : tuple
        (y, x), least_squares(...).x, leas_squares(...).jac
        I.e. the position of the fit result, the fit result parameters array, jacobian at solution
    """
    # shaped_data: [y, x, pl]
    # output: (y, x), result_params, jac
    fitres = least_squares(fun, p0, args=(sweep_vec, shaped_data[2]), **fit_optns)
    return ((shaped_data[0], shaped_data[1]), fitres.x, fitres.jac)


# ==========================================================================


def fit_all_pixels_pl_scipyfit(
    options, sig_norm, sweep_list, fit_model, roi_avg_fit_result
):
    """
    Fits each pixel and returns dictionary of param_name -> param_image.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    sig_norm : np array, 3D
        Normalised measurement array, shape: [sweep_list, y, x].
    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : `qdmpy.pl.model.FitModel`
        The model we're fitting to.
    roi_avg_fit_result : `qdmpy.pl.common.ROIAvgFitResult`
        `qdmpy.pl.common.ROIAvgFitResult` object, to pull fit_options from.

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    """
    sweep_ar = np.array(sweep_list)
    threads = options["threads"]
    num_pixels = np.shape(sig_norm)[1] * np.shape(sig_norm)[2]

    # this makes low binning work (idk why), else do chunksize = 1
    chunksize = int(num_pixels / (threads * 100))

    # randomize order of fitting pixels (will un-scramble later) so ETA is more correct
    if options["scramble_pixels"]:
        pixel_data, unshuffler = qdmpy.pl.common.shuffle_pixels(sig_norm)
    else:
        pixel_data = sig_norm

    if not chunksize:
        warn("chunksize was 0, setting to 1")
        chunksize = 1

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        init_guess_params = roi_avg_fit_result.best_params.copy()
    else:
        # this is just constructing the initial parameter guesses and bounds in the right format
        init_guess_params, _ = gen_scipyfit_init_guesses(
            options, *qdmpy.pl.common.gen_init_guesses(options)
        )
    # call into the library (measure time)
    t0 = timer()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=threads, initializer=limit_cpu
    ) as executor:
        fit_results = list(
            tqdm(
                executor.map(
                    to_squares_wrapper,
                    repeat(fit_model.residuals_scipyfit),
                    repeat(init_guess_params),
                    repeat(sweep_ar),
                    qdmpy.pl.common.pixel_generator(pixel_data),
                    repeat(roi_avg_fit_result.fit_options.copy()),
                    chunksize=chunksize,
                ),
                desc="pl-scipyfit",
                ascii=True,
                mininterval=1,
                total=num_pixels,
                unit=" PX",
                disable=(not options["scipyfit_show_progressbar"]),
            )
        )
    t1 = timer()
    # for the record
    options["fit_time_(s)"] = timedelta(seconds=t1 - t0).total_seconds()

    res, sigmas = qdmpy.pl.common.get_pixel_fitting_results(
        fit_model, fit_results, pixel_data, sweep_ar
    )
    if options["scramble_pixels"]:
        res = qdmpy.pl.common.unshuffle_fit_results(res, unshuffler)
        sigmas = qdmpy.pl.common.unshuffle_fit_results(sigmas, unshuffler)

    return res, sigmas

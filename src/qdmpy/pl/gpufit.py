# -*- coding: utf-8 -*-
"""
This module holds tools for fitting raw data via gpufit. (gpufit backend)

Functions
---------
 - `qdmpy.pl.gpufit.prep_gpufit_fit_options`
 - `qdmpy.pl.gpufit.get_gpufit_modelID`
 - `qdmpy.pl.gpufit.prep_gpufit_backend`
 - `qdmpy.pl.gpufit.gen_gpufit_init_guesses`
 - `qdmpy.pl.gpufit.fit_single_pixel_pl_gpufit`
 - `qdmpy.pl.gpufit.fit_roi_avg_pl_gpufit`
 - `qdmpy.pl.gpufit.fit_aois_pl_gpufit`
 - `qdmpy.pl.gpufit.fit_pl_pixels_gpufit`
 - `qdmpy.pl.gpufit.gpufit_data_shape`
 - `qdmpy.pl.gpufit.gpufit_reshape_result`
 - `qdmpy.pl.gpufit.get_params_to_fit`

"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.pl.gpufit.prep_gpufit_fit_options": True,
    "qdmpy.pl.gpufit.get_gpufit_modelID": True,
    "qdmpy.pl.gpufit.prep_gpufit_backend": True,
    "qdmpy.pl.gpufit.gen_gpufit_init_guesses": True,
    "qdmpy.pl.gpufit.fit_single_pixel_pl_gpufit": True,
    "qdmpy.pl.gpufit.fit_roi_avg_pl_gpufit": True,
    "qdmpy.pl.gpufit.fit_aois_pl_gpufit": True,
    "qdmpy.pl.gpufit.fit_all_pixels_pl_gpufit": True,
    "qdmpy.pl.gpufit.gpufit_data_shape": True,
    "qdmpy.pl.gpufit.gpufit_reshape_result": True,
    "qdmpy.pl.gpufit.get_params_to_fit": True,
}

# ============================================================================

import pygpufit.gpufit as gf
import numpy as np
from warnings import warn

# ============================================================================

import qdmpy.pl.model
import qdmpy.pl.common
import qdmpy.pl.funcs

# ============================================================================


def prep_gpufit_fit_options(options):
    """
    General options dict -> gpufit_fit_options
    in format that scipy least_squares expects.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    gpufit_fit_options : dict
        Dictionary with args that gpufit expects (i.e. expanded by **gpufit_fit_options).
    """

    gpufit_fit_options = {
        "tolerance": options["gpufit_tolerance"],
        "max_number_iterations": options["gpufit_max_iterations"],
    }

    if options["gpufit_estimator_id"] == "LSE":
        gpufit_fit_options["estimator_id"] = gf.EstimatorID.LSE
    elif options["gpufit_estimator_id"] == "MLE":
        gpufit_fit_options["estimator_id"] = gf.EstimatorID.MLE
    else:
        raise RuntimeError(
            "Didn't know what to do with 'gpfit_estimator_id' ="
            f" {options['gpufit_estimator_id']}" + " available options: 'LSE', 'MLE'"
        )
    return gpufit_fit_options


# ============================================================================


def get_gpufit_modelID(options, fit_model):  # noqa: N802
    """
    Find corresponding gpufit modelID for this fitmodel

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    fit_model : `qdmpy.pl.model.FitModel`
        for odmr, LORENTZ8, one of:
            {'linear': 1, 'lorentzian': 1<=n<=8}
            {'constant': 1, 'lorentzian': 1<=n<=8}
        for t1/etc., STRETCHED_EXP: {'constant': 1, 'stretched_exponential': 1}

    Returns
    -------
    ModelID : int
        Defined through a pygpufit.gpufit.ModelID object (essentially an enum).

        Model ID used by gpufit to specify fit model.
        Check pygpufit/gpufit.py for class/enum.
        Currently defined in gpufit: LORENTZ8 and STRETCHED_EXP

    """

    ffs = fit_model.fit_functions
    model = None
    for i in range(8):
        if ffs == {"constant": 1, "lorentzian": i + 1}:
            model = gf.ModelID.LORENTZ8_CONST
            break
        elif ffs == {"linear": 1, "lorentzian": i + 1}:
            model = gf.ModelID.LORENTZ8_LINEAR
            break
    if ffs in [{"constant": 1, "stretched_exponential": 1}]:
        model = gf.ModelID.STRETCHED_EXP
    if ffs in [{"constant": 1, "damped_rabi": 1}]:
        model = gf.ModelID.DAMPED_RABI

    if model is None:
        raise RuntimeError(
            "No gpufit modelID found for those fit_functions.\n"
            + "Available fit_functions:\n"  # noqa: W503
            + "LORENTZ8 (one of): \n"
            + "\t{'linear': 1, 'lorentzian': 1<=n<=8}\n"  # noqa: W503
            + "\t{'constant': 1, 'lorentzian': 1<=n<=8}\n"  # noqa: W503
            + "\t{'lorentzian': 1<=n<=8}\n"  # noqa: W503
            + "STRETCHED_EXP: \n"
            + "\t{'constant': 1, 'stretched_exponential': 1}"  # noqa: W503
            + "DAMPED_RABI: \n"
            + "\t{'constant': 1, 'damped_rabi': 1`}"
        )

    return model


# ============================================================================


def prep_gpufit_backend(options, fit_model):
    """
    Initial preparation of gpufit backend.

    First checks if cuda etc. are installed correctly, then determines
    the ModelID associated with chosen fit_model.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    fit_model : `qdmpy.pl.model.FitModel`
        Must be one of:
        for odmr, LORENTZ8, one of (in this order!):
            {'linear': 1, 'lorentzian': 1<=n<=8}
            {'constant': 1, 'lorentzian': 1<=n<=8}
        for t1/etc., STRETCHED_EXP: {'constant': 1, 'stretched_exponential': 1}

    """
    if not gf.cuda_available():
        warn(f"CUDA error:\n{gf.get_last_error()}")

    (
        options["CUDA_version_runtime"],
        options["CUDA_version_driver"],
    ) = gf.get_cuda_version()

    options["ModelID"] = get_gpufit_modelID(options, fit_model)


# ============================================================================


def gen_gpufit_init_guesses(options, init_guesses, init_bounds):
    """
    Generate arrays of initial fit guesses and bounds in correct form for gpufit.

    init_guesses and init_bounds are dictionaries up to this point, we now convert to np arrays,
    that gpufit will recognise. In particular, we specificy that each of the 'num' of each 'fn_type'
    have independent parameters, so must have independent init_guesses and init_bounds.

    Slightly differently to scipy, just in the format of the init_bounds. Also need to fill
    arrays up to 8 lorentzian peaks even if not fitting them all.

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
    fi_param_bound_ar : np array, shape: 2 * num_params
        Format: [parameter 1 lower bound, parameter 1 upper bound, parameter 2 lower bounds, ...]
    """
    param_lst = []
    bound_lst = []

    for fn_type, num in options["fit_functions"].items():
        # extract a guess/bounds for each of the copies of each fn_type (e.g. 8 lorentzians)
        if fn_type == "lorentzian":
            num_fns_required = 8
        else:
            num_fns_required = 1

        for n in range(num_fns_required):
            if n < num:
                for pos, key in enumerate(
                    qdmpy.pl.funcs.AVAILABLE_FNS[fn_type].param_defn
                ):
                    # these checks here are to handling the edge case of guesses/bounds
                    # options being provided as numbers rather than lists of numbers
                    try:
                        param_lst.append(init_guesses[key][n])
                    except (TypeError, KeyError):
                        param_lst.append(init_guesses[key])
                    if len(np.array(init_bounds[key]).shape) == 2:
                        bound_lst.append(init_bounds[key][n][0])
                        bound_lst.append(init_bounds[key][n][1])
                    else:
                        bound_lst.append(init_bounds[key][0])
                        bound_lst.append(init_bounds[key][1])
            else:
                # insert guesses and bounds for params we won't fit. (gpufit requires full array)
                for pos, key in enumerate(
                    qdmpy.pl.funcs.AVAILABLE_FNS[fn_type].param_defn
                ):
                    param_lst.append(0)
                    bound_lst.append(0)
                    bound_lst.append(1)

    fit_param_ar = np.array(param_lst)
    fit_param_bound_ar = np.array(bound_lst)
    return fit_param_ar, fit_param_bound_ar


# ============================================================================


def fit_single_pixel_pl_gpufit(
    options, pixel_pl_ar, sweep_list, fit_model, roi_avg_fit_result
):
    """
    Fit Single pixel and return optimal fit parameters with gpufit backend

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    pixel_pl_ar : np array, 1D
        Normalised pl as function of sweep_list for a single pixel.
    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : `qdmpy.pl.model.FitModel`
        Model we're fitting to.
    roi_avg_fit_result : `qdmpy.pl.common.ROIAvgFitResult`
        `qdmpy.pl.common.ROIAvgFitResult` object, to pull fit_options from.

    Returns
    -------
    pixel_parameters : np array, 1D
        Best fit parameters, as determined by gpufit
    """
    # NOTE we need to do the fit at least twice (gpufit requirements) so we do it
    # (indentically) twice here, and then disregard one result.
    # NB this isn't true, see the pygpufit examples, it just needs to be 2D
    # -- but fine leave this as is

    # this is just constructing the initial parameter guesses and bounds in the right format
    init_guess_params, init_bounds = gen_gpufit_init_guesses(
        options, *qdmpy.pl.common.gen_init_guesses(options)
    )

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        init_guess_params = roi_avg_fit_result.best_params.copy()

    # only fit the params we want to :)
    params_to_fit = get_params_to_fit(options, fit_model)

    params_to_fit = np.array(params_to_fit, dtype=np.int32)
    init_guess_params = np.repeat([init_guess_params], repeats=2, axis=0)
    init_guess_params = init_guess_params.astype(dtype=np.float32)

    # constraints needs to be reshaped too
    constraints = np.repeat([init_bounds], repeats=2, axis=0)
    constraints = constraints.astype(dtype=np.float32)

    pixel_pl_ar_doubled = np.repeat([pixel_pl_ar], repeats=2, axis=0)

    (
        parameters,
        states,
        chi_squares,
        number_iterations,
        execution_time,
    ) = gf.fit_constrained(
        pixel_pl_ar_doubled.astype(np.float32),
        None,
        options["ModelID"],
        init_guess_params,
        constraints=constraints,
        constraint_types=np.array(
            [gf.ConstraintType.LOWER_UPPER for i in range(len(params_to_fit))],
            dtype=np.int32,
        ),
        user_info=np.array(sweep_list, dtype=np.float32),
        parameters_to_fit=params_to_fit,
    )

    return parameters[0, :]  # only take one of the parameter results


# ============================================================================


def fit_roi_avg_pl_gpufit(options, sig, ref, sweep_list, fit_model):
    """
    Fit the average of the measurement over the region of interest specified, with gpufit.

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
        Model we're fitting to.

    Returns
    -------
    result : `qdmpy.pl.common.ROIAvgFitResult`
        object containing the fit result (see class specifics)
    """

    # note need to do the fit at least twice (gpufit requirements) so we do it twice here.

    # fit *all* pl data (i.e. summing over FOV)
    # collapse to just pl_ar (as function of sweep, 1D)
    if not options["used_ref"]:
        roi_norm = sig
    elif options["normalisation"] == "div":
        roi_norm = sig / ref
    elif options["normalisation"] == "sub":
        roi_norm = 1 + (sig - ref) / (sig + ref)
    elif options["normalisation"] == "true_sub":
        roi_norm = (sig - ref) / np.nanmax(sig - ref)

    roi_norm = np.nanmean(roi_norm, axis=(1,2))

    roi_norm_twice = np.repeat([roi_norm], repeats=2, axis=0)

    # this is just constructing the initial parameter guesses and bounds in the right format
    init_guess_params, init_bounds = gen_gpufit_init_guesses(
        options, *qdmpy.pl.common.gen_init_guesses(options)
    )

    gpufit_fit_options = prep_gpufit_fit_options(options)

    # only fit the params we want to :)
    params_to_fit = get_params_to_fit(options, fit_model)

    params_to_fit = np.array(params_to_fit, dtype=np.int32)
    init_guess_params = np.repeat([init_guess_params], repeats=2, axis=0)
    init_guess_params = init_guess_params.astype(dtype=np.float32)

    # constraints needs to be reshaped too
    constraints = np.repeat([init_bounds], repeats=2, axis=0)
    constraints = constraints.astype(dtype=np.float32)

    (
        best_params,
        states,
        chi_squares,
        number_iterations,
        execution_time,
    ) = gf.fit_constrained(
        roi_norm_twice.astype(np.float32),
        None,
        options["ModelID"],
        init_guess_params,
        constraints=constraints,
        constraint_types=np.array(
            [gf.ConstraintType.LOWER_UPPER for i in range(len(params_to_fit))],
            dtype=np.int32,
        ),
        user_info=np.array(sweep_list, dtype=np.float32),
        parameters_to_fit=params_to_fit,
        **gpufit_fit_options,
    )
    return qdmpy.pl.common.ROIAvgFitResult(
        "gpufit",
        gpufit_fit_options,
        fit_model,
        roi_norm,
        sweep_list,
        best_params[0, :],  # only take one of the results
        init_guess_params[0],  # return the params un-repeated
        init_bounds,
    )


# ============================================================================


def fit_aois_pl_gpufit(
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
    using gpufit.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    sig : np array, 3D
        Sig measurement array, unnormalised, shape: [sweep_list, y, x].
    ref : np array, 3D
        Ref measurement array, unnormalised, shape: [sweep_list, y, x].
    pixel_pl_ar : np array, 1D
        Normalised measurement array, for chosen single pixel check.
    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq).
    fit_model : `qdmpy.pl.model.FitModel`
        Model we're fitting to.
    aois : list
        List of AOI specifications - each a length-2 iterable that can be used to directly index
        into sig_norm to return that AOI region, e.g. sig_norm[:, AOI[0], AOI[1]].
    roi_avg_fit_result : `qdmpy.pl.common.ROIAvgFitResult`
        `qdmpy.pl.common.ROIAvgFitResult` object, to pull `qdmpy.pl.common.ROIAvgFitResult.fit_options`
        from.

    Returns
    -------
    fit_result_collection : `qdmpy.pl.common.FitResultCollection`
        `qdmpy.pl.common.FitResultCollection` object
    """
    # note need to do the fit at least twice (gpufit requirements) so we do it twice per AOI here.

    # this is just constructing the initial parameter guesses and bounds in the right format
    init_guess_params, init_bounds = gen_gpufit_init_guesses(
        options, *qdmpy.pl.common.gen_init_guesses(options)
    )

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        init_guess_params = roi_avg_fit_result.best_params.copy()

    single_pixel_fit_params = fit_single_pixel_pl_gpufit(
        options, pixel_pl_ar, sweep_list, fit_model, roi_avg_fit_result
    )

    aoi_avg_best_fit_results_lst = []

    # only fit the params we want to :)
    params_to_fit = get_params_to_fit(options, fit_model)

    params_to_fit = np.array(params_to_fit, dtype=np.int32)
    init_guess_params = np.repeat([init_guess_params], repeats=2, axis=0)
    init_guess_params = init_guess_params.astype(dtype=np.float32)

    # constraints needs to be reshaped too
    constraints = np.repeat([init_bounds], repeats=2, axis=0)
    constraints = constraints.astype(dtype=np.float32)

    for a in aois:
        this_sig = sig[:, a[0], a[1]]
        this_ref = ref[:, a[0], a[1]]

        if not options["used_ref"]:
            this_aoi = this_sig
        elif options["normalisation"] == "div":
            this_aoi = this_sig / this_ref
        elif options["normalisation"] == "sub":
            this_aoi = 1 + (this_sig - this_ref) / (this_sig + this_ref)
        elif options["normalisation"] == "true_sub":
            this_aoi = (this_sig - this_ref) / np.nanmax(this_sig - this_ref)

        this_aoi_avg = np.nanmean(this_aoi, axis=(1,2))
        this_aoi_twice = np.repeat([this_aoi_avg], repeats=2, axis=0)

        fitting_results, _, _, _, _ = gf.fit_constrained(
            this_aoi_twice.astype(np.float32),
            None,
            options["ModelID"],
            np.array(init_guess_params, dtype=np.float32),
            constraints=constraints,
            constraint_types=np.array(
                [gf.ConstraintType.LOWER_UPPER for i in range(len(params_to_fit))],
                dtype=np.int32,
            ),
            user_info=np.array(sweep_list, dtype=np.float32),
            parameters_to_fit=params_to_fit,
        )
        aoi_avg_best_fit_results_lst.append(fitting_results[0, :])

    return qdmpy.pl.common.FitResultCollection(
        "gpufit",
        roi_avg_fit_result,
        single_pixel_fit_params,
        aoi_avg_best_fit_results_lst,
    )


# ============================================================================


def fit_all_pixels_pl_gpufit(
    options,
    sig_norm,
    sweep_list,
    fit_model,
    roi_avg_fit_result,
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
        Model we're fitting to.
    roi_avg_fit_result : `qdmpy.pl.common.ROIAvgFitResult`
        `qdmpy.pl.common.ROIAvgFitResult` object, to pull fit_options from.

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    """
    sweep_ar = np.array(sweep_list)
    num_pixels = np.shape(sig_norm)[1] * np.shape(sig_norm)[2]

    # this is just constructing the initial parameter guesses and bounds in the right format
    init_guess_params, init_bounds = gen_gpufit_init_guesses(
        options, *qdmpy.pl.common.gen_init_guesses(options)
    )

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        init_guess_params = roi_avg_fit_result.best_params.copy()

    # need to reshape init_guess_params, one for each pixel
    guess_params = np.array([init_guess_params], dtype=np.float32)
    init_guess_params_reshaped = np.repeat(guess_params, repeats=num_pixels, axis=0)

    # constraints needs to be reshaped too
    constraints = np.repeat([init_bounds], repeats=num_pixels, axis=0)
    constraints = constraints.astype(dtype=np.float32)

    # only fit the params we want to :) {i.e. < 8 peak ODMR fit etc.}
    params_to_fit = get_params_to_fit(options, fit_model)

    # reshape sig_norm in a way that gpufit likes: (number_fits, number_points)
    sig_norm_shaped, pixel_posns = gpufit_data_shape(sig_norm)

    fitting_results, _, _, _, execution_time = gf.fit_constrained(
        sig_norm_shaped,
        None,
        options["ModelID"],
        init_guess_params_reshaped,
        constraints=constraints,
        constraint_types=np.array(
            [gf.ConstraintType.LOWER_UPPER for i in range(len(params_to_fit))],
            dtype=np.int32,
        ),
        user_info=np.array(sweep_list, dtype=np.float32),
        parameters_to_fit=np.array(params_to_fit, dtype=np.int32),
    )
    # for the record
    options["fit_time_(s)"] = execution_time

    # calculate jacobians via scipy as gpufit doesn't return them at soln
    jacs = [
        fit_model.jacobian_scipyfit(param_ar, sweep_ar, None)
        for param_ar in fitting_results
    ]

    fit_results = gpufit_reshape_result(fitting_results, pixel_posns, jacs)

    res, sigmas = qdmpy.pl.common.get_pixel_fitting_results(
        fit_model, fit_results, sig_norm, sweep_ar
    )

    return res, sigmas


# ============================================================================


def gpufit_data_shape(sig_norm):
    """
    Reformats sig_norm into two arrays that are more usable for gpufit.

    Arguments
    ---------
    sig_norm : np array, 3D
        Signal normalised by reference (via subtraction or normalisation, chosen in options),
        reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI.
        Format: [sweep_vals, y, x]

    Returns
    -------
    sig_norm_reshaped : np array
        np.float32, shape: (num_pixels, len(sweep_list)). Shaped as gpufit wants it!
    pixel_posns : list
        List of pixel positions for each rown of sig_norm_reshaped i.e. [(x1, y1), (x2, y2)]
    """
    pixel_posns = []
    sig_norm_reshaped = []
    for y, x, spectrum in qdmpy.pl.common.pixel_generator(sig_norm):
        pixel_posns.append((y, x))
        sig_norm_reshaped.append(spectrum)
    return np.array(sig_norm_reshaped, dtype=np.float32), pixel_posns


# ============================================================================


def gpufit_reshape_result(pixel_param_results, pixel_posns, jacs):
    """
    Mimics `qdmpy.pl.scipyfit.to_squares_wrapper`, so gpufit can use the
    `qdmpy.pl.common.get_pixel_fitting_results` function to get the nice
    usual dict of param result images.

    Arguments
    ---------
    pixel_param_results : np array, 2D
        parameter results as returned from gpufit. Shape: (num fits, num parameters)
    pixel_posns : list
        List of pixel positions (x,y) as returned by `qdmpy.pl.gpufit._gpufit_data_shape`.
        I.e. the position of pixel positions associated with rows of pixel_param_results.
    jacs : np array, 2D
        Same as pixel_param_results but containing jacobian at solution

    Returns
    -------
    fit_results : list
        List of [(y, x), best_fit_parameter, jac] lists
    """
    fit_results = []
    for params, pos, jac in zip(pixel_param_results, pixel_posns, jacs):
        fit_results.append([pos, params, jac])

    return fit_results


# ============================================================================


def get_params_to_fit(options, fit_model):
    if options["ModelID"] in [
        gf.ModelID.LORENTZ8_CONST,
        gf.ModelID.LORENTZ8_LINEAR,
    ]:
        num_lorentzians = options["fit_functions"]["lorentzian"]
        if options["ModelID"] == gf.ModelID.LORENTZ8_CONST:
            params_to_fit = [1 for i in range(3 * num_lorentzians + 1)]  # + 1 for const
            num_params = 25
        elif options["ModelID"] == gf.ModelID.LORENTZ8_LINEAR:
            params_to_fit = [1 for i in range(3 * num_lorentzians + 2)]  # + 2 for c, m
            num_params = 26
        while len(params_to_fit) < num_params:
            params_to_fit.append(0)
    else:
        params_to_fit = [1 for i in range(len(fit_model.get_param_defn()))]

    return params_to_fit

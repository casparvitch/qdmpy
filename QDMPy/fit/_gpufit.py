# -*- coding: utf-8 -*-
"""
This module holds tools for fitting raw data via gpufit. (gpufit backend)

Functions
---------
 - `QDMPy.fit._gpufit.prep_gpufit_fit_options`
 - `QDMPy.fit._gpufit.get_gpufit_modelID`
 - `QDMPy.fit._gpufit.prep_gpufit_backend`
 - `QDMPy.fit._gpufit.gen_gpufit_init_guesses`
 - `QDMPy.fit._gpufit.fit_single_pixel_gpufit`
 - `QDMPy.fit._gpufit.fit_ROI_avg_gpufit`
 - `QDMPy.fit._gpufit.fit_AOIs_gpufit`
 - `QDMPy.fit._gpufit.fit_pixels_gpufit`
 - `QDMPy.fit._gpufit.gpufit_data_shape`
 - `QDMPy.fit._gpufit.gpufit_reshape_result`

"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.fit._gpufit.prep_gpufit_fit_options": True,
    "QDMPy.fit._gpufit.get_gpufit_modelID": True,
    "QDMPy.fit._gpufit.prep_gpufit_backend": True,
    "QDMPy.fit._gpufit.gen_gpufit_init_guesses": True,
    "QDMPy.fit._gpufit.fit_single_pixel_gpufit": True,
    "QDMPy.fit._gpufit.fit_ROI_avg_gpufit": True,
    "QDMPy.fit._gpufit.fit_AOIs_gpufit": True,
    "QDMPy.fit._gpufit.fit_pixels_gpufit": True,
    "QDMPy.fit._gpufit.gpufit_data_shape": True,
    "QDMPy.fit._gpufit.gpufit_reshape_result": True,
}

# ============================================================================

import pygpufit.gpufit as gf
import numpy as np

# ============================================================================

import QDMPy.fit.model as fit_models
import QDMPy.fit._shared as fit_shared
import QDMPy.constants

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
            f"Didn't know what to do with 'gpfit_estimator_id' = {options['gpufit_estimator_id']}, available options: 'LSE', 'MLE'"
        )
    return gpufit_fit_options


# ============================================================================


def get_gpufit_modelID(options, fit_model):
    """
    Find corresponding gpufit modelID for this fitmodel

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    fit_model : `QDMPy.fit.model.FitModel`
        Must be one of (in this exact order/format):
        for odmr, LORENTZ8: {'linear': 1, 'lorentzian': 1<=n<=8}
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
    ModelID = None
    for i in range(8):
        if ffs == {"linear": 1, "lorentzian": i + 1}:
            ModelID = gf.ModelID.LORENTZ8
    if ffs == {"constant": 1, "stretched_exponential": 1}:
        ModelID = gf.ModelID.STRETCHED_EXP

    if ModelID is None:
        raise RuntimeError(
            "No gpufit modelID found for those fit_functions.\n"
            + "Available fit_functions:\n"  # noqa: W503
            + "LORENTZ8: {'linear': 1, 'lorentzian': 1<=n<=8}"  # noqa: W503
            + "STRETCHED_EXP: {'constant': 1, 'stretched_exponential': 1}"  # noqa: W503
            + "NOTE THEY MUST BE IN THAT ORDER & FORMAT!"  # noqa: W503
        )

    return ModelID


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
    fit_model : `QDMPy.fit.model.FitModel`
        Must be one of (in this exact order/format):
        for odmr, LORENTZ8: {'linear': 1, 'lorentzian': 1<=n<=8}
        for t1/etc., STRETCHED_EXP: {'constant': 1, 'stretched_exponential': 1}

    """
    if not gf.cuda_available():
        raise RuntimeError(f"CUDA error:\n{gf.get_last_error()}")

    options["CUDA_version_runtime"], options["CUDA_version_driver"] = gf.get_cuda_version()

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
                for pos, key in enumerate(QDMPy.constants.AVAILABLE_FNS[fn_type].param_defn):
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
                for pos, key in enumerate(QDMPy.constants.AVAILABLE_FNS[fn_type].param_defn):
                    param_lst.append(0)
                    bound_lst.append(0)
                    bound_lst.append(1)

    fit_param_ar = np.array(param_lst)
    fit_param_bound_ar = np.array(bound_lst)
    return fit_param_ar, fit_param_bound_ar


# ============================================================================


def fit_single_pixel_gpufit(options, pixel_pl_ar, sweep_list, fit_model, roi_avg_fit_result):
    """
    Fit Single pixel and return optimal fit parameters with gpufit backend

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    pixel_pl_ar : np array, 1D
        Normalised PL as function of sweep_list for a single pixel.
    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : `QDMPy.fit.model.FitModel`
        Model we're fitting to.
    roi_avg_fit_result : `QDMPy.fit._shared.ROIAvgFitResult`
        `QDMPy.fit._shared.ROIAvgFitResult` object, to pull fit_options from.

    Returns
    -------
    pixel_parameters : np array, 1D
        Best fit parameters, as determined by gpufit
    """
    # note need to do the fit at least twice (gpufit requirements) so we do it twice here.
    # (and disregard one of em)

    # this is just constructing the initial parameter guesses and bounds in the right format
    init_guess_params, init_bounds = gen_gpufit_init_guesses(
        options, *fit_shared.gen_init_guesses(options)
    )
    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        init_guess_params = roi_avg_fit_result.best_params.copy()

    # only fit the params we want to :)
    if options["ModelID"] == gf.ModelID.LORENTZ8:
        params_to_fit = [1 for i in range(len(fit_models.get_param_defn(fit_model)))]
        while len(params_to_fit) < 26:
            params_to_fit.append(0)
    else:
        params_to_fit = [1 for i in range(len(fit_models.get_param_defn(fit_model)))]

    params_to_fit = np.array(params_to_fit, dtype=np.int32)
    init_guess_params = np.repeat([init_guess_params], repeats=2, axis=0)
    init_guess_params = init_guess_params.astype(dtype=np.float32)

    pixel_pl_ar_doubled = np.repeat([pixel_pl_ar], repeats=2, axis=0)

    parameters, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(
        pixel_pl_ar_doubled.astype(np.float32),
        None,
        options["ModelID"],
        init_guess_params,
        constraints=init_bounds.astype(np.float32),
        constraint_types=np.array(
            [gf.ConstraintType.LOWER_UPPER for i in range(len(init_guess_params))], dtype=np.int32
        ),
        user_info=np.array(sweep_list, dtype=np.float32),
        parameters_to_fit=params_to_fit,
    )

    return parameters[0, :]  # only take one of the parameter results


# ============================================================================


def fit_ROI_avg_gpufit(options, sig_norm, sweep_list, fit_model):
    """
    Fit the average of the measurement over the region of interest specified, with gpufit.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    sig_norm : np array, 3D
        Normalised measurement array, shape: [sweep_list, y, x].
    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : `QDMPy.fit.model.FitModel`
        Model we're fitting to.

    Returns
    -------
    result : `QDMPy.fit._shared.ROIAvgFitResult`
        object containing the fit result (see class specifics)
    """

    # note need to do the fit at least twice (gpufit requirements) so we do it twice here.

    # fit *all* pl data (i.e. summing over FOV)
    # collapse to just pl_ar (as function of sweep, 1D)
    pl_roi = np.nanmean(np.nanmean(sig_norm, axis=2), axis=1)
    pl_roi_doubled = np.repeat([pl_roi], repeats=2, axis=0)

    # this is just constructing the initial parameter guesses and bounds in the right format
    init_guess_params, init_bounds = gen_gpufit_init_guesses(
        options, *fit_shared.gen_init_guesses(options)
    )

    gpufit_fit_options = prep_gpufit_fit_options(options)

    # only fit the params we want to :)
    if options["ModelID"] == gf.ModelID.LORENTZ8:
        params_to_fit = [1 for i in range(len(fit_models.get_param_defn(fit_model)))]
        while len(params_to_fit) < 26:
            params_to_fit.append(0)
    else:
        params_to_fit = [1 for i in range(len(fit_models.get_param_defn(fit_model)))]

    params_to_fit = np.array(params_to_fit, dtype=np.int32)
    init_guess_params = np.repeat([init_guess_params], repeats=2, axis=0)
    init_guess_params = init_guess_params.astype(dtype=np.float32)

    best_params, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(
        pl_roi_doubled.astype(np.float32),
        None,
        options["ModelID"],
        init_guess_params,
        constraints=np.array(init_bounds, dtype=np.float32),
        constraint_types=np.array(
            [gf.ConstraintType.LOWER_UPPER for i in range(len(init_guess_params))], dtype=np.int32
        ),
        user_info=np.array(sweep_list, dtype=np.float32),
        parameters_to_fit=params_to_fit,
        **gpufit_fit_options,
    )
    return fit_shared.ROIAvgFitResult(
        "gpufit",
        gpufit_fit_options,
        fit_model,
        pl_roi,
        sweep_list,
        best_params[0, :],  # only take one of the results
        init_guess_params[0],  # return the params un-repeated
        init_bounds,
    )


# ============================================================================


def fit_AOIs_gpufit(
    options, sig_norm, pixel_pl_ar, sweep_list, fit_model, AOIs, roi_avg_fit_result
):
    """
    Fit AOIs and single pixel and return list of (list of) fit results (optimal fit parameters),
    using gpufit.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    sig_norm : np array, 3D
        Normalised measurement array, shape: [sweep_list, y, x].
    single_pixel_pl : np array, 1D
        Normalised measurement array, for chosen single pixel check.
    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq).
    fit_model : `QDMPy.fit.model.FitModel`
        Model we're fitting to.
    AOIs : list
        List of AOI specifications - each a length-2 iterable that can be used to directly index
        into sig_norm to return that AOI region, e.g. sig_norm[:, AOI[0], AOI[1]].
    roi_avg_fit_result : `QDMPy.fit._shared.ROIAvgFitResult`
        `QDMPy.fit._shared.ROIAvgFitResult` object, to pull `QDMPy.fit._shared.ROIAvgFitResult.fit_options`
        from.

    Returns
    -------
    fit_result_collection : `QDMPy.fit._shared.FitResultCollection`
        `QDMPy.fit._shared.FitResultCollection` object
    """
    # note need to do the fit at least twice (gpufit requirements) so we do it twice per AOI here.

    # this is just constructing the initial parameter guesses and bounds in the right format
    init_guess_params, init_bounds = gen_gpufit_init_guesses(
        options, *fit_shared.gen_init_guesses(options)
    )
    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        init_guess_params = roi_avg_fit_result.best_params.copy()

    single_pixel_fit_params = fit_single_pixel_gpufit(
        options, pixel_pl_ar, sweep_list, fit_model, roi_avg_fit_result
    )

    AOI_avg_best_fit_results_lst = []

    # only fit the params we want to :)
    if options["ModelID"] == gf.ModelID.LORENTZ8:
        params_to_fit = [1 for i in range(len(fit_models.get_param_defn(fit_model)))]
        while len(params_to_fit) < 26:
            params_to_fit.append(0)
    else:
        params_to_fit = [1 for i in range(len(fit_models.get_param_defn(fit_model)))]

    params_to_fit = np.array(params_to_fit, dtype=np.int32)
    init_guess_params = np.repeat([init_guess_params], repeats=2, axis=0)
    init_guess_params = init_guess_params.astype(dtype=np.float32)

    for AOI in AOIs:
        aoi_sig_norm = sig_norm[:, AOI[0], AOI[1]]
        aoi_avg = np.nanmean(np.nanmean(aoi_sig_norm, axis=2), axis=1)
        aoi_avg_doubled = np.repeat([aoi_avg], repeats=2, axis=0)

        fitting_results, _, _, _, _ = gf.fit_constrained(
            aoi_avg_doubled.astype(np.float32),
            None,
            options["ModelID"],
            np.array(init_guess_params, dtype=np.float32),
            constraints=np.array(init_bounds, dtype=np.float32),
            constraint_types=np.array(
                [gf.ConstraintType.LOWER_UPPER for i in range(len(init_guess_params))],
                dtype=np.int32,
            ),
            user_info=np.array(sweep_list, dtype=np.float32),
            parameters_to_fit=params_to_fit,
        )
        AOI_avg_best_fit_results_lst.append(fitting_results[0, :])

    return fit_shared.FitResultCollection(
        "gpufit", roi_avg_fit_result, single_pixel_fit_params, AOI_avg_best_fit_results_lst
    )


# ============================================================================


def fit_pixels_gpufit(options, sig_norm, sweep_list, fit_model, roi_avg_fit_result):
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
    fit_model : `QDMPy.fit.model.FitModel`
        Model we're fitting to.
    roi_avg_fit_result : `QDMPy.fit._shared.ROIAvgFitResult`
        `QDMPy.fit._shared..ROIAvgFitResult` object, to pull fit_options from.

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    """
    sweep_ar = np.array(sweep_list)
    # data_length = np.shape(sig_norm)[1] * np.shape(sig_norm)[2]

    # randomize order of fitting pixels (will un-scramble later) so ETA is more correct
    # not really necessary for gpu?
    if options["scramble_pixels"]:
        pixel_data, unshuffler = fit_shared.shuffle_pixels(sig_norm)
    else:
        pixel_data = sig_norm

    # this is just constructing the initial parameter guesses and bounds in the right format
    init_guess_params, init_bounds = gen_gpufit_init_guesses(
        options, *fit_shared.gen_init_guesses(options)
    )
    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        init_guess_params = roi_avg_fit_result.best_params.copy()

    num_pixels = np.shape(sig_norm)[1] * np.shape(sig_norm)[2]

    # need to reshape init_guess_params, one for each pixel
    guess_params = np.array([init_guess_params], dtype=np.float32)
    init_guess_params_reshaped = np.repeat(guess_params, repeats=num_pixels, axis=0)

    # only fit the params we want to :) {i.e. < 8 peak ODMR fit etc.}
    if options["ModelID"] == gf.ModelID.LORENTZ8:
        params_to_fit = [1 for i in range(len(fit_models.get_param_defn(fit_model)))]
        while len(params_to_fit) < 26:
            params_to_fit.append(0)
    else:
        params_to_fit = [1 for i in range(len(fit_models.get_param_defn(fit_model)))]

    # reshape sig_norm in a way that gpufit likes: (number_fits, number_points)
    sig_norm_shaped, pixel_posns = gpufit_data_shape(pixel_data)

    fitting_results, _, _, _, execution_time = gf.fit_constrained(
        sig_norm_shaped,
        None,
        options["ModelID"],
        init_guess_params_reshaped,
        constraints=np.array(init_bounds, dtype=np.float32),
        constraint_types=np.array(
            [gf.ConstraintType.LOWER_UPPER for i in range(len(init_guess_params))], dtype=np.int32
        ),
        user_info=np.array(sweep_list, dtype=np.float32),
        parameters_to_fit=np.array(params_to_fit, dtype=np.int32),
    )
    # for the record
    options["fit_time_(s)"] = execution_time

    # calculate jacobians via scipy as gpufit doesn't return them at soln
    jacs = [fit_model.jacobian_scipyfit(param_ar, sweep_ar, None) for param_ar in fitting_results]

    fit_results = gpufit_reshape_result(fitting_results, pixel_posns, jacs)

    res, sigmas = fit_shared.get_pixel_fitting_results(
        fit_model, fit_results, pixel_data, sweep_ar
    )
    if options["scramble_pixels"]:
        res = fit_shared.unshuffle_fit_results(res, unshuffler)
        sigmas = fit_shared.unshuffle_fit_results(sigmas, unshuffler)

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
    for y, x, spectrum in fit_shared.pixel_generator(sig_norm):
        pixel_posns.append((y, x))
        sig_norm_reshaped.append(spectrum)
    return np.array(sig_norm_reshaped, dtype=np.float32), pixel_posns


# ============================================================================


def gpufit_reshape_result(pixel_param_results, pixel_posns, jacs):
    """
    Mimics `QDMPy.fit._scipy.to_squares_wrapper`, so gpufit can use the
    `QDMPy.fit._shared.get_pixel_fitting_results` function to get the nice
    usual dict of param result images.

    Arguments
    ---------
    pixel_param_results : np array, 2D
        parameter results as returned from gpufit. Shape: (num fits, num parameters)
    pixel_posns : list
        List of pixel positions (x,y) as returned by `QDMPy.fit.gpufit._gpufit_data_shape`.
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

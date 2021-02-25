# -*- coding: utf-8 -*-
"""
This module holds the general interface tools for fitting data, independent of fit backend
(e.g. scipy/gpufit etc.).

All of these functions are automatically loaded into the namespace when the fit
sub-package is imported. (e.g. import QDMPy.fit).

Functions
---------
 - `QDMPy.fit.interface.define_fit_model`
 - `QDMPy.fit.interface.fit_ROI_avg`
 - `QDMPy.fit.interface.fit_AOIs`
 - `QDMPy.fit.interface.fit_pixels`
 - `QDMPy.fit.interface._prep_fit_backends`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.fit.interface.define_fit_model": True,
    "QDMPy.fit.interface.fit_ROI_avg": True,
    "QDMPy.fit.interface.fit_AOIs": True,
    "QDMPy.fit.interface.fit_pixels": True,
    "QDMPy.fit.interface._prep_fit_backends": True,
}

# ============================================================================

import warnings

# ============================================================================

import QDMPy.fit.model as fit_models
import QDMPy.io.raw

# ============================================================================


def define_fit_model(options):
    """Define (and return) fit_model object, from options dictionary."""

    fit_functions = options["fit_functions"]

    fit_model = fit_models.FitModel(fit_functions)

    options["fit_param_defn"] = fit_models.get_param_odict(fit_model)

    _prep_fit_backends(options, fit_model)

    return fit_model


# ============================================================================


def _prep_fit_backends(options, fit_model):
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

    fit_model : `QDMPy.fit.model.FitModel`
        Model we're fitting to.
    """
    # ensure backend we want to use for pixel fittings is in comparison!
    if options["fit_backend"] not in options["fit_backend_comparison"]:
        warnings.warn(
            "Your chosen fit backend wasn't in the fit backend comparison list, so it has been added for you."
        )
        options["fit_backend_comparison"].append(options["fit_backend"])

    for fit_backend in options["fit_backend_comparison"]:
        if fit_backend == "scipyfit":
            # import, but make it globally available (to module)
            global fit_scipyfit
            _temp = __import__("QDMPy.fit._scipyfit", globals(), locals())
            fit_scipyfit = _temp.fit._scipyfit

        elif fit_backend == "gpufit":
            # here we use a programmatic import as we don't want to load (and crash)
            # if user doesn't have the gpufit stuff installed
            global fit_gpufit
            _temp = __import__("QDMPy.fit._gpufit", globals(), locals())
            fit_gpufit = _temp.fit._gpufit

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
        Normalised measurement array, shape: [sweep_list, y, x].

    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)

    fit_model : `QDMPy.fit.model.FitModel`
        Model we're fitting to.

    Returns
    -------
    backend_ROI_results_lst : list
        List of `QDMPy.fit._shared.ROIAvgFitResult` objects containing the fit result
        (see class specifics) for each fit backend selected for comparison

    """

    # list of ROI results, one element for each backend (as a (name, result) tuple)
    backend_ROI_results_lst = []
    # iterate through all possible fit backend choices
    for fit_backend in options["fit_backend_comparison"]:
        if fit_backend == "scipyfit":
            backend_ROI_results_lst.append(
                fit_scipyfit.fit_ROI_avg_scipyfit(options, sig_norm, sweep_list, fit_model)
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


def fit_AOIs(options, sig_norm, single_pixel_pl, sweep_list, fit_model, backend_ROI_results_lst):
    """
    Fit AOIs and single pixel with chosen backends and return fit_result_collection_lst

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

    roi_avg_fit_result : `QDMPy.fit._shared.ROIAvgFitResult`
        `QDMPy.fit._shared.ROIAvgFitResult` object, to pull `QDMPy.fit._shared.ROIAvgFitResult.fit_options`
        from.

    Returns
    -------
    fit_result_collection : `QDMPy.fit._shared.FitResultCollection`
        `QDMPy.fit._shared.FitResultCollection` object.
    """
    AOIs = QDMPy.io.raw._define_AOIs(options)

    fit_result_collection_lst = []  # list of FitResultCollection objects
    # iterate through all possible fit backend choices
    for n, fit_backend in enumerate(options["fit_backend_comparison"]):
        if fit_backend == "scipyfit":
            fit_result_collection_lst.append(
                fit_scipyfit.fit_AOIs_scipyfit(
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
        Normalised measurement array, shape: [sweep_list, y, x].

    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)

    fit_model : `QDMPy.fit.model.FitModel`
        Model we're fitting to.

    roi_avg_fit_result : `QDMPy.fit._shared.ROIAvgFitResult`
        `QDMPy.fit._shared.ROIAvgFitResult` object, to pull fit_options from.

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    """

    # here only use only chosen backend!
    if options["fit_backend"] == "scipyfit":
        return fit_scipyfit.fit_pixels_scipyfit(
            options, sig_norm, sweep_list, fit_model, roi_avg_fit_result
        )
    elif options["fit_backend"] == "gpufit":
        return fit_gpufit.fit_pixels_gpufit(
            options, sig_norm, sweep_list, fit_model, roi_avg_fit_result
        )
    else:
        raise RuntimeError(f"No fit_pixels fn defined for fit_backend = {options['fit_backend']}")

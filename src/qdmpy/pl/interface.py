# -*- coding: utf-8 -*-
"""
This module holds the general interface tools for fitting data, independent of fit backend
(e.g. scipy/gpufit etc.).

All of these functions are automatically loaded into the namespace when the fit
sub-package is imported. (e.g. import qdmpy.fit).

Functions
---------
 - `qdmpy.pl.interface.define_fit_model`
 - `qdmpy.pl.interface.fit_roi_avg_pl`
 - `qdmpy.pl.interface.fit_aois_pl`
 - `qdmpy.pl.interface.fit_all_pixels_pl`
 - `qdmpy.pl.interface._prep_fit_backends`
 - `qdmpy.pl.interface.get_pl_fit_result`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.pl.interface.define_fit_model": True,
    "qdmpy.pl.interface.fit_roi_avg_pl": True,
    "qdmpy.pl.interface.fit_aois_pl": True,
    "qdmpy.pl.interface.fit_all_pixels_pl": True,
    "qdmpy.pl.interface._prep_fit_backends": True,
    "qdmpy.pl.interface.get_pl_fit_result": True,
}

# ============================================================================

import importlib.metadata
import packaging

# ============================================================================

import qdmpy.pl.model
import qdmpy.shared.misc
from qdmpy.shared.misc import warn
import qdmpy.pl.io

# ============================================================================


def define_fit_model(options):
    """Define (and return) fit_model object, from options dictionary.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    fit_model : `qdmpy.pl.model.FitModel`
        FitModel used to fit to data.
    """

    fit_functions = options["fit_functions"]
    # reorder fit functions to as expected by gpufit (if model matches a gpufit model)
    ffs = sorted(list(fit_functions.items()))
    for i in range(8):
        if ffs == [("constant", 1), ("stretched_exponential", 1)]:
            fit_functions = {"constant": 1, "stretched_exponential": 1}
            break
        elif ffs == [("constant", 1), ("damped_rabi", 1)]:
            fit_functions = {"constant": 1, "damped_rabi": 1}
            break
        elif ffs == [("linear", 1), ("lorentzian", i + 1)]:
            fit_functions = {"linear": 1, "lorentzian": i + 1}
            break
        elif ffs == [("constant", 1), ("lorentzian", i + 1)]:
            fit_functions = {"constant": 1, "lorentzian": i + 1}
            break

    fit_model = qdmpy.pl.model.FitModel(fit_functions)

    options["fit_param_defn"] = fit_model.get_param_odict()

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
    fit_model : `qdmpy.pl.model.FitModel`
        FitModel used to fit to data.
    """
    # ensure backend we want to use for pixel fittings is in comparison!
    if options["fit_backend"] not in options["fit_backend_comparison"]:
        warn(
            "Your chosen fit backend wasn't in the fit backend comparison list, so it"
            " has been added for you."
        )
        options["fit_backend_comparison"].append(options["fit_backend"])

    for fit_backend in options["fit_backend_comparison"]:
        if fit_backend == "scipyfit":
            # import, but make it globally available (to module)
            global fit_scipyfit
            _temp = __import__("qdmpy.pl.scipyfit", globals(), locals())
            fit_scipyfit = _temp.pl.scipyfit

        elif fit_backend in ("gpufit", "cpufit"):
            # here we use a programmatic import as we don't want to load (and crash)
            # if user doesn't have the gpufit stuff installed

            if "fit_gpufit" not in globals():  # only import once for gpu/cpufit
                if fit_backend == "cpufit":
                    # we want to catch this if it errors
                    vrs = importlib.metadata.version("pygpufit")
                    if packaging.version.parse(vrs) <= packaging.version.parse("1.2.0"):
                        raise RuntimeError(
                            "cpufit requires pygpufit > 1.2.0 "
                            + "(with cpufit api exposed to python)"
                        )

                global fit_gpufit
                _temp = __import__("qdmpy.pl.gpufit", globals(), locals())
                fit_gpufit = _temp.pl.gpufit

                fit_gpufit.prep_gpufit_backend(options, fit_model)
        else:
            raise RuntimeError(
                "No backend preparation defined for fit_backend ="
                f" {options['fit_backend']}"
            )


# ============================================================================


def fit_roi_avg_pl(options, sig, ref, sweep_list, fit_model):
    """
    Fit the average of the measurement over the region of interest specified, with backend
    chosen via options["fit_backend_comparison"].

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
    backend_roi_results_lst : list
        List of `qdmpy.pl.common.ROIAvgFitResult` objects containing the fit result
        (see class specifics) for each fit backend selected for comparison.

    """

    # list of ROI results, one element for each backend (as a (name, result) tuple)
    backend_roi_results_lst = []
    # iterate through all possible fit backend choices
    for fit_backend in options["fit_backend_comparison"]:
        if fit_backend == "scipyfit":
            backend_roi_results_lst.append(
                fit_scipyfit.fit_roi_avg_pl_scipyfit(
                    options, sig, ref, sweep_list, fit_model
                )
            )
        elif fit_backend in ("gpufit", "cpufit"):
            backend_roi_results_lst.append(
                fit_gpufit.fit_roi_avg_pl_gpufit(
                    options,
                    sig,
                    ref,
                    sweep_list,
                    fit_model,
                    platform=fit_backend[:3],
                )
            )
        else:
            raise RuntimeError(
                "No fit_roi_avg_pl fn defined for fit_backend ="
                f" {options['fit_backend']}"
            )
    return backend_roi_results_lst


# ============================================================================


def fit_aois_pl(
    options,
    sig,
    ref,
    single_pixel_pl,
    sweep_list,
    fit_model,
    backend_roi_results_lst,
):
    """
    Fit AOIs and single pixel with chosen backends and return fit_result_collection_lst

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
        Model we're fitting to.
    backend_roi_results_lst : list
        List of `qdmpy.pl.common.ROIAvgFitResult` objects, to pull fit_options from.

    Returns
    -------
    fit_result_collection : `qdmpy.pl.common.FitResultCollection`
        `qdmpy.pl.common.FitResultCollection` object.
    """

    aois = qdmpy.shared.misc.define_aois(options)

    fit_result_collection_lst = []  # list of FitResultCollection objects
    # iterate through all possible fit backend choices
    for n, fit_backend in enumerate(options["fit_backend_comparison"]):
        if fit_backend == "scipyfit":
            fit_result_collection_lst.append(
                fit_scipyfit.fit_aois_pl_scipyfit(
                    options,
                    sig,
                    ref,
                    single_pixel_pl,
                    sweep_list,
                    fit_model,
                    aois,
                    backend_roi_results_lst[n],
                ),
            )
        elif fit_backend in ("gpufit", "cpufit"):
            fit_result_collection_lst.append(
                fit_gpufit.fit_aois_pl_gpufit(
                    options,
                    sig,
                    ref,
                    single_pixel_pl,
                    sweep_list,
                    fit_model,
                    aois,
                    backend_roi_results_lst[n],
                    platform=fit_backend[:3],
                )
            )
        else:
            raise RuntimeError(
                f"No fit_aois_pl fn defined for fit_backend = {options['fit_backend']}"
            )
    return fit_result_collection_lst


# ============================================================================


def fit_all_pixels_pl(options, sig_norm, sweep_list, fit_model, roi_avg_fit_result):
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
    fit_model : `qdmpy.pl.model.FitModel`
        Model we're fitting to.
    roi_avg_fit_result : `qdmpy.pl.common.ROIAvgFitResult`
        `qdmpy.pl.common.ROIAvgFitResult` object, to pull fit_options from.

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    sigmas : dict
        As above, but standard deviation for each param
    """

    # here only use only chosen backend!
    if options["fit_backend"] == "scipyfit":
        return fit_scipyfit.fit_all_pixels_pl_scipyfit(
            options, sig_norm, sweep_list, fit_model, roi_avg_fit_result
        )
    elif options["fit_backend"] in ("gpufit", "cpufit"):
        return fit_gpufit.fit_all_pixels_pl_gpufit(
            options,
            sig_norm,
            sweep_list,
            fit_model,
            roi_avg_fit_result,
            platform=options["fit_backend"][:3],
        )
    else:
        raise RuntimeError(
            "No fit_all_pixels_pl fn defined for fit_backend ="
            f" {options['fit_backend']}"
        )


# ============================================================================


def get_pl_fit_result(options, sig_norm, sweep_list, fit_model, wanted_roi_result):
    """Fit all pixels in image with chosen fit backend (or loads previous fit result)

    Wrapper for `qdmpy.pl.interface.fit_all_pixels_pl` with some options logic.


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
    wanted_roi_result : `qdmpy.pl.common.ROIAvgFitResult`
        `qdmpy.pl.common.ROIAvgFitResult` object, to pull fit_options from.

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
        (or None if not fitting pixels, this stops plotting, e.g. via
        `qdmpy.plot.pl.pl_param_images` from erroring)
    sigmas : dict
        As above, but standard deviation for each param
        (or None if not fitting pixels)
    """

    if options["found_prev_result"]:
        pixel_fit_params = qdmpy.pl.io.load_prev_pl_fit_results(options)
        sigmas = qdmpy.pl.io.load_prev_pl_fit_sigmas(options)
        warn("Using previous fit results.")
    elif options["fit_pl_pixels"]:
        pixel_fit_params, sigmas = fit_all_pixels_pl(
            options, sig_norm, sweep_list, fit_model, wanted_roi_result
        )
    else:
        pixel_fit_params = None  # not fitting pixels, this stops plotting (e.g. via plot.pl.pl_param_images) from erroring
        sigmas = None
    return pixel_fit_params, sigmas

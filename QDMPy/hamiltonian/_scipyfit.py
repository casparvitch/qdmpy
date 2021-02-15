# -*- coding: utf-8 -*-
"""

Classes
-------
 - `QDMPy.hamiltonian._scipyfit.`
Functions
---------
 - `QDMPy.hamiltonian._scipyfit.`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.hamiltonian._scipyfit.": True,
}
# ============================================================================

import numpy as np

# ============================================================================

from QDMPy.constants import AVAILABLE_HAMILTONIANS

# ============================================================================


def gen_scipyfit_init_guesses(options, init_guesses, init_bounds):
    """
    Generate arrays of initial fit guesses and bounds in correct form for scipy least_squares.

    init_guesses and init_bounds are dictionaries up to this point, we now convert to np arrays,
    that scipy will recognise.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    init_guesses : dict
        Dict holding guesses for each parameter, e.g. key -> list of guesses for the ham.

    init_bounds : dict
        Dict holding guesses for each parameter, e.g. key -> list of bounds for the ham.

    Returns
    -------
    fit_param_ar : np array, shape: num_params
        The initial fit parameter guesses.

    fi_param_bound_ar : np array, shape: (num_params, 2)
        Fit parameter bounds.
    """
    param_lst = []
    bound_lst = []

    for pos, key in enumerate(AVAILABLE_HAMILTONIANS[options["hamiltonian"]].param_defn):
        # this check is to handle the edge case of guesses/bounds
        # options being provided as numbers rather than lists of numbers
        try:
            param_lst.append(init_guesses[key])
        except (TypeError, KeyError):
            param_lst.append(init_guesses[key])
        if len(np.array(init_bounds[key]).shape) == 2:
            bound_lst.append(init_bounds[key])
        else:
            bound_lst.append(init_bounds[key])

    fit_param_ar = np.array(param_lst)
    fit_param_bound_ar = np.array(bound_lst)
    return fit_param_ar, fit_param_bound_ar


# ==========================================================================


def prep_scipyfit_options(options, fit_model):
    """
    General options dict -> scipyfit_options
    in format that scipy least_squares expects.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    fit_model : `QDMPy.fit._models.FitModel`
        Fit model object.

    Returns
    -------
    scipy_fit_options : dict
        Dictionary with options that scipy.optimize.least_squares expects, specific to fitting.
    """

    # this is just constructing the initial parameter guesses and bounds in the right format
    _, fit_param_bound_ar = gen_scipyfit_init_guesses(
        options, *fit_shared.gen_init_guesses(options)
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
    if fit_model.jacobian_scipyfit is None or not options["scipyfit_use_analytic_jac"]:
        scipyfit_options["jac"] = options["scipyfit_fit_jac_acc"]
    else:
        scipyfit_options["jac"] = fit_model.jacobian_scipyfit
    return scipyfit_options


# ==========================================================================

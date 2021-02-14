# -*- coding: utf-8 -*-

"""
This module defines the fit model used. We grab/use this regardless of fitting on cpu (scipy) or gpu etc.

Ensure any fit functions you define are added to the AVAILABLE_FNS module variable.
Try not to have overlapping parameter names in the same fit.

For ODMR peaks, ensure the frequency position of the peak is named something
prefixed by 'pos'. (see `QDMPy.bfield.calc.calc_bnv` for the reasoning).

Classes
-------
 - `QDMPy.fit._models.FitModel`

Functions
---------
 - `QDMPy.fit._models.get_param_defn`
 - `QDMPy.fit._models.get_param_odict`
 - `QDMPy.fit._models.get_param_unit`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.fit._models.FitModel": True,
    "QDMPy.fit._models.get_param_defn": True,
    "QDMPy.fit._models.get_param_odict": True,
    "QDMPy.fit._models.get_param_unit": True,
}

# ============================================================================

import numpy as np
from collections import OrderedDict

# ================================================================================================
# ================================================================================================
#
# FitModel Class
#
# ================================================================================================
# ================================================================================================


class FitModel:
    """FitModel used to fit to data."""

    def __init__(self, fit_functions):
        """
        Arguments
        ---------
        fit_functions: dict
            Dict of functions to makeup the fit model, key: fitfunc name, val: number of
            independent copies of that fitfunc.
            format: {"linear": 1, "lorentzian": 8} etc., i.e. options["fit_functions"]
        """

        self.fit_functions = fit_functions

        fn_chain = []
        all_param_len = 0

        # this import is not at top level to avoid cyclic import issues
        from QDMPy.constants import AVAILABLE_FNS as FN_SELECTOR

        for fn_type, num_fns in fit_functions.items():
            for i in range(num_fns):
                next_fn = FN_SELECTOR[fn_type]
                next_fn_param_len = len(next_fn.param_defn)
                next_fn_param_indices = [all_param_len + i for i in range(next_fn_param_len)]
                all_param_len += next_fn_param_len
                fn_chain.append(next_fn(next_fn_param_indices))

        self.fn_chain = fn_chain

    # =================================

    def __call__(self, param_ar, sweep_vec):
        """
        Evaluates fitmodel for given parameter values and sweep (affine) parameter values.

        Arguments
        ---------
        param_ar : np array, 1D
            Array of parameters fed into each fitfunc (these are what are fit by sc)

        sweep_vec : np array, 1D or number
            Affine parameter where the fit model is evaluated

        Returns
        -------
        Fit model evaluates at sweep_vec (output is same format as sweep_vec input)
        """
        out = np.zeros(np.shape(sweep_vec))
        for fn in self.fn_chain:
            this_fn_params = param_ar[fn.this_fn_param_indices]
            out += fn.eval(sweep_vec, *this_fn_params)

        return out

    # =================================

    def residuals_scipyfit(self, param_ar, sweep_vec, pl_val):
        """Evaluates residual: fit model - PL value """
        return self.__call__(param_ar, sweep_vec) - pl_val

    # =================================

    def jacobian_scipyfit(self, param_ar, sweep_vec, pl_val):
        """Evaluates jacobian of fitmodel in format expected by scipy least_squares"""

        for i, fn in enumerate(self.fn_chain):
            this_fn_params = param_ar[fn.this_fn_param_indices]
            if not i:
                val = fn.grad_fn(sweep_vec, *this_fn_params)
            else:
                val = np.hstack((val, fn.grad_fn(sweep_vec, *this_fn_params)))
        return val


# ====================================================================================


def get_param_defn(fit_model):
    """
    Returns list of parameters in fit_model, note there will be duplicates, and they do
    not have numbers e.g. 'pos_0'. Use `QDMPy.fit._models.get_param_odict` for that purpose.
    """
    param_defn_ar = []
    for fn in fit_model.fn_chain:
        param_defn_ar.extend(fn.param_defn)
    return param_defn_ar


# =================================


# get ordered dict of key: param name, val: param unit, for all parameters in chain
def get_param_odict(fit_model):
    """
    get ordered dict of key: param_key (param_name), val: param_unit for all parameters in fit_model
    """
    param_dict = OrderedDict()
    for fn in fit_model.fn_chain:
        for i in range(len(fn.param_defn)):
            param_name = fn.param_defn[i] + "_0"
            param_unit = fn.param_units[fn.param_defn[i]]
            # ensure no overlapping param names
            while param_name in param_dict.keys():
                param_name = param_name[:-1] + str(int(param_name[-1]) + 1)

            param_dict[param_name] = param_unit
    return param_dict


# =================================


def get_param_unit(fit_model, param_name, param_number):
    """
    Get unit for a given param_key (given by param_name + "_" + param_number)
    """
    if param_name == "residual":
        return "Error: sum( || residual(sweep_params) || ) (a.u.)"
    param_dict = get_param_odict(fit_model)
    return param_dict[param_name + "_" + str(param_number)]

# ====================================================================================



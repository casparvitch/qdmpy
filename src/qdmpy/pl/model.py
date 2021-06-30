# -*- coding: utf-8 -*-

"""
This module defines fit models used to fit QDM photoluminescence data.
We grab/use this regardless of fitting on cpu (scipy) or gpu etc.

Ensure any fit functions you define are added to the AVAILABLE_FNS module variable.
Try not to have overlapping parameter names in the same fit.

For ODMR peaks, ensure the frequency position of the peak is named something
prefixed by 'pos'. (see `qdmpy.field._bnv.get_bnvs_and_dshifts` for the reasoning).

Classes
-------
 - `qdmpy.pl.model.FitModel`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.pl.model.FitModel": True,
}

# ============================================================================

import numpy as np
from collections import OrderedDict

# ============================================================================

import qdmpy.pl.funcs

# ============================================================================

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
        for fn_type, num_fns in fit_functions.items():
            for _ in range(num_fns):
                next_fn = qdmpy.pl.funcs.AVAILABLE_FNS[fn_type]
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

    def residuals_scipyfit(self, param_ar, sweep_vec, pl_vals):
        """Evaluates residual: fit model (affine params/sweep_vec) - pl values"""
        return self.__call__(param_ar, sweep_vec) - pl_vals

    # =================================

    def jacobian_scipyfit(self, param_ar, sweep_vec, pl_vals):
        """Evaluates (analytic) jacobian of fitmodel in format expected by scipy least_squares"""

        # scipy just wants the jacobian wrt __call__, i.e. just derivs of param_ar
        for i, fn in enumerate(self.fn_chain):
            this_fn_params = param_ar[fn.this_fn_param_indices]
            grad = fn.grad_fn(sweep_vec, *this_fn_params)
            if not i:
                val = grad
            else:
                val = np.hstack((val, grad))
        return val

    # =================================

    def jacobian_defined(self):
        """Check if analytic jacobian is defined for this fit model."""
        for fn in self.fn_chain:
            dummy_params = np.array([1 for i in range(len(fn.param_defn))])
            if fn.grad_fn(np.array([0]), *dummy_params) is None:
                return False
        return True

    # =================================

    def get_param_defn(self):
        """
        Returns list of parameters in fit_model, note there will be duplicates, and they do
        not have numbers e.g. 'pos_0'. Use `qdmpy.fit.model.get_param_odict` for that purpose.

        Returns
        -------
        param_defn_ar : list
            List of parameter names (param_defn) in fit model.
        """
        param_defn_ar = []
        for fn in self.fn_chain:
            param_defn_ar.extend(fn.param_defn)
        return param_defn_ar

    # =================================

    def get_param_odict(self):
        """
        get ordered dict of key: param_key (param_name), val: param_unit for all parameters in fit_model

        Returns
        -------
        param_dict : dict
            Dictionary containing key: params, values: units.
        """
        param_dict = OrderedDict()
        for fn in self.fn_chain:
            for i in range(len(fn.param_defn)):
                param_name = fn.param_defn[i] + "_0"
                param_unit = fn.param_units[fn.param_defn[i]]
                # ensure no overlapping param names
                while param_name in param_dict.keys():
                    param_name = param_name[:-1] + str(int(param_name[-1]) + 1)

                param_dict[param_name] = param_unit
        return param_dict

    # =================================

    def get_param_unit(self, param_name, param_number):
        """Get unit for a given param_key (given by param_name + "_" + param_number)

        Arguments
        ---------
        param_name : str
            Name of parameter, e.g. 'pos'
        param_number : float or int
            Which parameter to use, e.g. 0 for 'pos_0'

        Returns
        -------
        unit : str
            Unit for that parameter, e.g. "constant" -> "Amplitude (a.u.)""
        """
        if param_name == "residual":
            return "Error: sum( || residual(sweep_params) || ) over affine param (a.u.)"
        param_dict = self.get_param_odict()
        return param_dict[param_name + "_" + str(param_number)]


# ====================================================================================

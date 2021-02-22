# -*- coding: utf-8 -*-
"""
This module holds the tools for loading/saving field results.

Functions
---------
 - `QDMPy.io.field.save_bnvs_and_dshifts`

"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.io.field.save_bnvs_and_dshifts": True,
}

# ============================================================================

import numpy as np

# ============================================================================

import QDMPy.io.raw

# ============================================================================


def save_bnvs_and_dshifts(options, name, bnvs, dshifts):
    if bnvs:
        for i, bnv in enumerate(bnvs):
            np.savetxt(options["sub_ref_data_dir"] / f"{name}_bnv_{i}.txt", bnv)
    if dshifts:
        for i, dshift in enumerate(dshifts):
            np.savetxt(options["sub_ref_data_dir"] / f"{name}_dshift_{i}.txt", dshift)


# ============================================================================


def save_ham_pixel_results(options, pixel_fit_params):
    """
    Saves hamiltonian pixel fit results to disk.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    pixel_fit_params : OrderedDict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    """
    if pixel_fit_params is not None:
        for param_key, result in pixel_fit_params.items():
            np.savetxt(options["sub_ref_data_dir"] / f"{param_key}.txt", result)


# ============================================================================


def load_prev_ham_results(options):
    prev_options = QDMPy.io.raw._get_prev_options._get_prev_options(options)

    fit_param_res_dict = {}

    from QDMPy.constants import AVAILABLE_FNS as FN_SELECTOR

    for fn_type, num in prev_options["fit_functions"].items():
        for param_name in FN_SELECTOR[fn_type].param_defn:
            for n in range(num):
                param_key = param_name + "_" + str(n)
                fit_param_res_dict[param_key] = load_ham_param(options, param_key)
    fit_param_res_dict["residual_0"] = load_ham_param(options, "residual_0")
    return fit_param_res_dict


# ============================================================================


def load_ham_param(options, param_key):
    """Load a previously hamiltonian param, of name 'param_key'."""
    return np.loadtxt(options["data_dir"] / (param_key + ".txt"))


# ============================================================================

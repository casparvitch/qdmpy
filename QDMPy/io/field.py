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


def save_field_results(options, bnv_tuple, dshift_tuple, params_tuple, sigmas_tuple):
    # save in correct places...
    # sig, ref, sig_sub_ref as name etc.
    save_bnvs_and_dshifts(options, "sig", bnv_tuple[0], dshift_tuple[0])
    save_bnvs_and_dshifts(options, "ref", bnv_tuple[1], dshift_tuple[1])
    save_bnvs_and_dshifts(options, "sig_sub_ref", bnv_tuple[3], [])

    save_field_params(options, "sig", params_tuple[0])
    save_field_params(options, "ref", params_tuple[1])
    # save_field_params(options, "sig_sub_ref", params_tuple[2]) # not implemented yet

    # FIXME continue this work eh


# ============================================================================


def save_bnvs_and_dshifts(options, name, bnvs, dshifts):
    if bnvs:
        for i, bnv in enumerate(bnvs):
            if name == "sig_sub_ref":
                np.savetxt(options["field_dir"] / f"sig_sub_ref_bnv_{i}.txt", bnv)
            else:
                np.savetxt(options[f"field_{name}_dir"] / f"{name}_bnv_{i}.txt", bnv)
    if dshifts:
        for i, dshift in enumerate(dshifts):
            if name == "sig_sub_ref":
                np.savetxt(options["field_dir"] / f"sig_sub_ref_dshift_{i}.txt", bnv)
            else:
                np.savetxt(options[f"field_{name}_dir"] / f"{name}_dshift_{i}.txt", bnv)


# ============================================================================


def save_field_params(options, name, pixel_fit_params):
    """
    Saves hamiltonian pixel fit results to disk.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    name : str
        TODO

    pixel_fit_params : OrderedDict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    """
    if pixel_fit_params is not None:
        for param_key, result in pixel_fit_params.items():
            np.savetxt(options["sub_ref_data_dir"] / f"{name}_{param_key}_.txt", result)


# ============================================================================


def load_prev_field_params(options):
    prev_options = QDMPy.io.raw._get_prev_options(options)

    field_param_dict = {}

    for param in prev_options.field_params():
        field_param_dict[param] = load_field_param(options, param)

    field_param_dict["residual_ham"] = load_field_param(options, "residual_ham")
    return field_param_dict


# ============================================================================


def load_field_param(options, param):
    """Load a previously field param, of name 'param' (string)."""
    return np.loadtxt(options["sub_ref_data_dir"] / (param + ".txt"))


# ============================================================================


def load_arb_field_params(path, param_names):
    """
    load field params from directory at 'path', of names 'param_names' (iterable of strings)
    (e.g. ["Bx", "By", "Bz"] etc.)
    """
    return {param: load_arb_field_param(path, param) for param in param_names}


# ============================================================================


def load_arb_field_param(path, param):
    """Load a previously field param, of name 'param' (string) stored in dir at 'path'."""
    return np.loadtxt(path / (param + ".txt"))


# ============================================================================


def check_for_prev_field_result(options):
    # FIXME
    pass

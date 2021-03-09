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
import os

# ============================================================================

import QDMPy.io.fit
import QDMPy.hamiltonian

# ============================================================================


def save_field_calcs(options, bnv_tuple, dshift_tuple, params_tuple, sigmas_tuple):
    # save in correct places...
    save_bnvs_and_dshifts(options, "sig", bnv_tuple[0], dshift_tuple[0])
    save_bnvs_and_dshifts(options, "ref", bnv_tuple[1], dshift_tuple[1])
    save_bnvs_and_dshifts(options, "sig_sub_ref", bnv_tuple[2], [])

    save_field_params(options, "sig", params_tuple[0])
    save_field_params(options, "ref", params_tuple[1])
    save_field_params(options, "sig_sub_ref", params_tuple[2])

    save_field_sigmas(options, "sig", sigmas_tuple[0])
    save_field_sigmas(options, "ref", sigmas_tuple[1])
    save_field_sigmas(options, "sig_sub_ref", sigmas_tuple[2])


# ============================================================================


def save_bnvs_and_dshifts(options, name, bnvs, dshifts):
    if bnvs:
        for i, bnv in enumerate(bnvs):
            if name in ["sig", "ref"]:
                path = options[f"field_{name}_dir"] / f"{name}_bnv_{i}.txt"
            else:
                path = options["field_dir"] / f"{name}_bnv_{i}.txt"
            np.savetxt(path, bnv)
    if dshifts:
        for i, dshift in enumerate(dshifts):
            if name in ["sig", "ref"]:
                path = options[f"field_{name}_dir"] / f"{name}_dshift_{i}.txt"
            else:
                path = options["field_dir"] / f"{name}_dshift_{i}.txt"
            np.savetxt(path, dshift)


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
            if name in ["sig", "ref"]:
                path = options[f"field_{name}_dir"] / f"{name}_{param_key}.txt"
            else:
                path = options["field_dir"] / f"{name}_{param_key}.txt"
            np.savetxt(path, result)


# ============================================================================


def save_field_sigmas(options, name, sigmas):
    """
    Saves hamiltonian pixel fit sigmas (SD) to disk.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    name : str
        TODO

    sigmas : OrderedDict
        Dictionary, key: param_keys, val: image (2D) of sigmas across FOV.
    """
    if sigmas is not None:
        for param_key, sigma in sigmas.items():
            if name in ["sig", "ref"]:
                path = options[f"field_{name}_dir"] / f"{name}_{param_key}_sigma.txt"
            else:
                path = options["field_dir"] / f"{name}_{param_key}_sigma.txt"
            np.savetxt(path, sigma)


# ============================================================================


def load_prev_field_calcs(options):
    sig_bnvs, sig_dshifts = load_prev_bnvs_and_dshifts(options, "sig")
    ref_bnvs, ref_dshifts = load_prev_bnvs_and_dshifts(options, "ref")
    sig_sub_ref_bnvs, _ = load_prev_bnvs_and_dshifts(options, "sig_sub_ref")

    sig_params = load_prev_field_params(options, "sig")
    ref_params = load_prev_field_params(options, "ref")
    sig_sub_ref_params = load_prev_field_params(options, "sig_sub_ref")

    sig_sigmas = load_prev_field_sigmas(options, "sig")
    ref_sigmas = load_prev_field_sigmas(options, "ref")
    sig_sub_ref_sigmas = load_prev_field_sigmas(options, "sig_sub_ref")

    return (
        (sig_bnvs, ref_bnvs, sig_sub_ref_bnvs),
        (sig_dshifts, ref_dshifts),
        (sig_params, ref_params, sig_sub_ref_params),
        (sig_sigmas, ref_sigmas, sig_sub_ref_sigmas),
    )


# ============================================================================


def load_prev_bnvs_and_dshifts(options, name):
    bnvs = []
    dshifts = []
    for i in range(4):
        if name in ["sig", "ref"]:
            bpath = options[f"field_{name}_dir"] / f"{name}_bnv_{i}.txt"
            dpath = options[f"field_{name}_dir"] / f"{name}_dshift_{i}.txt"
        else:
            bpath = options["field_dir"] / f"{name}_bnv_{i}.txt"
            dpath = options[f"field_dir"] / f"{name}_dshift_{i}.txt"
        if os.path.isfile(bpath):
            bnvs.append(np.loadtxt(bpath))
        if os.path.isfile(dpath):
            dshifts.append(np.loadtxt(dpath))
    return bnvs, dshifts


# ============================================================================


def load_prev_field_params(options, name):
    prev_options = QDMPy.io.fit._get_prev_options(options)

    field_param_dict = {}

    for param in prev_options["field_params"]:
        field_param_dict[param] = load_field_param(options, name, param)

    return field_param_dict


# ============================================================================


def load_prev_field_sigmas(options, name):
    prev_options = QDMPy.io.fit._get_prev_options(options)

    sigmas_dict = {}

    for param in prev_options["field_params"]:
        if param == "residual_field":
            continue
        sigmas_dict[param] = load_field_sigma(options, name, param)

    return sigmas_dict


# ============================================================================


def load_field_param(options, name, param):
    """Load a previously field param, 'param' (string), of type 'name' (e.g. sig/ref etc.)"""
    if name in ["sig", "ref"]:
        path = options[f"field_{name}_dir"] / f"{name}_{param}.txt"
    else:
        path = options["field_dir"] / f"{name}_{param}.txt"

    # handle if we don't have a ref param
    if not os.path.isfile(path) and name == "ref":
        return None
    else:
        return np.loadtxt(path)


# ============================================================================


def load_field_sigma(options, name, sigma):
    """Load a previously field sigma, 'sigma' (string), of type 'name' (e.g. sig/ref etc.)"""
    if name in ["sig", "ref"]:
        path = options[f"field_{name}_dir"] / f"{name}_{sigma}_sigma.txt"
    else:
        path = options["field_dir"] / f"{name}_{sigma}_sigma.txt"

    # handle if we don't have a ref sigma
    if not os.path.isfile(path) and name == "ref":
        return None
    else:
        return np.loadtxt(path)


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


def check_for_prev_field_calc(options):
    # loads all of sig, ref, and sig_sub_ref info.

    # first find prev_options
    if not options["force_field_calc"]:
        if not QDMPy.io.fit._prev_options_exist(options):
            options["found_prev_field_calc"] = False
            options["found_prev_field_calc_reason"] = "couldn't find previous options"
        elif not (res := _field_options_compatible(options))[0]:
            options["found_prev_field_calc"] = False
            options["found_prev_field_calc_reason"] = "option not compatible:\n" + res[1]
        elif not (res2 := _prev_pixel_field_calcs_exist(options))[0]:
            options["found_prev_field_calc"] = False
            options["found_prev_field_calc_reason"] = (
                "couldn't find prev field pixel results:\n" + res2[1]
            )
        else:
            options["found_prev_field_calc"] = True
            options["found_prev_field_calc_reason"] = "found prev calc :)"
    else:
        options["found_prev_field_calc"] = False
        options["found_prev_field_calc_reason"] = "option 'force_field_calc' was True"


# ============================================================================


def _prev_pixel_field_calcs_exist(options):
    prev_options = QDMPy.io.fit._get_prev_options(options)
    # skip 'ref' check, as it isn't always there!
    for param_key in prev_options["field_params"]:
        spath = options["field_sig_dir"] / f"sig_{param_key}.txt"
        if not os.path.isfile(spath):
            return False, f"couldn't find previous field param: sig_{param_key}"
        ssfpath = options["field_dir"] / f"sig_sub_ref_{param_key}.txt"
        if not os.path.isfile(ssfpath):
            return False, f"couldn't find previous field param: sig_sub_ref_{param_key}"

    return True, "found all prev field pixel results :)"


# ============================================================================


def _field_options_compatible(options):
    prev_options = QDMPy.io.fit._get_prev_options(options)
    if options["field_method_used"] != prev_options["field_method_used"]:
        return False, "method was different to that selected (or auto-selected) presently."

    if options["freqs_to_use"] != prev_options["freqs_to_use"]:
        return False, "different freqs_to_use option for sig & ref."

    if options["field_method_used"] == "prop_single_bnv":
        if options["single_bnv_choice"] != prev_options["single_bnv_choice"]:
            return False, "different single_bnv_choice option for sig & ref."

    if options["use_unvs"] != prev_options["use_unvs"]:
        return False, "different 'use_unvs' options for sig & ref."

    if options["use_unvs"]:
        if options["unvs"] != prev_options["unvs"]:
            return False, "different unvs chosen for sig & ref."

    if options["field_method_used"] == "hamiltonian_fitting":
        if options["hamiltonian"] != prev_options["hamiltonian"]:
            return False, "different hamiltonian selected for sig & ref."

        guesser = QDMPy.hamiltonian.get_ham_guess_and_bounds
        this_guess, this_bounds = guesser(options)
        prev_guess, prev_bounds = guesser(prev_options)

        for key in this_guess:
            if this_guess[key] != prev_guess[key]:
                return False, "guesses do not match."
        for key in this_bounds:
            if any(this_bounds[key] != prev_bounds[key]):
                return False, "bounds do not match."

        for fit_opt_name in [
            "scipyfit_method",
            "scipyfit_use_analytic_jac",
            "scipyfit_fit_jac_acc",
            "scipyfit_fit_gtol",
            "scipyfit_fit_xtol",
            "scipyfit_fit_ftol",
            "scipyfit_scale_x",
            "scipyfit_loss_fn",
        ]:
            if (
                fit_opt_name not in options
                or fit_opt_name not in prev_options
                or options[fit_opt_name] != prev_options[fit_opt_name]
            ):
                return False, f"scipyfit (field) option different: {fit_opt_name}"

    return True, "options are compatible :)"


# ============================================================================

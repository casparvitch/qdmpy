# -*- coding: utf-8 -*-
"""
This module holds the tools for loading/saving field results.

Functions
---------
 - `qdmpy.io.field.save_field_calcs`
 - `qdmpy.io.field.save_bnvs_and_dshifts`
 - `qdmpy.io.field.save_field_params`
 - `qdmpy.io.field.save_field_sigmas`
 - `qdmpy.io.field.load_prev_field_calcs`
 - `qdmpy.io.field.load_prev_bnvs_and_dshifts`
 - `qdmpy.io.field.load_prev_field_params`
 - `qdmpy.io.field.load_prev_bnvs_and_dshifts`
 - `qdmpy.io.field.load_prev_field_sigmas`
 - `qdmpy.io.field.load_field_param`
 - `qdmpy.io.field.load_field_sigma`
 - `qdmpy.io.field.load_arb_field_param`
 - `qdmpy.io.field.load_arb_field_params`
 - `qdmpy.io.field.check_for_prev_field_calc`
 - `qdmpy.io.field._prev_pixel_field_calcs_exist`
 - `qdmpy.io.field._field_options_compatible`

"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.io.field.save_field_calcs": True,
    "qdmpy.io.field.save_bnvs_and_dshifts": True,
    "qdmpy.io.field.save_field_params": True,
    "qdmpy.io.field.save_field_sigmas": True,
    "qdmpy.io.field.load_prev_field_calcs": True,
    "qdmpy.io.field.load_prev_bnvs_and_dshifts": True,
    "qdmpy.io.field.load_prev_field_params": True,
    "qdmpy.io.field.load_prev_field_sigmas": True,
    "qdmpy.io.field.load_field_param": True,
    "qdmpy.io.field.load_field_sigma": True,
    "qdmpy.io.field.load_arb_field_param": True,
    "qdmpy.io.field.load_arb_field_params": True,
    "qdmpy.io.field.check_for_prev_field_calc": True,
    "qdmpy.io.field._prev_pixel_field_calcs_exist": True,
    "qdmpy.io.field._field_options_compatible": True,
}

# ============================================================================

import numpy as np
import os

# ============================================================================

import qdmpy.io.fit
import qdmpy.hamiltonian

# ============================================================================


def save_field_calcs(options, bnv_ar, dshift_ar, params_ar, sigmas_ar):
    """save field calculations to disk.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    bnv_ar : array-like
        len-3 array of bnvs (each a list) (sig, ref, sig_sub_ref)
    dshift_ar : array-like
        len-2 array of dshifts (each a list) (sig, ref)
    params_ar : array-like
        len-3 array of param dicts (sig, ref, sig_sub_ref)
    sigmas_ar : array-like
        len-3 array of sigma dicts (sig, ref, sig_sub_ref)
    """
    # save in correct places...
    save_bnvs_and_dshifts(options, "sig", bnv_ar[0], dshift_ar[0])
    save_bnvs_and_dshifts(options, "ref", bnv_ar[1], dshift_ar[1])
    save_bnvs_and_dshifts(options, "sig_sub_ref", bnv_ar[2], [])

    save_field_params(options, "sig", params_ar[0])
    save_field_params(options, "ref", params_ar[1])
    save_field_params(options, "sig_sub_ref", params_ar[2])

    save_field_sigmas(options, "sig", sigmas_ar[0])
    save_field_sigmas(options, "ref", sigmas_ar[1])
    save_field_sigmas(options, "sig_sub_ref", sigmas_ar[2])


# ============================================================================


def save_bnvs_and_dshifts(options, name, bnvs, dshifts):
    """Save bnvs and dshifts to disk.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    name : str
        Name ascribed to this sigma, i.e. sig/ref/sig_sub_ref. Can handle others less gratiously.
    bnvs : list
        list of bnv results (2D image)
    dshifts : list
        list of dshift results (2D image)
    """
    if bnvs:
        for i, bnv in enumerate(bnvs):
            if bnv is not None:
                if name in ["sig", "ref", "sig_sub_ref"]:
                    path = options[f"field_{name}_dir"] / f"{name}_bnv_{i}.txt"
                else:
                    path = options["field_dir"] / f"{name}_bnv_{i}.txt"
                np.savetxt(path, bnv)
    if dshifts:
        for i, dshift in enumerate(dshifts):
            if dshift is not None:
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
        Name ascribed to this param, i.e. sig/ref/sig_sub_ref. Can handle others less gratiously.
    pixel_fit_params : OrderedDict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    """
    if pixel_fit_params:
        for param_key, result in pixel_fit_params.items():
            if result is not None:
                if name in ["sig", "ref", "sig_sub_ref"]:
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
        Name ascribed to this sigma, i.e. sig/ref/sig_sub_ref. Can handle others less gratiously.
    sigmas : OrderedDict
        Dictionary, key: param_keys, val: image (2D) of sigmas across FOV.
    """
    if sigmas is not None:
        for param_key, sigma in sigmas.items():
            if sigma is not None:
                if name in ["sig", "ref", "sig_sub_ref"]:
                    path = options[f"field_{name}_dir"] / f"{name}_{param_key}_sigma.txt"
                else:
                    path = options["field_dir"] / f"{name}_{param_key}_sigma.txt"
                np.savetxt(path, sigma)


# ============================================================================


def load_prev_field_calcs(options):
    """Load previous field calculation.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    bnv_ar : array-like
        len-3 array of bnvs (each a list) (sig, ref, sig_sub_ref)
    dshift_ar : array-like
        len-2 array of dshifts (each a list) (sig, ref)
    params_ar : array-like
        len-3 array of param dicts (sig, ref, sig_sub_ref)
    sigmas_ar : array-like
        len-3 array of sigma dicts (sig, ref, sig_sub_ref)
    """
    sig_bnvs, sig_dshifts = load_prev_bnvs_and_dshifts(options, "sig")
    ref_bnvs, ref_dshifts = load_prev_bnvs_and_dshifts(options, "ref")

    sig_params = load_prev_field_params(options, "sig")
    ref_params = load_prev_field_params(options, "ref")
    
    sig_sigmas = load_prev_field_sigmas(options, "sig")
    ref_sigmas = load_prev_field_sigmas(options, "ref")

    # Sam changed to below on 2021-04-21, so that background subtraction is done transparently.
    from qdmpy.field._bxyz import field_refsub, field_sigma_add
    from qdmpy.field._bnv import bnv_refsub
    sig_sub_ref_bnvs = bnv_refsub(options, sig_bnvs, ref_bnvs)
    # sig_sub_ref_bnvs, _ = load_prev_bnvs_and_dshifts(options, "sig_sub_ref")
    sig_sub_ref_params = field_refsub(options, sig_params, ref_params)
    # sig_sub_ref_params = load_prev_field_params(options, "sig_sub_ref")
    sig_sub_ref_sigmas = field_sigma_add(options, sig_sigmas, ref_sigmas)
    # sig_sub_ref_sigmas = load_prev_field_sigmas(options, "sig_sub_ref")

    return (
        [sig_bnvs, ref_bnvs, sig_sub_ref_bnvs],
        [sig_dshifts, ref_dshifts],
        [sig_params, ref_params, sig_sub_ref_params],
        [sig_sigmas, ref_sigmas, sig_sub_ref_sigmas],
    )


# ============================================================================


def load_prev_bnvs_and_dshifts(options, name):
    """Load previous bnv and dshift calculation

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    name : str
        Name ascribed to results you want to load,
        i.e. sig/ref/sig_sub_ref.

    Returns
    -------
    bnvs : list
        list of bnv results (2D image)
    dshifts : list
        list of dshift results (2D image)
    """
    bnvs = []
    dshifts = []
    for i in range(4):
        if name in ["sig", "ref", "sig_sub_ref"]:
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
    """Load previous field result ascribed to 'name'.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    name : str
        Name ascribed to results you want to load,
        i.e. sig/ref/sig_sub_ref.

    Returns
    -------
    field_params : dict
        Dictionary, key: param_keys, val: image (2D) of field param values across FOV.
    """

    prev_options = qdmpy.io.fit._get_prev_options(options)

    load_params = prev_options["field_params"]
    remove_these_params = [
        "Bx_recon",
        "By_recon",
        "Bz_recon",
    ]
    load_params = filter(lambda x: x not in remove_these_params, load_params)

    field_param_dict = {}

    for param in load_params:
        field_param_dict[param] = load_field_param(options, name, param)

    return field_param_dict


# ============================================================================


def load_prev_field_sigmas(options, name):
    """Load previous field sigma result ascribed to 'name'.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    name : str
        Name ascribed to results you want to load,
        i.e. sig/ref/sig_sub_ref.

    Returns
    -------
    sigma_params : dict
        Dictionary, key: param_keys, val: image (2D) of field sigma values across FOV.
    """
    prev_options = qdmpy.io.fit._get_prev_options(options)

    load_params = prev_options["field_params"]
    remove_these_params = [
        "Bx_recon",
        "By_recon",
        "Bz_recon",
    ]
    load_params = filter(lambda x: x not in remove_these_params, load_params)

    sigmas_dict = {}

    for param in load_params:
        if param == "residual_field":
            continue
        sigmas_dict[param] = load_field_sigma(options, name, param)

    return sigmas_dict


# ============================================================================


def load_field_param(options, name, param):
    """Load a previously field param, 'param' (string), of type 'name' (e.g. sig/ref etc.)"""
    trial_suffixes = ["_full.txt", ".txt"]
    # this loop is to try checking for the full result (i.e. pre-background subtraction) file
    #  if it exists. This way we can do a new background subtraction on the same field result.
    for s in trial_suffixes:
        if name in ["sig", "ref", "sig_sub_ref"]:
            path = options[f"field_{name}_dir"] / (f"{name}_{param}" + s)
        else:
            path = options["field_dir"] / (f"{name}_{param}" + s)
        if os.path.isfile(path):
            break

    # handle if we don't have a ref param (with any suffix candidate)
    if not os.path.isfile(path) and name == "ref":
        return None
    else:
        return np.loadtxt(path)


# ============================================================================


def load_field_sigma(options, name, sigma):
    """Load a previously field sigma, 'sigma' (string), of type 'name' (e.g. sig/ref etc.)"""
    trial_suffixes = ["_full.txt", ".txt"]
    # this loop is to try checking for the full result (i.e. pre-background subtraction) file
    #  if it exists. This way we can do a new background subtraction on the same field result.
    for s in trial_suffixes:
        if name in ["sig", "ref", "sig_sub_ref"]:
            path = options[f"field_{name}_dir"] / (f"{name}_{sigma}_sigma" + s)
        else:
            path = options["field_dir"] / (f"{name}_{sigma}_sigma" + s)
        if os.path.isfile(path):
            break

    # handle if we don't have a ref sigma (with any suffix candidate)
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
        if not qdmpy.io.fit._prev_options_exist(options):
            options["found_prev_field_calc"] = False
            options["found_prev_field_calc_reason"] = "couldn't find previous options"
        elif not qdmpy.io.fit._options_compatible(
            options, qdmpy.io.fit._get_prev_options(options)
        ):
            options["found_prev_field_calc"] = False
            options["found_prev_field_calc_reason"] = "incompatible fit options"
        elif not (res := _field_options_compatible(options))[0]:
            options["found_prev_field_calc"] = False
            options["found_prev_field_calc_reason"] = "option not compatible: " + res[1]
        elif not (res2 := _prev_pixel_field_calcs_exist(options))[0]:
            options["found_prev_field_calc"] = False
            options["found_prev_field_calc_reason"] = (
                "couldn't find prev field pixel results: " + res2[1]
            )
        else:
            options["found_prev_field_calc"] = True
            options["found_prev_field_calc_reason"] = "found prev calc :)"
    else:
        options["found_prev_field_calc"] = False
        options["found_prev_field_calc_reason"] = "option 'force_field_calc' was True"


# ============================================================================


def _prev_pixel_field_calcs_exist(options):
    """Can we find previous pixel field calculation?

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    do_exist : bool
        True if prev pixel field calculations have been found, else False
    reason : str
        Reason for the above decision
    """
    prev_options = qdmpy.io.fit._get_prev_options(options)
    if "field_params" not in prev_options or prev_options["field_params"] is None:
        return False, "no key 'field params' in prev_options"
    # skip 'ref' check, as it isn't always there!
    for param_key in prev_options["field_params"]:
        spath = options["field_sig_dir"] / f"sig_{param_key}.txt"
        if not os.path.isfile(spath):
            return False, f"couldn't find previous field param: sig_{param_key}"
        ssfpath = options["field_sig_sub_ref_dir"] / f"sig_sub_ref_{param_key}.txt"
        if not os.path.isfile(ssfpath):
            return False, f"couldn't find previous field param: sig_sub_ref_{param_key}"

    return True, "found all prev field pixel results :)"


# ============================================================================


def _field_options_compatible(options):
    """We have found some previous pixel field calculation, but do its parameters
    match what we want on this processing run?

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    do_match : bool
        True if options are compatible, else false
    reason : str
        Reason for the above decision
    """
    prev_options = qdmpy.io.fit._get_prev_options(options)
    if options["field_method_used"] != prev_options["field_method_used"]:
        return False, "method was different to that selected (or auto-selected) presently."

    if options["freqs_to_use"] != prev_options["freqs_to_use"]:
        return False, "different freqs_to_use option."

    if options["field_method_used"] == "prop_single_bnv":
        if options["single_bnv_choice"] != prev_options["single_bnv_choice"]:
            return False, "different single_bnv_choice option."

    if options["use_unvs"] != prev_options["use_unvs"]:
        return False, "different 'use_unvs' option."

    if options["use_unvs"]:
        if options["unvs"] != prev_options["unvs"]:
            return False, "different unvs chosen."

    if str(options["field_ref_dir"]) != str(prev_options["field_ref_dir"]):
        # Note: this could be done more intelligently -> grab field result etc. for a given file name
        # BUT it would be prone to error. More transparent to just calculate the field for sig & ref again
        return False, "different reference (field_ref_dir name is different)."

    # removed this check here -> want to be able to reload field and apply different
    # background method
    # if options["bfield_bground_method"] != prev_options["bfield_bground_method"]:
    #     return False, "different bfield_bground_method"

    # if options["bfield_bground_params"] != prev_options["bfield_bground_params"]:
    #     return False, "different bfield_bground_params"

    if options["diamond_ori"] != prev_options["diamond_ori"]:
        return False, "different diamond_ori"

    if options["field_method_used"] == "hamiltonian_fitting":
        if options["hamiltonian"] != prev_options["hamiltonian"]:
            return False, "different hamiltonian selected."

        guesser = qdmpy.hamiltonian.get_ham_guess_and_bounds
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

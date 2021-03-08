# -*- coding: utf-8 -*-
"""
This module holds tools for calculating Bxyz from Bnv.

Functions
---------
 - `QDMPy.field.interface.odmr_field_retrieval`
 - `QDMPy.field.interface.field_refsub`
 - `QDMPy.field.interface.get_B_bias`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.field.interface.odmr_field_retrieval": True,
    "QDMPy.field.interface.field_refsub": True,
    "QDMPy.field.interface.get_B_bias": True,
}
# ============================================================================

import warnings

# ============================================================================

import QDMPy.field._bnv as Qbnv
import QDMPy.field._bxyz as Qbxyz
import QDMPy.field._geom as Qgeom
import QDMPy.io as Qio

# ============================================================================


def odmr_field_retrieval(options, sig_fit_params, ref_fit_params):
    """
    fit results dict -> field results dict

    How to do bsub? well I guess do the bsub afterwards c:

    for AC fields/non-odmr datasets, need to write a new module.
    Yeah this is quite specific to things that contain bxyz data
    """
    # first check sig/ref consistency
    if not any(map(lambda x: x.startswith("pos"), sig_fit_params.keys())):
        raise RuntimeError("no 'pos' keys found in sig_fit_params")
    else:
        sig_poskey = next(filter(lambda x: x.startswith("pos"), sig_fit_params.keys()))[:3]

    if ref_fit_params:
        if not any(map(lambda x: x.startswith("pos"), ref_fit_params.keys())):
            raise RuntimeError("no 'pos_<something>' keys found in given ref_fit_params")
        else:
            ref_poskey = next(filter(lambda x: x.startswith("pos"), ref_fit_params.keys()))[:3]

        if not sig_fit_params[sig_poskey + "_0"].shape == ref_fit_params[ref_poskey + "_0"].shape:
            raise RuntimeError("Different FOV shape between sig & ref.")
        if not len(filter(lambda x: x.startswith("pos"), sig_fit_params)) == len(
            filter(lambda x: x.startswith("pos"), ref_fit_params)
        ):
            raise RuntimeError("Different number of frequencies fit in sig & ref.")

    # first get bnvs (as in global scope)
    sig_bnvs, sig_dshifts = Qbnv.get_bnvs_and_dshifts(sig_fit_params)
    ref_bnvs, ref_dshifts = Qbnv.get_bnvs_and_dshifts(ref_fit_params)

    meth = options["field_method"]
    if meth == "auto_dc" and not any(
        map(lambda x: x.startswith("pos"), options["fit_param_defn"])
    ):
        raise RuntimeError(
            """
            field_method 'auto_dc' not compatible with fit functions that do not include 'pos'
            parameters. You are probably not fitting ODMR data.
            Implement a module for non-odmr data, and an 'auto_ac' option for field retrieval in
            that regime. If you've done that you may implement an 'auto' option that selects the
            most applicable module and method :).
            """
        )

    # check how many peaks we want to use, and how many are available -> ensure compatible
    num_peaks_fit = len(options["pos_guess"])
    num_peaks_wanted = sum(options["freqs_to_use"])
    if num_peaks_wanted > num_peaks_fit:
        raise RuntimeError(
            f"Number of freqs wanted in option 'freqs_to_use' ({num_peaks_wanted})"
            + f"is greater than number fit ({num_peaks_fit}).\n"
            + "We need to identify which NVs each resonance corresponds to "
            + "for our algorithm to work, so please define this in the options dict/json."
        )
    # check that freqs_to_use is symmetric (necessary for bnvs retrieval methods)
    symmetric_freqs = list(reversed(options["freqs_to_use"][4:])) == options["freqs_to_use"][:4]

    if meth == "auto_dc":
        # need to select the appropriate one
        if num_peaks_wanted == 2:
            if symmetric_freqs:
                meth = "prop_single_bnv"
            else:
                meth = "hamiltonian_fitting"
        elif num_peaks_wanted == 6:
            if symmetric_freqs:
                meth = "invert_unvs"
            else:
                meth = "hamiltonian_fitting"
        elif num_peaks_wanted in [1, 3, 4, 5, 7, 8]:  # not sure how many of these will be useful
            meth = "hamiltonian_fitting"
        else:
            raise RuntimeError(
                "Number of true values in option 'freqs_to_use' is not between 1 and 8."
            )

    options["field_method_used"] = meth
    Qio.check_for_prev_field_calc(options)

    if options["found_prev_field_calc"]:
        warnings.warn("Using previous field calculation.")

        bnv_tuple, dshift_tuple, params_tuple, sigmas_tuple = Qio.load_prev_field_calcs(options)
        Qgeom.add_bfield_reconstructed(params_tuple[0])
        Qgeom.add_bfield_reconstructed(params_tuple[1])
        Qgeom.add_bfield_reconstructed(params_tuple[2])
        if params_tuple[0] is not None:
            options["field_params"] = tuple(params_tuple[0].keys())

    elif options["calc_field_pixels"]:
        if options["field_method_used"] == "prop_single_bnv":
            if num_peaks_wanted != 2:
                raise RuntimeError(
                    "field_method option was 'prop_single_bnv', but number of true values in option "
                    + "'freqs_to_use' was not 2."
                )
            else:
                sig_params = Qbxyz.from_single_bnv(options, sig_bnvs)
                ref_params = Qbxyz.from_single_bnv(options, ref_bnvs)
                sig_sigmas, ref_sigmas = None, None
        elif options["field_method_used"] == "invert_unvs":
            if num_peaks_wanted != 6:
                raise RuntimeError(
                    "field_method option was 'invert_unvs', but number of true values in option "
                    + "'freqs_to_use' was not 6."
                )
            else:
                sig_params = Qbxyz.from_unv_inversion(options, sig_bnvs)
                ref_params = Qbxyz.from_unv_inversion(options, ref_bnvs)
                sig_sigmas, ref_sigmas = None, None
        else:
            # hamiltonian fitting
            sig_params, sig_sigmas = Qbxyz.from_hamiltonian_fitting(options, sig_fit_params)
            ref_params, ref_sigmas = Qbxyz.from_hamiltonian_fitting(options, ref_fit_params)

        sub_ref_params = field_refsub(options, sig_params, ref_params)
        # for checking self-consistency (e.g. calc Bx from Bz via fourier methods) TODO
        # Qgeom.add_bfield_reconstructed(sig_params)
        # Qgeom.add_bfield_reconstructed(ref_params)
        # Qgeom.add_bfield_reconstructed(sub_ref_params)

        if sig_params is not None:
            options["field_params"] = tuple(sig_params.keys())
        else:
            options["field_params"] = None

        # both params and sigmas need a sub_ref method
        bnv_tuple = (sig_bnvs, ref_bnvs, Qbnv.bnv_refsub(options, sig_bnvs, ref_bnvs))
        dshift_tuple = (sig_dshifts, ref_dshifts)
        params_tuple = (sig_params, ref_params, sub_ref_params)
        sigmas_tuple = (sig_sigmas, ref_sigmas, field_sigma_add(options, sig_sigmas, ref_sigmas))

        Qio.save_field_calcs(options, bnv_tuple, dshift_tuple, params_tuple, sigmas_tuple)
    else:
        bnv_tuple, dshift_tuple, params_tuple, sigmas_tuple = (
            (None, None, None),
            (None, None),
            (None, None, None),
            (None, None, None),
        )
    return bnv_tuple, dshift_tuple, params_tuple, sigmas_tuple


# ============================================================================


def field_refsub(options, sig_params, ref_params):
    """
    sig - ref dictionaries, allow different options
    (blurred bground etc., see QDMPy.field._bnv.bnv_refsub)

    Don't need to be compatible, i.e. will only subtract params that exist in both dicts.
    """
    if ref_params:
        return {
            key: sig - ref_params[key] for (key, sig) in sig_params.items() if key in ref_params
        }
    else:
        return sig_params


# ============================================================================


def field_sigma_add(options, sig_sigmas, ref_sigmas):
    """ as field_refsub but we add sigmas (error propagation) """
    if ref_sigmas:
        return {
            key: sig + ref_sigmas[key] for (key, sig) in sig_sigmas.items() if key in ref_sigmas
        }
    else:
        return sig_sigmas


# ============================================================================


def get_B_bias(options):
    """
    Returns (bx, by, bz) for the bias field (supplied in options dict) in Gauss

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    bxyz : tuple
        (bx, by, bz) for the bias field, in Gauss.
    """
    return Qgeom.get_B_bias(options)

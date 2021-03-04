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
    """

    # first get bnvs (as in global scope)
    sig_bnvs, sig_dshifts = Qbnv.get_bnvs_and_dshifts(sig_fit_params)
    ref_bnvs, ref_dshifts = Qbnv.get_bnvs_and_dshifts(ref_fit_params)
    bnvs = Qbnv.bnv_refsub(options, sig_bnvs, ref_bnvs)

    Qio.save_bnvs_and_dshifts(options, "sig", sig_bnvs, sig_dshifts)
    Qio.save_bnvs_and_dshifts(options, "ref", ref_bnvs, ref_dshifts)
    Qio.save_bnvs_and_dshifts(options, "sig_sub_ref", bnvs, None)

    if options["hamiltonian"] not in ["approx_bxyz", "bxyz"]:
        # if hamiltonian not a simple bxyz => ham required to get other params, so force it.
        bmeth = "hamiltonian_fitting"
    else:
        # otherwise grab from options
        bmeth = options["bfield_method"]
    if bmeth == "auto_dc" and not any(
        map(lambda x: x.startswith("pos"), options["fit_param_defn"])
    ):
        raise RuntimeError(
            """
            bfield_method 'auto_dc' not compatible with fit functions that do not include 'pos'
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

    if bmeth == "auto_dc":
        # need to select the appropriate one
        if num_peaks_wanted == 2:
            if symmetric_freqs:
                bmeth = "prop_single_bnv"
            else:
                bmeth = "hamiltonian_fitting"
        elif num_peaks_wanted == 6:
            if symmetric_freqs:
                bmeth = "invert_unvs"
            else:
                bmeth = "hamiltonian_fitting"
        elif num_peaks_wanted in [1, 3, 4, 5, 7, 8]:  # not sure how many of these will be useful
            bmeth = "hamiltonian_fitting"
        else:
            raise RuntimeError(
                "Number of true values in option 'freqs_to_use' is not between 1 and 8."
            )

    if bmeth == "prop_single_bnv":
        if num_peaks_wanted != 2:
            raise RuntimeError(
                "bfield_method option was 'prop_single_bnv', but number of true values in option "
                + "'freqs_to_use' was not 2."
            )
        else:
            sig_params = Qbxyz.from_single_bnv(options, sig_bnvs)
            ref_params = Qbxyz.from_single_bnv(options, ref_bnvs)
            sig_sigmas, ref_sigmas = None, None
    elif bmeth == "invert_unvs":
        if num_peaks_wanted != 6:
            raise RuntimeError(
                "bfield_method option was 'invert_unvs', but number of true values in option "
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

    # for checking self-consistency (e.g. calc Bx from Bz via fourier methods)
    Qgeom.add_bfield_reconstructed(sig_params)
    Qgeom.add_bfield_reconstructed(ref_params)

    options["bfield_method_used"] = bmeth
    if sig_params is not None:
        options["field_params"] = tuple(sig_params.keys())
    else:
        options["field_params"] = None

    Qio.save_field_params(options, "sig", sig_params)
    Qio.save_field_params(options, "ref", ref_params)

    # TODO add subref params as output. Also combine sigmas
    return (
        (sig_bnvs, ref_bnvs, bnvs),
        (sig_dshifts, ref_dshifts),
        (sig_params, ref_params),
        (sig_sigmas, ref_sigmas),
    )


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

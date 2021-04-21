# -*- coding: utf-8 -*-
"""
This module holds tools for calculating Bxyz from Bnv.

Functions
---------
 - `qdmpy.field.interface.odmr_field_retrieval`
 - `qdmpy.field.interface.field_refsub`
 - `qdmpy.field.interface.field_sigma_add`
 - `qdmpy.field.interface.get_B_bias`
 - `qdmpy.field.interface.get_unvs`
 - `qdmpy.field.interface.add_bfield_reconstructed`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.field.interface.odmr_field_retrieval": True,
    "qdmpy.field.interface.field_refsub": True,
    "qdmpy.field.interface.field_sigma_add": True,
    "qdmpy.field.interface.get_B_bias": True,
    "qdmpy.field.interface.get_unvs": True,
    "qdmpy.field.interface.add_bfield_reconstructed": True,
}
# ============================================================================

import warnings
import numpy as np

# ============================================================================

import qdmpy.field._bnv as Qbnv
import qdmpy.field._bxyz as Qbxyz
import qdmpy.field._geom as Qgeom

import qdmpy.io as Qio
import qdmpy.fourier

# ============================================================================


def odmr_field_retrieval(options, sig_fit_params, ref_fit_params):
    """fit results dict -> field results dict

    For AC fields/non-odmr datasets, need to write a new module.
    Yeah this is quite specific to things that contain bxyz data

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    sig_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of fit param values across FOV.
    ref_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of fit param values across FOV.

    Returns
    -------
    bnv_lst : list
        List of bnv results (each a 2D image), [sig, ref, sig_sub_ref]
    dshift_lst : list
        List of dshift results (each a 2D image), [sig, ref]
    params_lst : list
        List of field parameters (each a dict), [sig_dict, ref_dict, sig_sub_ref_dict]
    sigmas_lst : list
        List of field sigmas (errors) (each a dict), [sig_dict, ref_dict, sig_sub_ref_dict]
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
        if not len(list(filter(lambda x: x.startswith("pos"), sig_fit_params))) == len(
            list(filter(lambda x: x.startswith("pos"), ref_fit_params))
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

        bnv_lst, dshift_lst, params_lst, sigmas_lst = Qio.load_prev_field_calcs(options)

        if options["bfield_bground_method"]:
            params_lst[2], sigmas_lst[2] = Qbxyz.sub_bground_Bxyz(
                options,
                params_lst[2],
                sigmas_lst[2],
                method=options["bfield_bground_method"],
                **options["bfield_bground_params"],
            )

        if options["bnv_bground_method"]:
            bnv_lst[2] = Qbnv.sub_bground_bnvs(
                options,
                bnv_lst[2],
                method=options["bnv_bground_method"],
                **options["bnv_bground_params"],
            )

        add_bfield_reconstructed(options, params_lst[0])
        add_bfield_reconstructed(options, params_lst[1])
        add_bfield_reconstructed(options, params_lst[2])

        if params_lst[0] is not None:
            options["field_params"] = tuple(params_lst[0].keys())

    elif options["calc_field_pixels"]:
        if options["field_method_used"] == "prop_single_bnv":
            if num_peaks_wanted != 2:
                raise RuntimeError(
                    "field_method option was 'prop_single_bnv', but number of true values in option "
                    + "'freqs_to_use' was not 2."
                )
            else:
                sig_params = Qbxyz.from_single_bnv(options, sig_bnvs)
                missing = np.empty(sig_params[list(sig_params.keys())[0]].shape)
                missing[:] = np.nan
                ref_params = Qbxyz.from_single_bnv(options, ref_bnvs)
                sig_sigmas = {key: missing for key in sig_params}
                ref_sigmas = None
        elif options["field_method_used"] == "invert_unvs":
            if num_peaks_wanted != 6:
                raise RuntimeError(
                    "field_method option was 'invert_unvs', but number of true values in option "
                    + "'freqs_to_use' was not 6."
                )
            else:
                sig_params = Qbxyz.from_unv_inversion(options, sig_bnvs)
                ref_params = Qbxyz.from_unv_inversion(options, ref_bnvs)
                missing = np.empty(sig_params[list(sig_params.keys())[0]].shape)
                missing[:] = np.nan
                sig_sigmas = {key: missing for key in sig_params}
                ref_sigmas = None
        else:
            # hamiltonian fitting
            sig_params, sig_sigmas = Qbxyz.from_hamiltonian_fitting(options, sig_fit_params)
            ref_params, ref_sigmas = Qbxyz.from_hamiltonian_fitting(options, ref_fit_params)

        sub_ref_params = Qbxyz.field_refsub(options, sig_params, ref_params)

        # both params and sigmas need a sub_ref method
        bnv_lst = [sig_bnvs, ref_bnvs, Qbnv.bnv_refsub(options, sig_bnvs, ref_bnvs)]
        dshift_lst = [sig_dshifts, ref_dshifts]
        params_lst = [sig_params, ref_params, sub_ref_params]
        sigmas_lst = [sig_sigmas, ref_sigmas, Qbxyz.field_sigma_add(options, sig_sigmas, ref_sigmas)]

        if options["bfield_bground_method"]:
            params_lst[2], sigmas_lst[2] = Qbxyz.sub_bground_Bxyz(
                options,
                params_lst[2],
                sigmas_lst[2],
                method=options["bfield_bground_method"],
                **options["bfield_bground_params"],
            )

        if options["bnv_bground_method"]:
            bnv_lst[2] = Qbnv.sub_bground_bnvs(
                options,
                bnv_lst[2],
                method=options["bnv_bground_method"],
                **options["bnv_bground_params"],
            )

        # for checking self-consistency (e.g. calc Bx from Bz via fourier methods)
        add_bfield_reconstructed(options, sig_params)
        add_bfield_reconstructed(options, ref_params)
        add_bfield_reconstructed(options, sub_ref_params)

        if sig_params is not None:
            options["field_params"] = tuple(sig_params.keys())
        else:
            options["field_params"] = None

    else:
        bnv_lst, dshift_lst, params_lst, sigmas_lst = (
            [sig_bnvs, ref_bnvs, Qbnv.bnv_refsub(options, sig_bnvs, ref_bnvs)],
            [sig_dshifts, ref_dshifts],
            [None, None, None],
            [None, None, None],
        )
        if options["bnv_bground_method"]:
            bnv_lst[2] = Qbnv.sub_bground_bnvs(
                options,
                bnv_lst[2],
                method=options["bnv_bground_method"],
                **options["bnv_bground_params"],
            )
    return bnv_lst, dshift_lst, params_lst, sigmas_lst


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


# ============================================================================


def get_unvs(options):
    """
    Returns orientation (relative to lab frame) of NVs. Shape: (4,3) regardless of sample.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    unvs : np array
        Shape: (4,3). Equivalent to uNV_Z for each NV.
    """
    return Qgeom.get_unvs(options)


# ============================================================================


def add_bfield_reconstructed(options, field_params):
    r"""Bxyz measured -> Bxyz_recon via fourier methods. Adds Bx_recon etc. to field_params.


    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options (for the main/signal experiment).
    field_params : dict
        Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV.

    Returns
    -------
    nothing (operates in place on field_params)

    For a proper explanation of methodology, see [CURR_RECON]_.

    $$  \nabla \times {\bf B} = 0 $$

    to get Bx_recon and By_recon from Bz (in a source-free region), and

    $$ \nabla \cdot {\bf B} = 0 $$

    to get Bz_recon from Bx and By

    Start with e.g.:

    $$ \frac{\partial B_x^{\rm recon}}{\partial z} = \frac{\partial B_z}{\partial x} $$

    with the definitions

    $$ \hat{B}:=  \hat{\mathcal{F}}_{xy} \big\{ B \big\} $$

    and

    $$ k = \sqrt{k_x^2 + k_y^2} $$

    we have:

    $$ \frac{\partial }{\partial z} \hat{B}_x^{\rm recon}(x,y,z=z_{\rm NV}) = i k_x \hat{B}_z(x,y, z=z_{\rm NV}) $$.

    Now using upward continuation [CURR_RECON]_ to evaluate the z partial:

    $$ -k \hat{B}_x^{\rm recon}(x,y,z=z_{\rm NV}) = i k_x \hat{B}_z(k_x, k_y, z_{\rm NV}) $$

    such that for

    $$ k \neq 0 $$

    we have (analogously for y)

    $$ (\hat{B}_x^{\rm recon}(x,y,z=z_{\rm NV}), \hat{B}_y^{\rm recon}(x,y,z=z_{\rm NV})) = \frac{-i}{k} (k_x, k_y) \hat{B}_z(x,y,,z=z_{\rm NV}) $$


    Utilising the zero-divergence property of the magnetic field, it can also be shown:

    $$ \hat{B}_z^{\rm recon}(x,y,z=z_{\rm NV}) = \frac{i}{k} \left( k_x \hat{B}_x(x,y,z=z_{\rm NV}) + k_y \hat{B}_y(x,y,z=z_{\rm NV}) \right) $$


    References
    ----------
    .. [CURR_RECON] E. A. Lima and B. P. Weiss,
                    Obtaining Vector Magnetic Field Maps from Single-Component Measurements of
                    Geological Samples, Journal of Geophysical Research: Solid Earth 114, (2009).
                    https://doi.org/10.1029/2008JB006006
    """
    # first check if Bx, By, Bz in fit_params
    # extract them
    if field_params is None:
        return None

    components = ["x", "y", "z"]

    for p in ["B" + comp for comp in components]:
        if p not in field_params:
            warnings.warn(f"bfield param '{p} missing from field_params, skipping bfield plot.")
            return None
        elif field_params[p] is None:
            return None

    bx, by, bz = [field_params["B" + comp] for comp in components]

    bx_recon, by_recon, bz_recon = qdmpy.fourier.get_reconstructed_bfield(
        [bx, by, bz],
        options["fourier_pad_mode"],
        options["fourier_pad_factor"],
        options["system"].get_raw_pixel_size(options) * options["total_bin"],
        options["fourier_k_vector_epsilon"],
    )
    field_params["Bx_recon"] = bx_recon
    field_params["By_recon"] = by_recon
    field_params["Bz_recon"] = bz_recon

    return None

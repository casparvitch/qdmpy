# -*- coding: utf-8 -*-
"""
This module holds tools for calculating Bxyz from Bnv.

Functions
---------
 - `qdmpy.field.interface.odmr_field_retrieval`
 - `qdmpy.field.interface._odmr_with_field_ref`
 - `qdmpy.field.interface._odmr_with_pre_glac_ref`
 - `qdmpy.field.interface.get_unvs`
 - `qdmpy.field.interface.add_bfield_reconstructed`
 - `qdmpy.field.interface.add_bfield_theta_phi`
 - `qdmpy.field.interface._check_fit_params_are_ok`
 - `qdmpy.field.interface.get_bnv_sd`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.field.interface.odmr_field_retrieval": True,
    "qdmpy.field.interface._odmr_with_field_ref": True,
    "qdmpy.field.interface._odmr_with_pre_glac_ref": True,
    "qdmpy.field.interface.get_unvs": True,
    "qdmpy.field.interface.add_bfield_reconstructed": True,
    "qdmpy.field.interface.add_bfield_theta_phi": True,
    "qdmpy.field.interface._check_fit_params_are_ok": True,
    "qdmpy.field.interface.get_bnv_sd": True,
}
# ============================================================================

import numpy as np

# ============================================================================

import qdmpy.field.bnv
import qdmpy.field.bxyz
import qdmpy.field.io
import qdmpy.shared.geom
from qdmpy.shared.misc import warn

# ===========================================================================


def odmr_field_retrieval(options, sig_fit_params, ref_fit_params):
    """fit results dict -> field results dict

    For AC fields/non-odmr datasets, need to write a new (sub-?)module.
    Yeah this is quite specific to things that contain dc bxyz data.

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

    if options["calc_field_pixels"]:
        _check_fit_params_are_ok(options, sig_fit_params, ref_fit_params)

    if options["exp_reference_type"] == "field":
        return _odmr_with_field_ref(options, sig_fit_params, ref_fit_params)
    elif options["exp_reference_type"] == "pre_gslac":
        return _odmr_with_pre_glac_ref(options, sig_fit_params, ref_fit_params)
    else:
        raise RuntimeError(
            f"exp_reference_type_type: {options['exp_reference_type']} not recognised."
        )


# ============================================================================


def _odmr_with_field_ref(options, sig_fit_params, ref_fit_params):
    """Calculate field, for case where we are using a field reference (even if field ref is None,
    as long as it isn't a pre_gslac etc.).

    Parameters
    ----------
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

    # first get bnvs (as in global scope)
    sig_bnvs, sig_dshifts = qdmpy.field.bnv.get_bnvs_and_dshifts(
        sig_fit_params, options["bias_field_spherical_deg_gauss"]
    )
    ref_bnvs, ref_dshifts = qdmpy.field.bnv.get_bnvs_and_dshifts(
        ref_fit_params, options["ref_bias_field_spherical_deg_gauss"]
    )

    qdmpy.field.io.choose_field_method(options)

    if options["calc_field_pixels"] and options["found_prev_field_calc"]:
        warn("Using previous field calculation.")

        (
            bnv_lst,
            dshift_lst,
            params_lst,
            sigmas_lst,
        ) = qdmpy.field.io.load_prev_field_calcs(options)

        if options["bfield_bground_method"]:
            params_lst[2], sigmas_lst[2] = qdmpy.field.bxyz.sub_bground_bxyz(
                options,
                params_lst[2],
                sigmas_lst[2],
                method=options["bfield_bground_method"],
                **options["bfield_bground_params"],
            )

        if options["bnv_bground_method"]:
            bnv_lst[2] = qdmpy.field.bnv.sub_bground_bnvs(
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
        num_peaks_wanted = sum(options["freqs_to_use"])
        if options["field_method_used"] == "prop_single_bnv":
            if num_peaks_wanted not in [1, 2]:
                raise RuntimeError(
                    "field_method option was 'prop_single_bnv', but number of true"
                    " values in option " + "'freqs_to_use' was not 1 or 2."
                )
            else:   
                sig_params = qdmpy.field.bxyz.from_single_bnv(options, sig_bnvs)
                missing = np.empty(sig_params[list(sig_params.keys())[0]].shape)
                missing[:] = np.nan
                ref_params = qdmpy.field.bxyz.from_single_bnv(options, ref_bnvs)
                sig_sigmas = {key: missing for key in sig_params}
                ref_sigmas = None
        elif options["field_method_used"] == "invert_unvs":
            if num_peaks_wanted != 6:
                raise RuntimeError(
                    "field_method option was 'invert_unvs', but number of true values"
                    " in option " + "'freqs_to_use' was not 6."
                )
            else:
                sig_params = qdmpy.field.bxyz.from_unv_inversion(options, sig_bnvs)
                ref_params = qdmpy.field.bxyz.from_unv_inversion(options, ref_bnvs)
                missing = np.empty(sig_params[list(sig_params.keys())[0]].shape)
                missing[:] = np.nan
                sig_sigmas = {key: missing for key in sig_params}
                ref_sigmas = None
        else:
            # hamiltonian fitting
            sig_params, sig_sigmas = qdmpy.field.bxyz.from_hamiltonian_fitting(
                options, sig_fit_params, options["bias_field_spherical_deg_gauss"]
            )
            ref_params, ref_sigmas = qdmpy.field.bxyz.from_hamiltonian_fitting(
                options, ref_fit_params, options["ref_bias_field_spherical_deg_gauss"]
            )

        sub_ref_params = qdmpy.field.bxyz.field_refsub(options, sig_params, ref_params)

        # both params and sigmas need a sub_ref method
        bnv_lst = [
            sig_bnvs,
            ref_bnvs,
            qdmpy.field.bnv.bnv_refsub(options, sig_bnvs, ref_bnvs),
        ]
        dshift_lst = [sig_dshifts, ref_dshifts]
        params_lst = [sig_params, ref_params, sub_ref_params]
        sigmas_lst = [
            sig_sigmas,
            ref_sigmas,
            qdmpy.field.bxyz.field_sigma_add(options, sig_sigmas, ref_sigmas),
        ]

        if options["bfield_bground_method"]:
            params_lst[2], sigmas_lst[2] = qdmpy.field.bxyz.sub_bground_bxyz(
                options,
                params_lst[2],
                sigmas_lst[2],
                method=options["bfield_bground_method"],
                **options["bfield_bground_params"],
            )

        if options["bnv_bground_method"]:
            bnv_lst[2] = qdmpy.field.bnv.sub_bground_bnvs(
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
            [
                sig_bnvs,
                ref_bnvs,
                qdmpy.field.bnv.bnv_refsub(options, sig_bnvs, ref_bnvs),
            ],
            [sig_dshifts, ref_dshifts],
            [None, None, None],
            [None, None, None],
        )
        if options["bnv_bground_method"]:
            bnv_lst[2] = qdmpy.field.bnv.sub_bground_bnvs(
                options,
                bnv_lst[2],
                method=options["bnv_bground_method"],
                **options["bnv_bground_params"],
            )
    return bnv_lst, dshift_lst, params_lst, sigmas_lst


# ============================================================================


def _odmr_with_pre_glac_ref(options, sig_fit_params, ref_fit_params):
    """Calculate field, for case where we are using a pre-gslac reference.
    This is a bit of an ad-hoc addon. Can't be reloaded etc.

    Note
    ----
    - assumes sig/ref are measured along the same unv
    - required 1 peak fit in sig, 2 in ref
    - doesn't currently work for ref past gslac (not sure how you could achieve that)
    - bnv background sub works on sig sub ref only (as in field case).
    - implied assumption that dshift does not depend on frequency (which is false)

    Parameters
    ----------
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

    # first get bnvs (as in global scope)
    sig_bnvs, sig_dshifts = qdmpy.field.bnv.get_bnvs_and_dshifts(
        sig_fit_params, options["bias_field_spherical_deg_gauss"]
    )
    ref_bnvs, ref_dshifts = qdmpy.field.bnv.get_bnvs_and_dshifts(
        ref_fit_params, options["ref_bias_field_spherical_deg_gauss"]
    )

    if not options["calc_field_pixels"]:
        bnv_lst, dshift_lst, params_lst, sigmas_lst = (
            [
                sig_bnvs,
                ref_bnvs,
                qdmpy.field.bnv.bnv_refsub(options, sig_bnvs, ref_bnvs),
            ],
            [sig_dshifts, ref_dshifts],
            [None, None, None],
            [None, None, None],
        )
        if options["bnv_bground_method"]:
            bnv_lst[2] = qdmpy.field.bnv.sub_bground_bnvs(
                options,
                bnv_lst[2],
                method=options["bnv_bground_method"],
                **options["bnv_bground_params"],
            )
    else:
        # always force field calc for this method (it's quick)
        if not ref_fit_params:
            raise RuntimeError(
                "with exp_reference_type = 'pre_gslac' you must define a reference."
            )
        else:
            warn("Using pre_gslac reference. Assuming unv is same for sig & ref.")
            # must match expected pattern
            num_freqs_sig = len(
                list(filter(lambda x: x.startswith("pos"), sig_fit_params))
            )
            num_freqs_ref = len(
                list(filter(lambda x: x.startswith("pos"), ref_fit_params))
            )
            if num_freqs_sig != 1:
                raise ValueError("num freqs fit (sig) for pre_gslac ref type is not 1.")
            if num_freqs_ref != 2:
                raise ValueError("num freqs fit (ref) for pre_gslac ref type is not 2.")

            chosen_freqs = options["freqs_to_use"]
            if sum(chosen_freqs) != 1:
                raise ValueError(
                    "Only select 1 freq ('freqs_to_use') for exp_reference_type:"
                    " pre_gslac."
                )
            if chosen_freqs[:4] == [
                0,
                0,
                0,
                0,
            ]:  # only single freq used, R transition rel to bias
                idx = np.argwhere(np.array(list(reversed(chosen_freqs[4:]))) == 1)[0][0]
            else:
                idx = np.argwhere(np.array(chosen_freqs[:4]) == 1)[0][0]

            # sig_bias = options["bias_field_spherical_deg_gauss"] not required..?
            ref_bias = options["ref_bias_field_spherical_deg_gauss"]
            # sig_bias_mag = np.abs(sig_bias[0]) not required..?
            ref_bias_mag = np.abs(ref_bias[0])

            if ref_bias_mag > qdmpy.field.bnv.GSLAC:
                raise RuntimeError(
                    "As currently coded, ref bias mag must be < GSLAC for dshift"
                    " reference."
                )

            unv = qdmpy.shared.geom.get_unvs(options)[idx]
            sig_bnv = sig_bnvs[0]
            ref_bnv = ref_bnvs[0]
            ref_dshift = ref_dshifts[0] / qdmpy.field.bnv.GAMMA

            # glac +- should be sorted in freq -> bnv
            sig_sub_ref_bnv = (
                sig_bnv + ref_dshift #if sig_bias_mag > GSLAC else sig_bnv - ref_dshift
            )

            other_opts = [
                options["fourier_pad_mode"],
                options["fourier_pad_factor"],
                options["system"].get_raw_pixel_size(options) * options["total_bin"],
                options["fourier_k_vector_epsilon"],
                options["NVs_above_sample"],
            ]

            sig_bxyz = qdmpy.field.bnv.prop_single_bnv(sig_bnv, unv, *other_opts)
            ref_bxyz = qdmpy.field.bnv.prop_single_bnv(ref_bnv, unv, *other_opts)
            sig_sub_ref_bxyz = qdmpy.field.bnv.prop_single_bnv(
                sig_sub_ref_bnv, unv, *other_opts
            )

            sig_params, ref_params, sub_ref_params = [
                {
                    "Bx": bxyz[0],
                    "By": bxyz[1],
                    "Bz": bxyz[2],
                    "residual_field": np.zeros((bxyz[2]).shape),
                }
                for bxyz in [sig_bxyz, ref_bxyz, sig_sub_ref_bxyz]
            ]
            missing = np.empty(sig_params[list(sig_params.keys())[0]].shape)
            missing[:] = np.nan
            sigmas = {key: missing for key in sig_params}

            bnv_lst = [sig_bnvs, ref_bnvs, [sig_sub_ref_bnv]]
            dshift_lst = [sig_dshifts, ref_dshifts]
            params_lst = [sig_params, ref_params, sub_ref_params]
            sigmas_lst = [sigmas, sigmas, sigmas]

            if options["bfield_bground_method"]:
                params_lst[2], sigmas_lst[2] = qdmpy.field.bxyz.sub_bground_bxyz(
                    options,
                    params_lst[2],
                    sigmas_lst[2],
                    method=options["bfield_bground_method"],
                    **options["bfield_bground_params"],
                )
            if options["bnv_bground_method"]:
                bnv_lst[2] = qdmpy.field.bnv.sub_bground_bnvs(
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

    return bnv_lst, dshift_lst, params_lst, sigmas_lst


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
    return qdmpy.shared.geom.get_unvs(options)


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
            warn(f"bfield param '{p} missing from field_params, skipping bfield plot.")
            return None
        elif field_params[p] is None:
            return None

    bx, by, bz = [field_params["B" + comp] for comp in components]

    bx_recon, by_recon, bz_recon = qdmpy.field.bxyz.get_reconstructed_bfield(
        [bx, by, bz],
        options["fourier_pad_mode"],
        options["fourier_pad_factor"],
        options["system"].get_raw_pixel_size(options) * options["total_bin"],
        options["fourier_k_vector_epsilon"],
        options["NVs_above_sample"],
    )
    field_params["Bx_recon"] = bx_recon
    field_params["By_recon"] = by_recon
    field_params["Bz_recon"] = bz_recon

    return None


# ============================================================================


def add_bfield_theta_phi(options, field_params, theta, phi):
    r"""Bxyz measured projected onto unit vector u: B_theta_phi (added to field params in-place)

    Calculates the magnetic field projected onto a given unit vector specified by
    theta (polar, from +z) and phi (azimuthal, from +x towards +y) angles in degrees.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options (for the main/signal experiment).
    field_params : dict
        Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV.
    theta : float
        Polar angle of unit vector to project onto, in degrees, from +z towards equator.
    phi : float
        Azimuthal angle of unit vector to project onto, in degrees, from +x towards +y.

    Returns
    -------
    nothing (operates in place on field_params)

    """
    # first check if Bx, By, Bz in fit_params
    # extract them
    if field_params is None:
        return None

    components = ["x", "y", "z"]

    for p in ["B" + comp for comp in components]:
        if p not in field_params:
            warn(f"bfield param '{p} missing from field_params, skipping bfield plot.")
            return None
        elif field_params[p] is None:
            return None

    bvec = np.array([field_params["B" + comp] for comp in components])

    ux = np.sin(theta) * np.cos(phi)
    uy = np.sin(theta) * np.sin(phi)
    uz = np.cos(theta)
    u = np.array([ux, uy, uz])
    uhat = u / np.linalg.norm(u)

    field_params["B_theta_phi"] = np.apply_along_axis(
        lambda b: np.dot(uhat, b), 0, bvec
    )
    options["bfield_proj_angles_(deg)"] = [theta, phi]

    return None


# ============================================================================


def add_bfield_proj_bias(options, field_params):
    """calls add_bfield_theta_phi but grabs angle from bias field"""
    mag, theta_deg, phi_deg = options["bias_field_spherical_deg_gauss"]
    add_bfield_theta_phi(options, field_params, theta_deg, phi_deg)
    return None


# ============================================================================


def _check_fit_params_are_ok(options, sig_fit_params, ref_fit_params):
    """Helper function to just ensure fit params are correct format etc. """
    if not any(map(lambda x: x.startswith("pos"), sig_fit_params.keys())):
        raise RuntimeError("No 'pos' keys found in sig_fit_params")
    else:
        sig_poskey = next(filter(lambda x: x.startswith("pos"), sig_fit_params.keys()))[
            :-2
        ]

    if ref_fit_params:
        if not any(map(lambda x: x.startswith("pos"), ref_fit_params.keys())):
            raise RuntimeError(
                "No 'pos_<something>' keys found in given ref_fit_params"
            )
        else:
            ref_poskey = next(
                filter(lambda x: x.startswith("pos"), ref_fit_params.keys())
            )[:-2]

        if (
            not sig_fit_params[sig_poskey + "_0"].shape
            == ref_fit_params[ref_poskey + "_0"].shape
        ):
            raise RuntimeError("Different FOV shape between sig & ref.")
        if options["exp_reference_type"] == "field" and not len(
            list(filter(lambda x: x.startswith("pos"), sig_fit_params))
        ) == len(list(filter(lambda x: x.startswith("pos"), ref_fit_params))):
            raise RuntimeError(
                "Different # of frequencies fit in sig & ref. (ref. type = field)."
            )


# ============================================================================


def get_ham_guess_and_bounds(options):
    """
    Generate initial guesses (and bounds) in fit parameters from options dictionary.

    Both are returned as dictionaries, you need to use 'gen_{scipy/gpufit/...}_init_guesses'
    to convert to the correct (array) format for each specific fitting backend.


    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    init_guesses : dict
        Dict holding guesses for each parameter, e.g. key -> list of guesses for each independent
        version of that fn_type.
    init_bounds : dict
        Dict holding guesses for each parameter, e.g. key -> list of bounds for each independent
        version of that fn_type.
    """
    return qdmpy.field.hamiltonian.ham_gen_init_guesses(options)


# ============================================================================


def get_bnv_sd(sigmas):
    """ get standard deviation of bnvs given SD of peaks. """
    return qdmpy.field.bnv.get_bnv_sd(sigmas)

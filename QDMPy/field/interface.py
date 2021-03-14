# -*- coding: utf-8 -*-
"""
This module holds tools for calculating Bxyz from Bnv.

Functions
---------
 - `QDMPy.field.interface.odmr_field_retrieval`
 - `QDMPy.field.interface.field_refsub`
 - `QDMPy.field.interface.sub_bground_Bxyz`
 - `QDMPy.field.interface.sub_bground_bnvs`
 - `QDMPy.field.interface.field_sigma_add`
 - `QDMPy.field.interface.get_B_bias`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.field.interface.odmr_field_retrieval": True,
    "QDMPy.field.interface.field_refsub": True,
    "QDMPy.field.interface.sub_bground_Bxyz": True,
    "QDMPy.field.interface.sub_bground_bnvs": True,
    "QDMPy.field.interface.field_sigma_add": True,
    "QDMPy.field.interface.get_B_bias": True,
}
# ============================================================================

import warnings
import numpy as np

# ============================================================================

import QDMPy.field._bnv as Qbnv
import QDMPy.field._bxyz as Qbxyz
import QDMPy.field._geom as Qgeom
import QDMPy.io as Qio
import QDMPy.itools as Qitools

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
        # Qgeom.add_bfield_reconstructed(params_lst[0])
        # Qgeom.add_bfield_reconstructed(params_lst[1])
        # Qgeom.add_bfield_reconstructed(params_lst[2])

        if options["bfield_bground_method"]:
            params_lst[2], sigmas_lst[2] = sub_bground_Bxyz(
                options,
                params_lst[2],
                sigmas_lst[2],
                method=options["bfield_bground_method"],
                **options["bfield_bground_params"],
            )

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

        sub_ref_params = field_refsub(options, sig_params, ref_params)
        # for checking self-consistency (e.g. calc Bx from Bz via fourier methods) TODO
        # Qgeom.add_bfield_reconstructed(sig_params)
        # Qgeom.add_bfield_reconstructed(ref_params)
        # Qgeom.add_bfield_reconstruczted(sub_ref_params)

        # both params and sigmas need a sub_ref method
        bnv_lst = [sig_bnvs, ref_bnvs, Qbnv.bnv_refsub(options, sig_bnvs, ref_bnvs)]
        dshift_lst = [sig_dshifts, ref_dshifts]
        params_lst = [sig_params, ref_params, sub_ref_params]
        sigmas_lst = [sig_sigmas, ref_sigmas, field_sigma_add(options, sig_sigmas, ref_sigmas)]

        if options["bfield_bground_method"]:
            params_lst[2], sigmas_lst[2] = sub_bground_Bxyz(
                options,
                params_lst[2],
                sigmas_lst[2],
                method=options["bfield_bground_method"],
                **options["bfield_bground_params"],
            )

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
            bnv_lst[2] = sub_bground_bnvs(
                options,
                bnv_lst[2],
                method=options["bnv_bground_method"],
                **options["bnv_bground_params"],
            )
    return bnv_lst, dshift_lst, params_lst, sigmas_lst


# ============================================================================


def field_refsub(options, sig_params, ref_params):
    """Calculate sig - ref dict.

    Don't need to be compatible, i.e. will only subtract params that exist in both dicts.


    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options (for the main/signal experiment).
    sig_params : dict
        Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV.
    ref_params : dict
        Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV.

    Returns
    -------
    sig_sub_ref_params : dict
        sig - ref dictionary
    """
    if ref_params:
        return {
            key: sig - ref_params[key] for (key, sig) in sig_params.items() if key in ref_params
        }
    else:
        return sig_params.copy()


# ============================================================================


def sub_bground_Bxyz(options, field_params, field_sigmas, method, **method_settings):
    """Calculate and subtract a background from the Bx, By and Bz keys in params and sigmas

    Methods available for background calculation:
        Methods available (& required params in method_settings):
        - "fix_zero"
            - Fix background to be a constant offset (z value)
            - params required in method_settings:
                "zero" an int/float, defining the constant offset of the background
        - "three_point"
            - Calculate plane background with linear algebra from three [x,y] lateral positions
              given
            - params required in method_settings:
                - "points" a len-3 iterable containing [x, y] points
        - "mean"
            - background calculated from mean of image
            - no params required
        - "poly"
            - background calculated from polynomial fit to image.
            - params required in method_settings:
                - "order": an int, the 'order' polynomial to fit. (e.g. 1 = plane).
        - "gaussian"
            - background calculated from gaussian fit to image.
            - no params required
        - "interpolate"
            - Background defined by the dataset smoothed via a sigma-gaussian filtering,
                and method-interpolation over masked (polygon) regions.
            - params required in method_settings:
                - "method":
                - "sigma": sigma passed to gaussian filter (see scipy.ndimage.gaussian_filter)
                    which is utilized on the background before interpolating
        - "gaussian_filter"
            - background calculated from image filtered with a gaussian filter.
            - params required in method_settings:
                - "sigma": sigma passed to gaussian filter (see scipy.ndimage.gaussian_filter)

    See `QDMPy.itools.interface.get_background` for implementation etc.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options (for the main/signal experiment).
    field_params : dict
        Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV.
    field_sigmas : dict
        Dictionary, key: param_keys, val: image (2D) of (field) sigma (error) values across FOV.
    method : str
        Method to use for background subtraction. See above for details.
    **method_settings : dict
        (i.e. keyword arguments).
        Parameters passed to background subtraction algorithm. See above for details

    Returns
    -------
    field_params : dict
        Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV.
        Now with keys: "Bx_full" (unsubtracted), "Bx_bground", and "Bx" which has bground subbed.
    field_sigmas : dict
        Dictionary, key: param_keys, val: image (2D) of (field) sigma (error) values across FOV.
        Now with keys: "Bx_full" (unsubtracted), "Bx_bground", and "Bx" which has bground subbed.
    """
    if not field_params:
        return field_params, field_sigmas

    for b in ["Bx", "By", "Bz"]:
        if b not in field_params:
            warnings.warn("no B params found in field_params? Doing nothing.")
            return field_params, field_sigmas

    if "polygons" in options and (options["mask_polygons_bground"] or method == "interpolate"):
        polygons = options["polygons"]
    else:
        polygons = None
    x_bground = Qitools.get_background(
        field_params["Bx"], method, polygons=polygons, **method_settings
    )
    y_bground = Qitools.get_background(
        field_params["By"], method, polygons=polygons, **method_settings
    )
    z_bground = Qitools.get_background(
        field_params["Bz"], method, polygons=polygons, **method_settings
    )

    field_params["Bx_bground"] = x_bground
    field_params["By_bground"] = y_bground
    field_params["Bz_bground"] = z_bground

    field_params["Bx_full"] = field_params["Bx"]
    field_params["By_full"] = field_params["By"]
    field_params["Bz_full"] = field_params["Bz"]

    field_params["Bx"] = field_params["Bx_full"] - x_bground
    field_params["By"] = field_params["By_full"] - y_bground
    field_params["Bz"] = field_params["Bz_full"] - z_bground

    if (
        field_sigmas is not None
        and "Bx" in field_sigmas
        and "By" in field_sigmas
        and "Bz" in field_sigmas
    ):
        field_sigmas["Bx_full"] = field_sigmas["Bx"]
        field_sigmas["By_full"] = field_sigmas["By"]
        field_sigmas["Bz_full"] = field_sigmas["Bz"]

        missing = np.empty(field_sigmas["Bx"].shape)
        missing[:] = np.nan
        field_sigmas["Bx_bground"] = missing
        field_sigmas["By_bground"] = missing
        field_sigmas["Bz_bground"] = missing
        # leave field_sigmas["Bx"] etc. the same

    return field_params, field_sigmas


# ============================================================================


def sub_bground_bnvs(options, bnvs, method, **method_settings):
    """Subtract a background from the bnvs.

    Methods available for background calculation: TODO

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options (for the main/signal experiment).
    bnvs : list
        List of bnvs images (2D ndarrays)
    method : str
        Method to use for background subtraction. See above for details.
    **method_settings : dict
        (i.e. keyword arguments).
        Parameters passed to background subtraction algorithm. See above for details

    Returns
    -------
    output_bnvs
        bnvs with background subtracted
    """
    if options["mask_polygons_bground"] and "polygons" in options:
        polygons = options["polygons"]
    else:
        polygons = None
    output_bnvs = []
    for bnv in bnvs:
        if not bnvs:
            continue
        bground = Qitools.get_background(bnv, method, polygons=polygons, **method_settings)
        output_bnvs.append(bnv - bground)

    return output_bnvs


# ============================================================================


def field_sigma_add(options, sig_sigmas, ref_sigmas):
    """as QDMPy.field.interface.field_refsub` but we add sigmas (error propagation).

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options (for the main/signal experiment).
    sig_sigmas : dict
        Dictionary, key: param_keys, val: image (2D) of (field) sigma (error) values across FOV
        for the signal experiment.
    ref_sigmas : dict
        Dictionary, key: param_keys, val: image (2D) of (field) sigma (error) values across FOV
        for the reference experiment.

    Returns
    -------
    sig_sub_ref_sigmas : dict
        Same as sig_sigmas, but with ref subtracted.
    """
    if ref_sigmas:
        return {
            key: sig + ref_sigmas[key] for (key, sig) in sig_sigmas.items() if key in ref_sigmas
        }
    else:
        return sig_sigmas.copy()


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

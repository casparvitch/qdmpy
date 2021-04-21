# -*- coding: utf-8 -*-
"""
This module holds tools for calculating the vector magnetic field via
different methods.

Functions
---------
 - `qdmpy.field._bxyz.from_single_bnv`
 - `qdmpy.field._bxyz.from_unv_inversion`
 - `qdmpy.field._bxyz.from_hamiltonian_fitting`
 - `qdmpy.field._bxyz.sub_bground_Bxyz`
 - `qdmpy.field._bxyz.field_refsub`
 - `qdmpy.field._bxyz.field_sigma_add`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.field._bxyz.from_single_bnv": True,
    "qdmpy.field._bxyz.from_unv_inversion": True,
    "qdmpy.field._bxyz.from_hamiltonian_fitting": True,
    "qdmpy.field._bxyz.sub_bground_Bxyz": True,
    "qdmpy.field._bxyz.field_refsub": True,
    "qdmpy.field._bxyz.field_sigma_add": True,
}
# ============================================================================

import numpy as np
import numpy.linalg as LA

# ============================================================================

import qdmpy.field._geom as Qgeom
import qdmpy.field._bnv as Qbnv
import qdmpy.hamiltonian as Qham
import qdmpy.fourier
import qdmpy.itool as Qitool

# ============================================================================


def from_single_bnv(options, bnvs):
    """
    Use fourier propagation to take a single bnv to vector field (Bx, By, Bz).
    Heavily influenced by propagation artifacts.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    bnvs : list
        List of np arrays (2D) giving B_NV for each NV family/orientation.
        If num_peaks is odd, the bnv is given as the shift of that peak,
        and the dshifts is left as np.nans.
        if [], returns None

    Returns
    -------
    field_result : dict
        Dictionary, key: param_keys (Bx, By, Bz), val: image (2D) of param values across FOV.
        Also contains "residual_field" as a key/val.
    """
    if not bnvs:
        return None

    chosen_freqs = options["freqs_to_use"]
    if len(bnvs) > 1 and not list(reversed(chosen_freqs[4:])) == chosen_freqs[:4]:
        raise ValueError(
            """
            'field_method' method was 'prop_single_bnv' with more than one bnv,
            but option 'freqs_to_use' was not symmetric.
            Change method to 'auto_dc' or 'hamiltonian_fitting'.
            """
        )
    if len(bnvs) > 1 and sum(chosen_freqs) != 2:
        raise ValueError("Only 2 freqs should be chosen for the 'prop_single_bnv' method.")

    unvs = Qgeom.get_unvs(options)
    if len(bnvs) == 1:  # just use the first one (i.e. the only one...)
        single_bnv = bnvs[0]
        unv = unvs[0]
    elif len(bnvs) == 4:  # just use the chosen freq
        idx = np.argwhere(np.array(chosen_freqs[:4]) == 1)[0][0]
        single_bnv = bnvs[idx]
        unv = unvs[idx]
    else:  # need to use 'single_bnv_choice' option to resolve amiguity.
        single_bnv = bnvs[options["single_bnv_choice"] - 1]
        unv = bnvs[options["single_bnv_choice"] - 1]

    # (get unv data from chosen freqs, method in _geom)
    # (then follow methodology in DB code -> import a `fourier` module)

    bxyzs = qdmpy.fourier.prop_single_bnv(
        single_bnv,
        unv,
        options["fourier_pad_mode"],
        options["fourier_pad_factor"],
        options["system"].get_raw_pixel_size(options) * options["total_bin"],
        options["fourier_k_vector_epsilon"],
    )
    return {
        "Bx": bxyzs[0],
        "By": bxyzs[1],
        "Bz": bxyzs[2],
        "residual_field": np.zeros((bxyzs[2]).shape),  # no residual as there's no fit
    }


# ============================================================================


def from_unv_inversion(options, bnvs):
    r"""
    Diagonal Hamiltonian approximation. To calculate the magnetic field only fields aligned with
    the NV are considered and thus a simple dot product can be used.

    Instead of fitting to bnvs via:
    $$ \overline{\overline{B}}_{\rm NV} = overline{\overline{u}}_{\rm NV} \cdot \overline{B} $$

    instead (with 3 bnvs) just calculate inverse of unvs:
    $$ \overline{B} = overline{\overline{u}}_{\rm NV}^{-1} \cdot \overline{\overline{B}}_{\rm NV} $$

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    bnvs : list
        List of np arrays (2D) giving B_NV for each NV family/orientation.
        If num_peaks is odd, the bnv is given as the shift of that peak,
        and the dshifts is left as np.nans.
        if [], returns None

    Returns
    -------
    field_result : dict
        Dict of bfield, keys: ["Bx", "By", "Bz"], vals: image of those vals (2D np array)
        Also contains "residual_field" as a key/val.
    """
    if not bnvs:
        return None

    chosen_freqs = options["freqs_to_use"]

    if sum(chosen_freqs) != 6:
        raise ValueError("Only 6 freqs should be chosen for the 'invert_unvs' method.")

    if not len(bnvs) >= 3:
        raise ValueError(
            "'field_method' was 'invert_unvs' but there were not 3 or 4 bnvs in the dataset."
        )

    unvs = Qgeom.get_unvs(options)  # z unit vectors of unv frame (in lab frame) = nv orientations

    # cut unvs down to only bnvs (freqs) we want

    # first assert chosen freqs is symmetric
    if not list(reversed(chosen_freqs[4:])) == chosen_freqs[:4]:
        raise ValueError(
            """
            'field_method' was 'invert_unvs' but option 'freqs_to_use' was not
            symmetric. Change method to 'auto_dc' or 'hamiltonian_fitting'.
            """
        )
    nv_idxs_to_use = [i for i, do_use in enumerate(chosen_freqs[:4]) if do_use]
    if len(bnvs) == 3:
        bnvs_to_use = bnvs  # if only 3 bnvs passed, well we just use em all :)
    else:
        bnvs_to_use = [bnvs[j] for j in nv_idxs_to_use]
    unvs_to_use = np.vstack([unvs[j] for j in nv_idxs_to_use])  # 3x3 matrix

    unv_inv = LA.inv(unvs_to_use)

    # reshape bnvs to be [bnv_1, bnv_2, bnv_3] for each pixel of image
    bnvs_reshaped = np.stack(bnvs_to_use, axis=-1)

    # unv_inv * [bnv_1, bnv_2, bnv_3] for pxl in image -> VERY fast. (applied over last axis)
    bxyzs = np.apply_along_axis(lambda bnv_vec: np.matmul(unv_inv, bnv_vec), -1, bnvs_reshaped)

    return {
        "Bx": bxyzs[:, :, 0],
        "By": bxyzs[:, :, 1],
        "Bz": bxyzs[:, :, 2],
        "residual_field": np.zeros((bxyzs[:, :, 2]).shape),  # no residual as there's no fit
    }


# ============================================================================


def from_hamiltonian_fitting(options, fit_params):
    """
    (PL fitting) fit_params -> (freq/bnvs fitting) ham_results.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        (fit results from PL fitting).
        Also has 'residual' as a key.
        If None, returns None

    Returns
    -------
    ham_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual_field' as a key.
    sigmas : dict
        as ham_results but each val contains te sigmas (errors) for that param.
    """
    if fit_params is None:
        return None, None

    use_bnvs = options["hamiltonian"] in ["approx_bxyz"]

    if use_bnvs:
        if not list(reversed(options["freqs_to_use"][4:])) == options["freqs_to_use"][:4]:
            raise ValueError(
                "'hamiltonian' option used bnvs, but chosen frequencies are not symmetric."
            )
        chooser_ar = options["freqs_to_use"][:4]  # i.e. bnv chooser
    else:
        chooser_ar = options["freqs_to_use"]

    # if user doesn't want to fit all freqs, then don't fit em! (with Chooser obj.)
    ham = Qham.define_hamiltonian(options, Qham.Chooser(chooser_ar), Qgeom.get_unv_frames(options))

    # ok now need to get useful data out of fit_params (-> data)
    if use_bnvs:
        # data shape: [bnvs/freqs, y, x]
        bnv_lst, _ = Qbnv.get_bnvs_and_dshifts(fit_params)
        if sum(chooser_ar) < 4:
            unwanted_bnvs = np.argwhere(np.array(chooser_ar) == 0)[0]
            shape = bnv_lst[0].shape
            missings = np.empty(shape)
            missings[:] = np.nan
            full_bnv_lst = []
            for i in range(4):  # insert 'missing' bnvs (as nans)
                if i in unwanted_bnvs:
                    full_bnv_lst.append(missings)
                else:
                    full_bnv_lst.append(bnv_lst.pop(0))
            data = np.array(full_bnv_lst)
        else:
            data = np.array(bnv_lst)
    else:
        # get pos_param_name
        # use freqs, same data shape
        freqs_given_lst = []
        for param_name, param_map in fit_params.items():
            if param_name.startswith("pos"):
                freqs_given_lst.append(param_map)

        shape = freqs_given_lst[0].shape
        missings = np.empty(shape)
        missings[:] = np.nan
        freq_lst = []
        for i, do_use in enumerate(options["freqs_to_use"]):
            if not do_use:
                freq_lst.append(missings)
            else:
                freq_lst.append(freqs_given_lst.pop(0))
        data = np.array(freq_lst)

    return Qham.fit_hamiltonian_pixels(options, data, ham)


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

    See `qdmpy.itool.interface.get_background` for implementation etc.

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
    x_bground = Qitool.get_background(
        field_params["Bx"], method, polygons=polygons, **method_settings
    )
    y_bground = Qitool.get_background(
        field_params["By"], method, polygons=polygons, **method_settings
    )
    z_bground = Qitool.get_background(
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
    def subtractor(key, sig, ref_params):
        if ref_params[key] is not None:
            return sig - ref_params[key]
        else:
            return sig
    if ref_params:
        return {
            key: subtractor(key, sig, ref_params) for (key, sig) in sig_params.items() if key in ref_params
        }
    else:
        return sig_params.copy()



# ============================================================================


def field_sigma_add(options, sig_sigmas, ref_sigmas):
    """as qdmpy.field.interface.field_refsub` but we add sigmas (error propagation).

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
    def adder(key, sig, ref_sigmas):
        if ref_sigmas[key] is not None:
            return sig + ref_sigmas[key]
        else:
            return sig
    if ref_sigmas:
        return {
            key: adder(key, sig, ref_sigmas) for (key, sig) in sig_sigmas.items() if key in ref_sigmas
        }
    else:
        return sig_sigmas.copy()

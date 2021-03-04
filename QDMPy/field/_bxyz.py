# -*- coding: utf-8 -*-
"""
This module holds tools for ...

Functions
---------
 - `QDMPy.field._bxyz.bxyz_from_single_bnv`
 - `QDMPy.field._bxyz.bxyz_from_unv_inversion`
 - `QDMPy.field._bxyz.from_hamiltonian_fitting`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.field._bxyz.bxyz_from_single_bnv": True,
    "QDMPy.field._bxyz.bxyz_from_unv_inversion": True,
    "QDMPy.field._bxyz.from_hamiltonian_fitting": True,
}
# ============================================================================

import numpy as np
import numpy.linalg as LA

# ============================================================================

import QDMPy.field._geom as Qgeom
import QDMPy.field._bnv as Qbnv
import QDMPy.hamiltonian as Qham


# ============================================================================
# TODO


def from_single_bnv(options, bnvs):
    """
    TODO
    Use fourier propagation to take a single bnv to vector field (Bx, By, Bz).
    Heavily influenced by propagation artifacts.

    Arguments
    ---------
    bnvs : list
        List of np arrays (2D) giving B_NV for each NV family/orientation.
        If num_peaks is odd, the bnv is given as the shift of that peak,
        and the dshifts is left as np.nans.
        if [], returns None

    Returns
    -------
    param_results : dict
        Dictionary, key: param_keys (Bx, By, Bz), val: image (2D) of param values across FOV.
    """
    if not bnvs:
        return None

    chosen_freqs = options["freqs_to_use"]
    if not all(list(reversed(chosen_freqs[4:])) == chosen_freqs[:4]):
        raise ValueError(
            """
            'bfield_method' method was 'prop_single_bnv' but option 'freqs_to_use' was not
            symmetric. Change method to 'auto_dc' or 'hamiltonian_fitting'.
            """
        )
    if sum(chosen_freqs) != 2:
        raise ValueError("Only 2 freqs should be chosen for the 'prop_single_bnv' method.")

    if len(bnvs) == 1:  # just use the first one
        single_bnv = bnvs[0]
    elif len(bnvs) == 4:  # just use the chosen freq
        single_bnv = bnvs[np.argwhere(np.array(chosen_freqs[:4]) == 1)[0][0]]
    else:  # need to use 'single_bnv_choice' option to resolve amiguity.
        single_bnv = bnvs[options["single_bnv_choice"] + 1]

    # (get unv data from chosen freqs, method in _geom)
    # (then follow methodology in DB code -> import a `fourier` module, separeate to source.)
    raise NotImplementedError()


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
        if [], return None

    Returns
    -------
    Bxyz : dict
        Dict of bfield, keys: ["Bx", "By", "Bz"], vals: image of those vals (2D np array)
    """
    if not bnvs:
        return None

    chosen_freqs = options["freqs_to_use"]

    if sum(chosen_freqs) != 6:
        raise ValueError("Only 6 freqs should be chosen for the 'invert_unvs' method.")

    if not len(bnvs) >= 3:
        raise ValueError(
            "'bfield_method' was 'invert_unvs' but there were not 3 or 4 bnvs in the dataset."
        )

    unvs = Qgeom.get_unvs(options)  # z unit vectors of unv frame (in lab frame) = nv orientations

    # cut unvs down to only bnvs (freqs) we want

    # first assert chosen freqs is symmetric
    if not list(reversed(chosen_freqs[4:])) == chosen_freqs[:4]:
        raise ValueError(
            """
            'bfield_method' was 'invert_unvs' but option 'freqs_to_use' was not
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

    # unv_inv * [bnv_1, bnv_2, bnv_3] for pxl in image -> VERY fast.
    bxyzs = np.apply_along_axis(lambda bnv_vec: np.matmul(unv_inv, bnv_vec), -1, bnvs_reshaped)

    return {"Bx": bxyzs[:, :, 0], "By": bxyzs[:, :, 1], "Bz": bxyzs[:, :, 2]}


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
        Also has 'residual' as a key.
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
        # use freqs, same data shape
        freqs_given_lst = [fit_params[f"pos_{j}"] for j in range(8) if f"pos_{j}" in fit_params]

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

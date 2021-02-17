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
import QDMPy.hamiltonian as Qham

# ============================================================================


# TODO
# - This should be a propagation for just one bnv.
def from_single_bnv(options, bnvs):
    """
    TODO
    Arguments
    ---------
    bnvs : list
        List of np arrays (2D) giving B_NV for each NV family/orientation.
        If num_peaks is odd, the bnv is given as the shift of that peak,
        and the dshifts is left as np.nans.

    Returns
    -------
    TODO
    """
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

    Returns
    -------
    Bxyz : dict
        Dict of bfield, keys: ["Bx", "By", "Bz"], vals: image of those vals (2D np array)
    """
    chosen_freqs = options["freqs_to_use"]

    if sum(chosen_freqs) != 6:
        raise ValueError("Only 6 freqs should be chosen for the 'invert_unvs' method.")

    unvs = Qgeom.get_unvs(options)  # z unit vectors of unv frame (in lab frame) = nv orientations

    # cut unvs down to only bnvs (freqs) we want

    # first assert chosen freqs is symmetric
    if not all(list(reversed(chosen_freqs[4:])) == chosen_freqs[:4]):
        raise ValueError(
            """
            'bfield_method' method was 'invert_unvs' but option 'freqs_to_use' was not
            symmetric. Change method to 'auto_dc' or 'hamiltonian_fitting'.
            """
        )
    nv_idxs_to_use = [i for i, do_use in enumerate(chosen_freqs[:4]) if do_use]
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
    TODO

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        (fit results from PL fitting).
        Also has 'residual' as a key.

    Returns
    -------
    ham_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    """

    if options["hamiltonian"] in ["approx_bxyz"]:
        if not all(list(reversed(options["freqs_to_use"][4:])) == options["freqs_to_use"][:4]):
            raise ValueError(
                "'hamiltonian' chosen was 'approx_bxyz', but chosen frequencies are not symmetric."
            )
        chooser = options["freqs_to_use"][:4]
    else:
        chooser = options["freqs_to_use"]

    # if user doesn't want to fit all freqs, then don't fit em!
    def _indices_fn(ar):
        return [ar[i] for i, do_use in enumerate(chooser) if do_use]

    ham = Qham.define_hamiltonian(options, _indices_fn, Qgeom.get_unv_frames(options))

    # ok now need to get useful data out of fit_params (-> data)
    # data shape: [bnvs/freqs, y, x]

    return Qham.fit_hamiltonian_pixels(options, data, ham)

# -*- coding: utf-8 -*-
"""
The module holds Hamiltonian objects that will be fit to.

Classes
-------
 - `qdmpy.hamiltonian._hamiltonians.Hamiltonian`
 - `qdmpy.hamiltonian._hamiltonians.ApproxBxyz`
 - `qdmpy.hamiltonian._hamiltonians.Bxyz`

Functions
---------
 - `qdmpy.hamiltonian._hamiltonians.get_param_defn`
 - `qdmpy.hamiltonian._hamiltonians.get_param_odict`
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.hamiltonian._hamiltonians.Hamiltonian": True,
    "qdmpy.hamiltonian._hamiltonians.ApproxBxyz": True,
    "qdmpy.hamiltonian._hamiltonians.Bxyz": True,
    "qdmpy.hamiltonian._hamiltonians.get_param_defn": True,
    "qdmpy.hamiltonian._hamiltonians.get_param_odict": True,
}

# ============================================================================

import numpy as np
import numpy.linalg as LA
from collections import OrderedDict

# ============================================================================


class Hamiltonian:

    param_defn = []
    param_units = {}
    jac_defined = False

    def __init__(self, chooser_obj, unv_frames):
        """
        chooser_obj is used on __call__ and measured_data to return an array
        of only the required parts.
        """
        self.chooser_obj = chooser_obj
        self.unv_frames = unv_frames
        self.unvs = unv_frames[:, 2, :].copy()  # i.e. z axis of each nv ref. frame in lab frame

    # =================================

    def __call__(self, param_ar):
        """
        Evaluates Hamiltonian for given parameter values.

        Arguments
        ---------
        param_ar : np array, 1D
            Array of hamiltonian parameters fed in.
        """
        raise NotImplementedError("You MUST override __call__, check your spelling.")

    # =================================

    def grad_fn(self, param_ar):
        """
        Return jacobian, shape: (len(bnvs/freqs), len(param_ar))
        Each column is a partial derivative, with respect to each param in param_ar
            (i.e. rows, or first index, is indexing though the bnvs/freqs.)
        """
        raise NotImplementedError("No grad_fn defined for this Hamiltonian.")

    # =================================

    def residuals_scipyfit(self, param_ar, measured_data):
        """
        Evaluates residual: fit model - measured_data. Returns a vector!
        Measured data must be a np array (of the same shape that __call__ returns),
        i.e. freqs, or bnvs.
        """
        return self.chooser_obj(self.__call__(param_ar)) - self.chooser_obj(measured_data)

    # =================================

    def jacobian_scipyfit(self, param_ar, measured_data):
        """Evaluates (analytic) jacobian of ham in format expected by scipy least_squares."""

        # need to take out rows (first index) according to chooser_obj.
        keep_rows = self.chooser_obj([i for i in range(len(measured_data))])
        delete_rows = [r for r in range(len(measured_data)) if r not in keep_rows]
        return np.delete(self.grad_fn(param_ar), delete_rows, axis=0)

    # =================================

    def jacobian_defined(self):
        return self.jac_defined


# ============================================================================


def get_param_defn(hamiltonian):
    return hamiltonian.param_defn


# ====================================


def get_param_odict(hamiltonian):
    """
    get ordered dict of key: param_key (param_name), val: param_unit for all parameters in ham
    """
    return OrderedDict(hamiltonian.param_units)


# ====================================


def get_param_unit(hamiltonian, param_key):
    """
    Get unit for a given param_key
    """
    if param_key == "residual_field":
        return "Error: sum( || residual(params) || ) over bnvs/freqs (a.u.)"
    param_dict = get_param_odict(hamiltonian)
    return param_dict[param_key]


# ============================================================================


class ApproxBxyz(Hamiltonian):
    r"""
    Diagonal Hamiltonian approximation. To calculate the magnetic field only fields aligned with
    the NV are considered and thus a simple dot product can be used.

    Fits to bnvs rather than frequencies, i.e.:
    $$ \overline{\overline{B}}_{\rm NV} = overline{\overline{u}}_{\rm NV} \cdot \overline{B} $$
    Where overline denotes qst-order tensor (vector), double overline denotes 2nd-order tensor
    (matrix).
    """

    param_defn = ["Bx", "By", "Bz"]
    param_units = {
        "Bx": "Magnetic field, Bx (G)",
        "By": "Magnetic field, By (G)",
        "Bz": "Magnetic field, Bz (G)",
    }
    jac_defined = True

    def __call__(self, param_ar):
        r"""
        $$ \overline{\overline{B}}_{\rm NV} = overline{\overline{u}}_{\rm NV} \cdot \overline{B} $$
        Where overline denotes qst-order tensor (vector), double overline denotes 2nd-order tensor
        (matrix).
        Fit to bnv rather than frequency positions.

        param_ar = [Bx, By, Bz]
        """
        return np.dot(self.unvs, param_ar)

    def grad_fn(self, param_ar):
        return self.unvs
        # J = np.empty((4, 3))  # size: (len(bnvs), len(param_ar)), both known ahead of time.
        # J[:, 0] = self.unvs[:, 0]
        # J[:, 1] = self.unvs[:, 1]
        # J[:, 2] = self.unvs[:, 2]
        # return J
        # i.e. equiv. to:
        # for bnv in range(4):
        #     J[bnv, 0] = self.unvs[bnv][0]  # i.e. ith NV orientation (bnv), x component
        #     J[bnv, 1] = self.unvs[bnv][1]  # y component
        #     J[bnv, 2] = self.unvs[bnv][2]  # z component
        # should be rather obvious from __call__


# ============================================================================


class Bxyz(Hamiltonian):
    r"""
    $$ H_i = D S_{Z_i}^{2} + \gamma_{\rm{NV}} \bf{B} \cdot \bf{S} $$

    where \( {\bf S}_i = (S_{X_i}, S_{Y_i}, S_{Z_i}) \) are the spin-1 operators.
    Here \( (X_i, Y_i, Z_i) \) is the coordinate system of the NV and \( i = 1,2,3,4 \) labels
    each NV orientation with respect to the lab frame.
    """

    param_defn = ["D", "Bx", "By", "Bz"]
    param_units = {
        "D": "Zero field splitting (MHz)",
        "Bx": "Magnetic field, Bx (G)",
        "By": "Magnetic field, By (G)",
        "Bz": "Magnetic field, Bz (G)",
    }
    jac_defined = False

    def __call__(self, param_ar):
        """
        Hamiltonain of the NV spin using only the zero field splitting D and the magnetic field
        bxyz. Takes the fit_params in the order [D, bx, by, bz] and returns the nv frequencies.

        The spin operators need to be rotated to the NV reference frame. This is achieved
        by projecting the magnetic field onto the unv frame.

        i.e. we use the spin operatores in the NV frame, the unvs in the NV frame,
        and thus project the magnetic field into this frame (from the lab frame) to determine
        the nv frequencies (eigenvalues of hamiltonian).
        """
        from qdmpy.constants import S_MAT_X, S_MAT_Y, S_MAT_Z, GAMMA

        nv_frequencies = np.zeros(8)
        # D, Bx, By, Bz = param_ar

        Hzero = param_ar[0] * (S_MAT_Z * S_MAT_Z)
        for i in range(4):
            bx_proj_onto_unv = np.dot(param_ar[1:4], self.unv_frames[i, 0, :])
            by_proj_onto_unv = np.dot(param_ar[1:4], self.unv_frames[i, 1, :])
            bz_proj_onto_unv = np.dot(param_ar[1:4], self.unv_frames[i, 2, :])

            HB = GAMMA * (
                bx_proj_onto_unv * S_MAT_X
                + by_proj_onto_unv * S_MAT_Y
                + bz_proj_onto_unv * S_MAT_Z
            )
            freq, _ = LA.eig(Hzero + HB)
            freq = np.sort(np.real(freq))
            # freqs: ms=0, ms=+-1 -> transition freq is delta
            nv_frequencies[i] = np.real(freq[1] - freq[0])
            nv_frequencies[7 - i] = np.real(freq[2] - freq[0])
        return nv_frequencies

    # this method didn't work :(
    # def grad_fn(self, param_ar):

    # method here: do partial before calculating eigenvalues

    # from qdmpy.constants import S_MAT_X, S_MAT_Y, S_MAT_Z, GAMMA

    # J = np.empty((8, 4))  # shape: num freqs, num params

    # # p equiv. to D, Bx, By, Bz
    # for p in range(4):
    #     if not p:  # sort out D
    #         dH = S_MAT_Z * S_MAT_Z
    #         for i in range(4):
    #             df, _ = LA.eig(dH)
    #             df = np.sort(np.real(df))
    #             J[i, p] = np.real(df[1] - df[0])
    #             J[7 - i, p] = np.real(df[2] - df[0])
    #     else:  # sort out \vec{B}
    #         dparams = np.zeros(4)
    #         dparams[p] = 1  # no polynomials or anything like that so B_comp -> 1, others -> 0
    #         for i in range(4):

    #             dbx_proj_onto_unv = np.dot(dparams[1:4], self.unv_frames[i, 0, :])
    #             dby_proj_onto_unv = np.dot(dparams[1:4], self.unv_frames[i, 1, :])
    #             dbz_proj_onto_unv = np.dot(dparams[1:4], self.unv_frames[i, 2, :])

    #             dH = GAMMA * (
    #                 dbx_proj_onto_unv * S_MAT_X
    #                 + dby_proj_onto_unv * S_MAT_Y
    #                 + dbz_proj_onto_unv * S_MAT_Z
    #             )
    #             df, _ = LA.eig(dH)
    #             df = np.sort(np.real(df))
    #             # freqs: ms=0, ms=+-1 -> transition freq is delta
    #             J[i, p] = np.real(df[1] - df[0])
    #             J[7 - i, p] = np.real(df[2] - df[0])

    # return J

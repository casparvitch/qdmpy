# -*- coding: utf-8 -*-
"""
The module holds Hamiltonian objects that will be fit to.

Classes
-------
 - `QDMPy.hamiltonian._hamiltonians.Hamiltonian`
 - `QDMPy.hamiltonian._hamiltonians.ApproxBxyz`
 - `QDMPy.hamiltonian._hamiltonians.Bxyz`

Functions
---------
 - `QDMPy.hamiltonian._hamiltonians.get_param_defn`
 - `QDMPy.hamiltonian._hamiltonians.get_param_odict`
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.hamiltonian._hamiltonians.Hamiltonian": True,
    "QDMPy.hamiltonian._hamiltonians.ApproxBxyz": True,
    "QDMPy.hamiltonian._hamiltonians.Bxyz": True,
    "QDMPy.hamiltonian._hamiltonians.get_param_defn": True,
    "QDMPy.hamiltonian._hamiltonians.get_param_odict": True,
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

    def __init__(self, indices_fn, unv_frames):
        """
        indices_fn operates on __call__ and measured_data to return an array
        of only the required parts.
        """
        self.indices_fn = indices_fn
        self.unv_frames = unv_frames

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

    def grad_fn(self, param_ar, measured_data):
        """
        Return jacobian, shaoe: (len(bnvs/freqs), len(param_ar))
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
        return self.indices_fn(self.__call__(param_ar)) - self.indices_fn(measured_data)

    # =================================

    def jacobian_scipyfit(self, param_ar, measured_data):
        """Evaluates (analytic) jacobian of ham in format expected by scipy least_squares."""

        # need to take out rows (first index) according to indices_fn.
        keep_rows = self.indices_fn([i for i in range(len(measured_data))])
        delete_rows = [r for r in range(len(measured_data)) if r not in keep_rows]
        return np.delete(self.grad_fn(param_ar, measured_data), delete_rows, axis=0)

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
    if param_key == "residual_ham":
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
        return np.dot(self.unv_frames[:, 2, :], param_ar)

    def grad_fn(self, param_ar, measured_data):
        J = np.array((4, 3))  # size: (len(bnvs), len(param_ar)), both know ahead of time.
        J[:, 0] = 0
        J[:, 1] = 0
        J[:, 2] = 0
        return


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
        """
        from QDMPy.constants import S_MAT_X, S_MAT_Y, S_MAT_Z, GAMMA

        nv_frequencies = np.zeros(8)
        D, Bx, By, Bz = param_ar

        Hzero = D * (S_MAT_Z * S_MAT_Z)
        for i in range(4):
            bx_proj_onto_unv = np.dot([Bx, By, Bz], self.unv_frames[i, 0, :])
            by_proj_onto_unv = np.dot([Bx, By, Bz], self.unv_frames[i, 1, :])
            bz_proj_onto_unv = np.dot([Bx, By, Bz], self.unv_frames[i, 2, :])

            HB = GAMMA * (
                bx_proj_onto_unv * S_MAT_X
                + by_proj_onto_unv * S_MAT_Y
                + bz_proj_onto_unv * S_MAT_Z
            )
            freq, length = LA.eig(Hzero + HB)
            freq = np.sort(np.real(freq))
            nv_frequencies[i] = np.real(freq[1] - freq[0])
            nv_frequencies[7 - i] = np.real(freq[2] - freq[0])
        return nv_frequencies

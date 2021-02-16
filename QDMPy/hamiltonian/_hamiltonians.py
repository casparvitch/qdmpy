# -*- coding: utf-8 -*-
"""

Classes
-------
 - `QDMPy.hamiltonian._hamiltonians.Hamiltonian`
 - `QDMPy.hamiltonian._hamiltonians.ApproxBxyz`
 - `QDMPy.hamiltonian._hamiltonians.Bxyz`

Functions
---------
 - `QDMPy.hamiltonian._hamiltonians.get_param_defn`
 - `QDMPy.hamiltonian._hamiltonians.get_param_units`
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.hamiltonian._hamiltonians.Hamiltonian": True,
    "QDMPy.hamiltonian._hamiltonians.ApproxBxyz": True,
    "QDMPy.hamiltonian._hamiltonians.Bxyz": True,
    "QDMPy.hamiltonian._hamiltonians.get_param_defn": True,
    "QDMPy.hamiltonian._hamiltonians.get_param_units": True,
}

# ============================================================================

import numpy as np
import numpy.linalg as LA
from math import radians

# ============================================================================


# ============================================================================


class Hamiltonian:

    param_defn = []
    param_units = {}

    # TODO also ask for number of frequences?
    def __init__(self, diamond_ori, Bmag, Btheta, Bphi, unvs=None):
        self.unvs = np.array(unvs)
        if self.unvs.shape != (4, 3):
            raise ValueError("Incorrect unvs format passed to Hamiltonian. Expected shape: (4,3).")
        self.diamond_ori = diamond_ori
        self.Bmag = Bmag
        self.Btheta = Btheta
        self.Bphi = Bphi
        self.Btheta_rad = radians(Btheta)
        self.Bphi_rad = radians(Bphi)

        # calculate nv signd ori etc. here!

    # =================================

    def __call__(self, *param_ar):
        """
        Evaluates Hamiltonian for given parameter values.

        Arguments
        ---------
        param_ar : np array, 1D
            Array of parameters fed in.

        """
        raise NotImplementedError("You MUST override __call__, check your spelling.")

    # =================================

    def residuals_scipyfit(self, param_ar, measured_data):
        """
        Evaluates residual: fit model - measured_data. Returns a vector!
        Measured data must be a np array (of the same shape that __call__ returns),
        i.e. freqs, or bnvs.
        """
        return self.__call__(param_ar) - measured_data

    # =================================

    def jacobian_scipyfit(self, param_ar, **kwargs):
        raise NotImplementedError(
            "Analytic jacobians not currently accepted for Hamiltonian fitting."
        )

    # =================================
    def _calc_nv_ori(self):
        # Declare here to help with sizing etc.
        self.nv_ori = np.zeros((4, 3))
        self.nv_signs = np.zeros(4)
        self.nv_signed_ori = np.zeros(
            (4, 3)
        )  # these are the 'z' unit vectors of the unv frame, in the lab frame

        # get the cartesian magnetic fields
        bx = self.Bmag * np.sin(self.Btheta_rad) * np.cos(self.Bphi_rad)
        by = self.Bmag * np.sin(self.Btheta_rad) * np.sin(self.Bphi_rad)
        bz = self.Bmag * np.cos(self.Btheta_rad)
        # uses these values for the initial guesses (later)
        self.b_guess = {}
        self.b_guess["bx"] = bx
        self.b_guess["by"] = by
        self.b_guess["bz"] = bz
        # Get the NV orientations B magnitude and sign (from the B guess)

        from QDMPy.constants import NV_AXES_100_110, NV_AXES_100_100 # avoid cyclic dependencies

        if self.diamond_ori == "<100>_<100>":
            nv_axes = NV_AXES_100_100
        elif self.diamond_ori == "<100>_<110>":
            nv_axes = NV_AXES_100_110
        else:
            if self.unvs is not None:
                self.nv_signed_ori = self.unvs
            else:
                raise RuntimeError("diamond_ori not recognised and no unvs supplied.")
            return

        for key in range(len(nv_axes)):
            projection = np.dot(nv_axes[key]["ori"], [bx, by, bz])
            nv_axes[key]["mag"] = np.abs(projection)
            nv_axes[key]["sign"] = np.sign(projection)
        # Sort the dictionary in the correct order
        sorted_dict = sorted(nv_axes, key=lambda x: x["mag"], reverse=True)
        # define the nv orientation list for the fit
        for idx in range(len(sorted_dict)):
            self.nv_ori[idx, :] = sorted_dict[idx]["ori"]
            self.nv_signs[idx] = sorted_dict[idx]["sign"]
            self.nv_signed_ori[idx, :] = (
                np.array(sorted_dict[idx]["ori"]) * sorted_dict[idx]["sign"]
            )

        # # Calculate the inverse of the nv orientation matrix - unused???
        # self.nv_signed_ori_inv = self.nv_signed_ori.copy()
        # self.nv_signed_ori_inv[self.nv_signed_ori_inv == 0] = np.inf
        # self.nv_signed_ori_inv = 1 / self.nv_signed_ori_inv


# ============================================================================


def get_param_defn(hamiltonian):
    return hamiltonian.param_defn


# ============================================================================

# TODO add get_param_unit etc.
def get_param_units(hamiltonian):
    return hamiltonian.param_units


# ============================================================================


# Note just invert matrix if three bnvs? How to encode that... elsewhere...
# (i.e. instead of fitting:
#                           \vec{Bnv} = \tensor{unv} \cdot \vec{B},
#                        do:
#                           \vec{unv}.T \cdot \vec{Bnv} = \vec{B} )
class ApproxBxyz(Hamiltonian):
    r"""
    Diagonal Hamiltonian approximation. To calculate the magnetic field only fields aligned with
    the NV are considered and thus a simple dot product can be used.

    Fits to bnvs rather than frequencies, i.e.:
    $$ \vec{B}_{\rm NV} = \vec{u}_{\rm NV} \cdot \vec{B} $$
    """

    param_defn = ["Bx", "By", "Bz"]
    param_units = {
        "Bx": "Magnetic field, Bx (G)",
        "By": "Magnetic field, By (G)",
        "Bz": "Magnetic field, Bz (G)",
    }

    def __call__(self, Bx, By, Bz):
        r"""
        $$ \vec{B}_{\rm NV} = \vec{u}_{\rm NV} \cdot \vec{B} $$
        Fit to bnv rather than frequency positions.
        """
        return np.dot(self.nv_signed_ori, [Bx, By, Bz])


# FIXME what to do if < 8 frequencies? Fix this here/somehow.
# I think it matters which NVs they are yeah? Or assume they're the outside nv fams
# otherwise the user can supply the unvs? -> need to actually think through the maths
#  of what this would entail etc.
#       ALSO: then for e.g. single nv ori, what do we do? pass in unv? Critical for <111>
#           -> 111 they can just chuck in [0, 0, 1] and be fine.
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # this calculates nv_signed_ori
        self.unv_frame = np.zeros((4, 3, 3))
        self.nv_frequencies = np.zeros(8)
        for i in range(4):
            # calculate uNV frame in the lab frame
            uNV_Z = self.nv_signed_ori[i]
            # We have full freedom to pick x/y as long as xyz are all orthogonal
            # we can ensure this by picking Y to be orthog. to both the NV axis
            # and another NV axis, then get X to be the cross between those two.
            uNV_Y = np.cross(uNV_Z, self.nv_signed_ori[-i - 1])
            uNV_Y = uNV_Y / LA.norm(uNV_Y)
            uNV_X = np.cross(uNV_Y, uNV_Z)
            self.unv_frame[i, ::] = [uNV_X, uNV_Y, uNV_Z]

    def __call__(self, D, Bx, By, Bz):
        """
        Hamiltonain of the NV spin using only the zero field splitting D and the magnetic field
        bxyz. Takes the fit_params in the order [D, bx, by, bz] and returns the nv frequencies.

        The spin operators need to be rotated to the NV reference frame. This is achieved
        by projecting the magnetic field onto the unv frame.
        """
        from QDMPy.constants import S_MAT_X, S_MAT_Y, S_MAT_Z, GAMMA

        Hzero = D * (S_MAT_Z * S_MAT_Z)
        for i in range(4):
            bx_proj_onto_unv = np.dot([Bx, By, Bz], self.unv_frame[i, 0, ::])
            by_proj_onto_unv = np.dot([Bx, By, Bz], self.unv_frame[i, 1, ::])
            bz_proj_onto_unv = np.dot([Bx, By, Bz], self.unv_frame[i, 2, ::])

            HB = GAMMA * (
                bx_proj_onto_unv * S_MAT_X
                + by_proj_onto_unv * S_MAT_Y
                + bz_proj_onto_unv * S_MAT_Z
            )
            freq, length = LA.eig(Hzero + HB)
            freq = np.sort(np.real(freq))
            self.nv_frequencies[i] = np.real(freq[1] - freq[0])
            self.nv_frequencies[7 - i] = np.real(freq[2] - freq[0])
        return self.nv_frequencies

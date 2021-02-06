# -*- coding: utf-8 -*-
"""

Classes
-------
 - ``

Functions
---------
 - `blaa`
"""

# ============================================================================

import numpy as np
import numpy.linalg as LA
import math

# ============================================================================

from QDMPy.constants import S_MAT_X, S_MAT_Y, S_MAT_Z, GAMMA, NV_AXES_100_110, NV_AXES_100_100

# ============================================================================


class Hamiltonian:
    """FitModel used to fit to data."""

    param_defn = []
    param_units = {}

    # TODO also ask for number of frequences?
    def __init__(self, diamond_ori, Bmag, Btheta, Bphi, unvs=None):
        self.unvs = unvs  # TODO check unvs shape/size?
        self.diamond_ori = diamond_ori
        self.Bmag = Bmag
        self.Btheta = Btheta
        self.Bphi = Bphi
        self.Btheta_rad = math.radians(self.theta)
        self.Bphi_rad = math.radians(self.phi)

        # calculate nv signd ori etc. here!

    # =================================

    def __call__(self, param_ar):
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
        raise NotImplementedError("Jacobians currently not implemented for Hamiltonians.")

    # =================================

    def get_param_defn(self):
        return self.param_defn

    # =================================

    def get_param_odict(self):
        return self.param_odict

    # =================================
    def calc_nv_ori(self):
        # TODO document these variables etc. Declare here to help with sizing etc.
        self.nv_ori = np.zeros((4, 3))
        self.nv_signs = np.zeros(4)
        self.nv_signed_ori = np.zeros((4, 3))

        # get the cartesian magnetic fields
        bx = self.Bmag * np.sin(self.Btheta_rad) * np.cos(self.Bphi_rad)
        by = self.Bmag * np.sin(self.Btheta_rad) * np.sin(self.Bphi_rad)
        bz = self.Bmag * np.cos(self.Btheta_rad)
        # uses these values for the initial guesses
        self.b_guess = {}
        self.b_guess["bx"] = bx
        self.b_guess["by"] = by
        self.b_guess["bz"] = bz
        # Get the NV orientations B magnitude and sign
        if self.diamond_ori == "<100>_<100>":
            self.nv_axes = NV_AXES_100_100
        else:
            self.nv_axes = NV_AXES_100_110
        for key in range(len(self.nv_axes)):
            projection = np.dot(self.nv_axes[key]["ori"], [bx, by, bz])
            self.nv_axes[key]["mag"] = np.abs(projection)
            self.nv_axes[key]["sign"] = np.sign(projection)
        # Sort the dictionary in the correct order
        sorted_dict = sorted(self.nv_axes, key=lambda x: x["mag"], reverse=True)
        # define the nv orientation list for the fit
        for idx in range(len(sorted_dict)):
            self.nv_ori[idx, :] = sorted_dict[idx]["ori"]
            self.nv_signs[idx] = sorted_dict[idx]["sign"]
            self.nv_signed_ori[idx, :] = (
                np.array(sorted_dict[idx]["ori"]) * sorted_dict[idx]["sign"]
            )
        if self.unvs is not None:
            self.nv_signed_ori = np.array(self.unvs)
        # Calculate the inverse of the nv orientation matrix
        self.nv_signed_ori_inv = self.nv_signed_ori.copy()
        self.nv_signed_ori_inv[self.nv_signed_ori_inv == 0] = np.inf
        self.nv_signed_ori_inv = 1 / self.nv_signed_ori_inv


# ============================================================================


def get_param_defn(hamiltonian):
    return hamiltonian.param_defn


def get_param_odict(hamiltonian):
    return hamiltonian.param_units


# ============================================================================


# Note just invert matrix if three bnvs? How to encode that... elsewhere...
# (i.e. instead of fitting:
#                           \vec{Bnv} = \tensor{unv} \cdot \vec{B},
#                        do:
#                           \vec{unv}.T \cdot \vec{Bnv} = \vec{B} )
class ApproxBxyz(Hamiltonian):
    """
    Diagonal Hamiltonian approximation. To calculate the magnetic field only fields aligned with
    the NV are considered and thus a simple dot product can be used.
    """

    param_defn = ["Bx", "By", "Bz"]
    param_units = {
        "Bx": "Magnetic field, Bx (G)",
        "By": "Magnetic field, By (G)",
        "Bz": "Magnetic field, Bz (G)",
    }

    def __call__(self, Bx, By, Bz):
        return np.dot([Bx, By, Bz], self.nv_signed_ori.T)


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
        super().__init__(*args, **kwargs)
        self.rotations = np.zeros((4, 3, 3))
        self.nv_frequencies = np.zeros(8)
        for i in range(4):
            zrot = self.nv_signed_ori[i].T
            yrot = np.cross(zrot, self.nv_signed_ori[-i - 1].T)
            yrot = yrot / LA.norm(yrot)
            xrot = np.cross(yrot, zrot)
            self.rotations[i, ::] = [xrot, yrot, zrot]

    def __call__(self, D, Bx, By, Bz):
        """Hamiltonain of the NV spin using only the zero field splitting D and the magnetic field
        bxyz. Takes the fit_params in the order [D, bx, by, bz] and returns the nv frequencies.
        """
        Hzero = D * (S_MAT_Z * S_MAT_Z)
        for i in range(4):
            bx = np.dot([Bx, By, Bz], self.rotations[i, 0, ::])
            by = np.dot([Bx, By, Bz], self.rotations[i, 1, ::])
            bz = np.dot([Bx, By, Bz], self.rotations[i, 2, ::])

            HB = GAMMA * (bx * S_MAT_X + by * S_MAT_Y + bz * S_MAT_Z)
            freq, length = LA.eig(Hzero + HB)
            freq = np.sort(np.real(freq))
            self.nv_frequencies[i] = np.real(freq[1] - freq[0])
            self.nv_frequencies[7 - i] = np.real(freq[2] - freq[0])
        return self.nv_frequencies

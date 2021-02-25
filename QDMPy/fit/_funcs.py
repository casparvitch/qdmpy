# -*- coding: utf-8 -*-
"""
This module defines the fit functions used in the `QDMPy.fit.model.FitModel`.
We grab/use this regardless of fitting on cpu (scipy) or gpu etc.

Ensure any fit functions you define are added to the AVAILABLE_FNS variable in `QDMPy.constants`.
Try not to have overlapping parameter names in the same fit.

For ODMR peaks, ensure the frequency position of the peak is named something
prefixed by 'pos'. (see `QDMPy.field._bnv.get_bnvs_and_dshifts` for the reasoning).

Classes
-------
 - `QDMPy.fit._funcs.FitFunc`
 - `QDMPy.fit._funcs.Constant`
 - `QDMPy.fit._funcs.Linear`
 - `QDMPy.fit._funcs.Circular`
 - `QDMPy.fit._funcs.Gaussian`
 - `QDMPy.fit._funcs.Gaussian_hyperfine_14`
 - `QDMPy.fit._funcs.Gaussian_hyperfine_15`
 - `QDMPy.fit._funcs.Lorentzian`
 - `QDMPy.fit._funcs.Lorentzian_hyperfine_14`
 - `QDMPy.fit._funcs.Lorentzian_hyperfine_15`
 - `QDMPy.fit._funcs.Stretched_exponential`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.fit._funcs.FitFunc": True,
    "QDMPy.fit._funcs.Constant": True,
    "QDMPy.fit._funcs.Linear": True,
    "QDMPy.fit._funcs.Circular": True,
    "QDMPy.fit._funcs.Gaussian": True,
    "QDMPy.fit._funcs.Gaussian_hyperfine_14": True,
    "QDMPy.fit._funcs.Gaussian_hyperfine_15": True,
    "QDMPy.fit._funcs.Lorentzian": True,
    "QDMPy.fit._funcs.Lorentzian_hyperfine_14": True,
    "QDMPy.fit._funcs.Lorentzian_hyperfine_15": True,
    "QDMPy.fit._funcs.Stretched_exponential": True,
}

# ============================================================================

import numpy as np
from numba import njit

# ================================================================================================
# ================================================================================================
#
# FitFunc Class
#
# ================================================================================================
# ================================================================================================


class FitFunc:
    """Singular fit function"""

    def __init__(self, param_indices):
        """
        Arguments
        ---------
        param_indices : np array
            Where the parameters for this fitfunc are located within broader fitmodel param array.
        """
        self.this_fn_param_indices = param_indices

    # =================================

    @staticmethod
    def eval(sweep_vec, *fit_params):
        raise NotImplementedError("You MUST override eval, check your spelling.")

    # =================================

    @staticmethod
    def grad_fn(sweep_vec, *fit_params):
        """ if you want to use a grad_fn override this in the subclass """
        return None


# ====================================================================================
# ====================================================================================
#
# Fit Functions themselves
#
# ====================================================================================
# ====================================================================================


# ====================================================================================
# Constant
# ====================================================================================


class Constant(FitFunc):
    """Constant"""

    param_defn = ["constant"]
    param_units = {"constant": "Amplitude (a.u.)"}

    # =================================

    @staticmethod
    @njit(fastmath=True)
    def eval(x, c):
        """ speed tested multiple methods, this was the fastest """
        ret = np.empty(np.shape(x))
        ret.fill(c)
        return ret

    # =================================

    @staticmethod
    @njit
    def grad_fn(x, c):
        """Compute the grad of the residue, excluding PL as a param
        {output shape: (len(x), 1)}
        """
        J = np.empty((x.shape[0], 1))
        J[:, 0] = 1
        return J


# ====================================================================================
# Linear
# ====================================================================================


class Linear(FitFunc):
    """Linear function, y=mx+c"""

    param_defn = ["c", "m"]
    param_units = {"c": "Amplitude (a.u.)", "m": "Amplitude per Freq (a.u.)"}

    #    def __init__(self, num_peaks):
    #        super().__init__(num_peaks)

    # =================================

    # speed tested, marginally faster with fastmath off (idk why)
    @staticmethod
    @njit(fastmath=False)
    def eval(x, c, m):
        return m * x + c

    # =================================

    @staticmethod
    @njit(fastmath=True)
    def grad_fn(x, c, m):
        """Compute the grad of the residue, excluding PL as a param
        {output shape: (len(x), 2)}
        """
        J = np.empty((x.shape[0], 2))
        J[:, 0] = 1
        J[:, 1] = x
        return J


# ====================================================================================
# Circular
# ====================================================================================


class Circular(FitFunc):
    """
    Circular function (sine)
    """

    param_defn = ["rabi_freq", "t0_circ", "amp_circ"]
    param_units = {"rabi_freq": "Nu (Hz)", "t0_circ": "Tau (s)", "amp_circ": "Amp (a.u.)"}

    @staticmethod
    @njit
    def eval(x, rabi_freq, pos, amp):
        return amp * np.sin(2 * np.pi * rabi_freq * (x - pos))

    @staticmethod
    @njit
    def grad_fn(x, rabi_freq, pos, amp):
        """Compute the grad of the residue, excluding PL as a param
        {output shape: (len(x), 3)}
        """
        # Lorentzian: a*g^2/ ((x-c)^2 + g^2)
        J = np.empty((x.shape[0], 3), dtype=np.float32)
        J[:, 0] = 2 * np.pi * amp * (x - pos) * np.cos(2 * np.pi * rabi_freq * (x - pos))
        J[:, 1] = -2 * np.pi * rabi_freq * amp * np.cos(2 * np.pi * rabi_freq * (x - pos))
        J[:, 2] = np.sin(2 * np.pi * rabi_freq * (x - pos))
        return J


# ====================================================================================
# Gaussians
# ====================================================================================
SCALE_SIGMA = 4 * np.log(2)


class Gaussian(FitFunc):
    """Gaussian function"""

    param_defn = ["fwhm_gauss", "pos_gauss", "amp_gauss"]
    param_units = {
        "fwhm_gauss": "Freq (MHz)",
        "pos_gauss": "Freq (MHz)",
        "amp_gauss": "Amp (a.u.)",
    }

    # =================================

    @staticmethod
    @njit(fastmath=True)
    def eval(x, fwhm, pos, amp):
        return amp * np.exp(-SCALE_SIGMA * (x - pos) ** 2 / fwhm ** 2)


class Gaussian_hyperfine_14(FitFunc):

    param_defn = [
        "pos_gauss_h14",
        "amp_gauss_h14_hyp_1",
        "amp_gauss_h14_hyp_2",
        "amp_gauss_h14_hyp_3",
        "fwhm_gauss_h14_hyp_1",
        "fwhm_gauss_h14_hyp_2",
        "fwhm_gauss_h14_hyp_3",
    ]
    param_units = {
        "pos_gauss_h14": "Frequency (MHz)",
        "amp_gauss_h14_hyp_1": "Amplitude (a.u.)",
        "amp_gauss_h14_hyp_2": "Amplitude (a.u.)",
        "amp_gauss_h14_hyp_3": "Amplitude (a.u.)",
        "fwhm_gauss_h14_hyp_1": "Frequency (MHz)",
        "fwhm_gauss_h14_hyp_2": "Frequency (MHz)",
        "fwhm_gauss_h14_hyp_23": "Frequency (MHz)",
    }

    # A14 para = -2.14 MHz
    @staticmethod
    @njit(fastmath=True)
    def eval(x, pos, amp_1_hyp, amp_2_hyp, amp_3_hyp, fwhm_1_hyp, fwhm_2_hyp, fwhm_3_hyp):
        return (
            amp_1_hyp * np.exp(-SCALE_SIGMA * (x - pos - 2.14) ** 2 / fwhm_1_hyp ** 2)
            + amp_2_hyp * np.exp(-SCALE_SIGMA * (x - pos) ** 2 / fwhm_2_hyp ** 2)
            + amp_3_hyp * np.exp(-SCALE_SIGMA * (x - pos + 2.14) ** 2 / fwhm_3_hyp ** 2)
        )


class Gaussian_hyperfine_15(FitFunc):

    param_defn = [
        "pos_gauss_h15",
        "amp_gauss_h15_hyp_1",
        "amp_gauss_h15_hyp_2",
        "fwhm_gauss_h15_hyp_1",
        "fwhm_gauss_h15_hyp_2",
    ]
    param_units = {
        "pos_gauss_h15": "Frequency (MHz)",
        "amp_gauss_h15_hyp_1": "Amplitude (a.u.)",
        "amp_gauss_h15_hyp_2": "Amplitude (a.u.)",
        "fwhm_gauss_h15_hyp_1": "Frequency (MHz)",
        "fwhm_gauss_h15_hyp_2": "Frequency (MHz)",
    }

    # A15 para = 3.03 MHz
    @staticmethod
    @njit(fastmath=True)
    def eval(x, pos, amp_1_hyp, amp_2_hyp, fwhm_1_hyp, fwhm_2_hyp):
        return amp_1_hyp * np.exp(
            -SCALE_SIGMA * (x - pos - 1.515) ** 2 / fwhm_1_hyp ** 2
        ) + amp_2_hyp * np.exp(-SCALE_SIGMA * (x - pos + 1.515) ** 2 / fwhm_2_hyp ** 2)


# ====================================================================================
# Lorentzians
# ====================================================================================


class Lorentzian(FitFunc):
    """Lorentzian function"""

    param_defn = ["fwhm", "pos", "amp"]
    param_units = {"fwhm": "Freq (MHz)", "pos": "Freq (MHz)", "amp": "Amp (a.u.)"}

    # =================================

    # fastmath gives a ~10% speed up on my testing
    @staticmethod
    @njit(fastmath=True)
    def eval(x, fwhm, pos, amp):
        hwhmsqr = (fwhm ** 2) / 4
        return amp * hwhmsqr / ((x - pos) ** 2 + hwhmsqr)

    # =================================

    @staticmethod
    @njit
    def grad_fn(x, fwhm, pos, amp):
        """Compute the grad of the residue, excluding PL as a param
        {output shape: (len(x), 3)}
        """
        # Lorentzian: a*g^2/ ((x-c)^2 + g^2)
        J = np.empty((x.shape[0], 3), dtype=np.float32)
        g = fwhm / 2
        c = pos
        a = amp
        J[:, 0] = ((2 * a * g) / (g ** 2 + (x - c) ** 2)) - (
            (2 * a * g ** 3) / (g ** 2 + (x - c) ** 2) ** 2
        )
        J[:, 1] = (2 * a * g ** 2 * (x - c)) / (g ** 2 + (x - c) ** 2) ** 2
        J[:, 2] = g ** 2 / ((x - c) ** 2 + g ** 2)
        return J


class Lorentzian_hyperfine_14(FitFunc):

    param_defn = [
        "pos_h14",
        "amp_h14_hyp_1",
        "amp_h14_hyp_2",
        "amp_h14_hyp_3",
        "fwhm_h14_hyp_1",
        "fwhm_h14_hyp_2",
        "fwhm_h14_hyp_3",
    ]
    param_units = {
        "pos_h14": "Frequency (MHz)",
        "amp_h14_hyp_1": "Amplitude (a.u.)",
        "amp_h14_hyp_2": "Amplitude (a.u.)",
        "amp_h14_hyp_3": "Amplitude (a.u.)",
        "fwhm_h14_hyp_1": "Frequency (MHz)",
        "fwhm_h14_hyp_2": "Frequency (MHz)",
        "fwhm_h14_hyp_23": "Frequency (MHz)",
    }

    # A14 para = -2.14 MHz
    @staticmethod
    @njit(fastmath=True)
    def eval(x, pos, amp_1_hyp, amp_2_hyp, amp_3_hyp, fwhm_1_hyp, fwhm_2_hyp, fwhm_3_hyp):
        hwhmsqr1 = (fwhm_1_hyp ** 2) / 4
        hwhmsqr2 = (fwhm_2_hyp ** 2) / 4
        hwhmsqr3 = (fwhm_3_hyp ** 2) / 4
        return (
            amp_1_hyp * hwhmsqr1 / ((x - pos - 2.14) ** 2 + hwhmsqr1)
            + amp_2_hyp * hwhmsqr2 / ((x - pos) ** 2 + hwhmsqr2)
            + amp_3_hyp * hwhmsqr3 / ((x - pos + 2.14) ** 2 + hwhmsqr3)
        )


class Lorentzian_hyperfine_15(FitFunc):

    param_defn = ["pos_h15", "amp_h15_hyp_1", "amp_h15_hyp_2", "fwhm_h15_hyp_1", "fwhm_h15_hyp_2"]
    param_units = {
        "pos_h15": "Frequency (MHz)",
        "amp_h15_hyp_1": "Amplitude (a.u.)",
        "amp_h15_hyp_2": "Amplitude (a.u.)",
        "fwhm_h15_hyp_1": "Frequency (MHz)",
        "fwhm_h15_hyp_2": "Frequency (MHz)",
    }

    # A15 para = 3.03 MHz
    @staticmethod
    @njit(fastmath=True)
    def eval(x, pos, amp_1_hyp, amp_2_hyp, fwhm_1_hyp, fwhm_2_hyp):
        hwhmsqr1 = fwhm_1_hyp ** 2 / 4
        hwhmsqr2 = fwhm_2_hyp ** 2 / 4
        return amp_1_hyp * hwhmsqr1 / (
            (x - pos - 1.515) ** 2 + hwhmsqr1
        ) + amp_2_hyp * hwhmsqr2 / ((x - pos + 1.515) ** 2 + hwhmsqr2)


# ==========================================================================
# Exponential fit functions
# ==========================================================================


class Stretched_exponential(FitFunc):

    param_defn = ["charac_exp_t", "amp_exp", "power_exp"]
    param_units = {
        "charac_exp_t": "Time (s)",
        "amp_exp": "Amplitude (a.u.)",
        "power_exp": "Unitless",
    }

    # =================================

    @staticmethod
    # njit here not speed tested
    @njit
    def eval(x, charac_exp_t, amp_exp, power_exp):
        return amp_exp * np.exp(-((x / charac_exp_t) ** power_exp))

    # =================================

    @staticmethod
    @njit
    def grad_fn(x, charac_exp_t, amp_exp, power_exp):
        """Compute the grad of the residue, excluding PL as a param
        {output shape: (len(x), 3)}
        """

        J = np.empty((x.shape[0], 3), dtype=np.float32)
        # stretched exponential = a * e ^ ((x / t) ^ p)
        # -(a p e^((x/t)^p) (x/t)^p)/t
        J[:, 0] = (1 / charac_exp_t) * (
            amp_exp
            * power_exp
            * np.exp(-((x / charac_exp_t) ** power_exp))
            * (x / charac_exp_t) ** power_exp
        )
        # just lose the 'a'
        J[:, 1] = np.exp(-((x / charac_exp_t) ** power_exp))
        # a e^((x/t)^p) (x/t)^p log(x/t)
        J[:, 2] = (
            -amp_exp
            * np.exp(-((x / charac_exp_t) ** power_exp))
            * (x / charac_exp_t) ** power_exp
            * np.log(x / charac_exp_t)
        )
        return J


# ==========================================================================

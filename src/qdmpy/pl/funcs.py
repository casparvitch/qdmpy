# -*- coding: utf-8 -*-
"""
This module defines the fit functions used in the `qdmpy.pl.model.FitModel`.
We grab/use this regardless of fitting on cpu (scipy) or gpu etc.

Ensure any fit functions you define are added to the AVAILABLE_FNS module variable at bottom.
Try not to have overlapping parameter names in the same fit.

For ODMR peaks, ensure the frequency position of the peak is named something
prefixed by 'pos'. (see `qdmpy.field.bnv.get_bnvs_and_dshifts` for the reasoning).

Classes
-------
 - `qdmpy.pl.funcs.FitFunc`
 - `qdmpy.pl.funcs.Constant`
 - `qdmpy.pl.funcs.Linear`
 - `qdmpy.pl.funcs.Circular`
 - `qdmpy.pl.funcs.Gaussian`
 - `qdmpy.pl.funcs.GaussianHyperfine14`
 - `qdmpy.pl.funcs.GaussianHyperfine15`
 - `qdmpy.pl.funcs.Lorentzian`
 - `qdmpy.pl.funcs.LorentzianHyperfine14`
 - `qdmpy.pl.funcs.LorentzianHyperfine15`
 - `qdmpy.pl.funcs.StretchedExponential`
 - `qdmpy.pl.funcs.DampedRabi`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.pl.funcs.FitFunc": True,
    "qdmpy.pl.funcs.Constant": True,
    "qdmpy.pl.funcs.Linear": True,
    "qdmpy.pl.funcs.Circular": True,
    "qdmpy.pl.funcs.Gaussian": True,
    "qdmpy.pl.funcs.GaussianHyperfine14": True,
    "qdmpy.pl.funcs.GaussianHyperfine15": True,
    "qdmpy.pl.funcs.Lorentzian": True,
    "qdmpy.pl.funcs.LorentzianHyperfine_14": True,
    "qdmpy.pl.funcs.LorentzianHyperfine_15": True,
    "qdmpy.pl.funcs.StretchedExponential": True,
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
    def eval(x, *fit_params):
        raise NotImplementedError("You MUST override eval, check your spelling.")

    # =================================

    @staticmethod
    def grad_fn(x, *fit_params):
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
    def eval(x, *fit_params):
        """ speed tested multiple methods, this was the fastest """
        c = fit_params[0]
        ret = np.empty(np.shape(x))
        ret.fill(c)
        return ret

    # =================================

    @staticmethod
    @njit
    def grad_fn(x, *fit_params):
        """Compute the grad of the residue, excluding pl as a param
        {output shape: (len(x), 1)}
        """
        j = np.empty((x.shape[0], 1))
        j[:, 0] = 1
        return j


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
    def eval(x, *fit_params):
        c, m = fit_params
        return m * x + c

    # =================================

    @staticmethod
    @njit(fastmath=True)
    def grad_fn(x, *fit_params):
        """
        Compute the grad of the residual, excluding pl as a param
        {output shape: (len(x), 2)}
        """
        c, m = fit_params
        j = np.empty((x.shape[0], 2))
        j[:, 0] = 1
        j[:, 1] = x
        return j


# ====================================================================================
# Circular
# ====================================================================================


class Circular(FitFunc):
    """
    Circular function (sine)
    """

    param_defn = ["circ_freq", "t0_circ", "amp_circ"]
    param_units = {"circ_freq": "Nu (Hz)", "t0_circ": "Tau (s)", "amp_circ": "Amp (a.u.)"}

    @staticmethod
    @njit
    def eval(x, *fit_params):
        circ_freq, pos, amp = fit_params
        return amp * np.sin(2 * np.pi * circ_freq * (x - pos))

    @staticmethod
    @njit
    def grad_fn(x, *fit_params):
        """Compute the grad of the residue, excluding pl as a param
        {output shape: (len(x), 3)}
        """
        # Lorentzian: a*g^2/ ((x-c)^2 + g^2)
        circ_freq, pos, amp = fit_params
        j = np.empty((x.shape[0], 3), dtype=np.float32)
        j[:, 0] = 2 * np.pi * amp * (x - pos) * np.cos(2 * np.pi * circ_freq * (x - pos))
        j[:, 1] = -2 * np.pi * circ_freq * amp * np.cos(2 * np.pi * circ_freq * (x - pos))
        j[:, 2] = np.sin(2 * np.pi * circ_freq * (x - pos))
        return j


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
    def eval(x, *fit_params):
        fwhm, pos, amp = fit_params
        return amp * np.exp(-SCALE_SIGMA * (x - pos) ** 2 / fwhm ** 2)


class GaussianHyperfine14(FitFunc):

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
    def eval(x, *fit_params):
        pos, amp_1_hyp, amp_2_hyp, amp_3_hyp, fwhm_1_hyp, fwhm_2_hyp, fwhm_3_hyp = fit_params
        return (
            amp_1_hyp * np.exp(-SCALE_SIGMA * (x - pos - 2.14) ** 2 / fwhm_1_hyp ** 2)
            + amp_2_hyp * np.exp(-SCALE_SIGMA * (x - pos) ** 2 / fwhm_2_hyp ** 2)
            + amp_3_hyp * np.exp(-SCALE_SIGMA * (x - pos + 2.14) ** 2 / fwhm_3_hyp ** 2)
        )


class GaussianHyperfine15(FitFunc):

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
    def eval(x, *fit_params):
        pos, amp_1_hyp, amp_2_hyp, fwhm_1_hyp, fwhm_2_hyp = fit_params
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
    def eval(x, *fit_params):
        fwhm, pos, amp = fit_params
        hwhmsqr = (fwhm ** 2) / 4
        return amp * hwhmsqr / ((x - pos) ** 2 + hwhmsqr)

    # =================================

    @staticmethod
    @njit
    def grad_fn(x, *fit_params):
        """Compute the grad of the residue, excluding pl as a param
        {output shape: (len(x), 3)}
        """
        # Lorentzian: a*g^2/ ((x-c)^2 + g^2)
        fwhm, pos, amp = fit_params
        j = np.empty((x.shape[0], 3), dtype=np.float32)
        g = fwhm / 2
        c = pos
        a = amp
        j[:, 0] = (a * g * (x - c) ** 2) / ((x - c) ** 2 + g ** 2) ** 2
        j[:, 1] = (2 * a * g ** 2 * (x - c)) / (g ** 2 + (x - c) ** 2) ** 2
        j[:, 2] = g ** 2 / ((x - c) ** 2 + g ** 2)
        return j


class LorentzianHyperfine14(FitFunc):

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
    def eval(x, *fit_params):
        pos, amp_1_hyp, amp_2_hyp, amp_3_hyp, fwhm_1_hyp, fwhm_2_hyp, fwhm_3_hyp = fit_params
        hwhmsqr1 = (fwhm_1_hyp ** 2) / 4
        hwhmsqr2 = (fwhm_2_hyp ** 2) / 4
        hwhmsqr3 = (fwhm_3_hyp ** 2) / 4
        return (
            amp_1_hyp * hwhmsqr1 / ((x - pos - 2.14) ** 2 + hwhmsqr1)
            + amp_2_hyp * hwhmsqr2 / ((x - pos) ** 2 + hwhmsqr2)
            + amp_3_hyp * hwhmsqr3 / ((x - pos + 2.14) ** 2 + hwhmsqr3)
        )


class LorentzianHyperfine15(FitFunc):

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
    def eval(x, *fit_params):
        pos, amp_1_hyp, amp_2_hyp, fwhm_1_hyp, fwhm_2_hyp = fit_params
        hwhmsqr1 = fwhm_1_hyp ** 2 / 4
        hwhmsqr2 = fwhm_2_hyp ** 2 / 4
        return amp_1_hyp * hwhmsqr1 / (
            (x - pos - 1.515) ** 2 + hwhmsqr1
        ) + amp_2_hyp * hwhmsqr2 / ((x - pos + 1.515) ** 2 + hwhmsqr2)


# ==========================================================================
# Exponential fit functions
# ==========================================================================


class StretchedExponential(FitFunc):

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
    def eval(x, *fit_params):
        charac_exp_t, amp_exp, power_exp = fit_params
        return amp_exp * np.exp(-((x / charac_exp_t) ** power_exp))

    # =================================

    @staticmethod
    @njit
    def grad_fn(x, *fit_params):
        """Compute the grad of the residue, excluding pl as a param
        {output shape: (len(x), 3)}
        """
        charac_exp_t, amp_exp, power_exp = fit_params
        j = np.empty((x.shape[0], 3), dtype=np.float32)
        # stretched exponential = a * e ^ ((x / t) ^ p)
        # -(a p e^((x/t)^p) (x/t)^p)/t
        j[:, 0] = (1 / charac_exp_t) * (
            amp_exp
            * power_exp
            * np.exp(-((x / charac_exp_t) ** power_exp))
            * (x / charac_exp_t) ** power_exp
        )
        # just lose the 'a'
        j[:, 1] = np.exp(-((x / charac_exp_t) ** power_exp))
        # a e^((x/t)^p) (x/t)^p log(x/t)
        j[:, 2] = (
            -amp_exp
            * np.exp(-((x / charac_exp_t) ** power_exp))
            * (x / charac_exp_t) ** power_exp
            * np.log(x / charac_exp_t)
        )
        return j


# ==========================================================================


class DampedRabi(FitFunc):
    """
    Damped oscillation
    """

    param_defn = ["rabi_freq", "rabi_t_offset", "rabi_amp", "rabi_decay_time"]
    param_units = {
        "rabi_freq": "Omega (rad/s)",
        "rabi_t_offset": "Tau_0 (s)",
        "amp_circ": "Amp (a.u.)",
        "rabi_decay_time": "Tau_d (s)",
    }

    @staticmethod
    @njit
    def eval(x, *fit_params):
        omega, pos, amp, tau = fit_params
        return amp * np.exp(-(x / tau)) * np.cos(omega * (x - pos))

    @staticmethod
    @njit
    def grad_fn(x, *fit_params):
        """Compute the grad of the residue, excluding pl as a param
        {output shape: (len(x), 4)}
        """
        omega, pos, amp, tau = fit_params
        j = np.empty((x.shape[0], 4), dtype=np.float32)
        j[:, 0] = amp * (pos - x) * np.sin(omega * (x - pos)) * np.exp(-x / tau)  # wrt omega
        j[:, 1] = (amp * omega * np.sin(omega * (x - pos))) * np.exp(-x / tau)  # wrt pos
        j[:, 2] = np.exp(-x / tau) * np.cos(omega * (x - pos))  # wrt amp
        j[:, 3] = (amp * x * np.cos(omega * (x - pos))) / (np.exp(x / tau) * tau ^ 2)  # wrt tau
        return j


# ==========================================================================

AVAILABLE_FNS = {
    "lorentzian": Lorentzian,
    "lorentzian_hyperfine_14": LorentzianHyperfine14,
    "lorentzian_hyperfine_15": LorentzianHyperfine15,
    "gaussian": Gaussian,
    "gaussian_hyperfine_14": GaussianHyperfine14,
    "gaussian_hyperfine_15": GaussianHyperfine15,
    "constant": Constant,
    "linear": Linear,
    "circular": Circular,
    "stretched_exponential": StretchedExponential,
    "damped_rabi": DampedRabi,
}
"""Dictionary that defines fit functions available for use.

Add any functions you define here so you can use them.

Aviod overlapping function parameter names.
"""

# ==========================================================================

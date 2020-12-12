# -*- coding: utf-8 -*-

"""
This module defines the fit model used. Currently only supports scipy least_squares on CPU.

Ensure any fit functions you define are added to the AVAILABLE_FNS module variable.
Try not to have overlapping parameter names in the same fit.

Classes
-------
 - `QDMPy.fit_models.FitModel`
 - `QDMPy.fit_models.FitFunc`
 - `QDMPy.fit_models.Constant`
 - `QDMPy.fit_models.Linear`
 - `QDMPy.fit_models.Circular`
 - `QDMPy.fit_models.Gaussian`
 - `QDMPy.fit_models.Gaussian_hyperfine_14`
 - `QDMPy.fit_models.Gaussian_hyperfine_15`
 - `QDMPy.fit_models.Lorentzian`
 - `QDMPy.fit_models.Lorentzian_hyperfine_14`
 - `QDMPy.fit_models.Lorentzian_hyperfine_15`

Functions
---------
 - `QDMPy.fit_models.get_param_defn`
 - `QDMPy.fit_models.get_param_odict`
 - `QDMPy.fit_models.get_param_unit`

Module variables
----------------
 - `QDMPy.fit_models.AVAILABLE_FNS`
"""

# ============================================================================

__author__ = "Sam Scholten"

# ============================================================================

import numpy as np
from numba import njit
from collections import OrderedDict


# ================================================================================================
# ================================================================================================
#
# FitModel Class
#
# ================================================================================================
# ================================================================================================


class FitModel:
    """FitModel used to fit to data."""

    def __init__(self, fit_functions):
        """
        Arguments
        ---------
        fit_functions: dict
            Dict of functions to makeup the fit model, key: fitfunc name, val: number of
            independent copies of that fitfunc.
            format: {"linear": 1, "lorentzian": 8} etc., i.e. options["fit_functions"]
        """

        self.fit_functions = fit_functions

        fn_chain = []
        all_param_len = 0

        for fn_type, num_fns in fit_functions.items():
            for i in range(num_fns):
                next_fn = AVAILABLE_FNS[fn_type]  # NOTE: AVAILABLE_FNS defined at end of file
                next_fn_param_len = len(next_fn.param_defn)
                next_fn_param_indices = [all_param_len + i for i in range(next_fn_param_len)]
                all_param_len += next_fn_param_len
                fn_chain.append(next_fn(next_fn_param_indices))

        self.fn_chain = fn_chain

    # =================================

    def __call__(self, param_ar, sweep_vec):
        """
        Evaluates fitmodel for given parameter values and sweep (affine) parameter values.

        Arguments
        ---------
        param_ar : np array, 1D
            Array of parameters fed into each fitfunc (these are what are fit by sc)

        sweep_vec : np array, 1D or number
            Affine parameter where the fit model is evaluated

        Returns
        -------
        Fit model evaluates at sweep_vec (output is same format as sweep_vec input)
        """

        out = np.zeros(np.shape(sweep_vec))
        for fn in self.fn_chain:
            this_fn_params = param_ar[fn.this_fn_param_indices]
            out += fn.eval(sweep_vec, *this_fn_params)

        return out

    # =================================

    def residuals_scipy(self, param_ar, sweep_vec, pl_val):
        """Evaluates residual: fit model - PL value """
        return self.__call__(param_ar, sweep_vec) - pl_val

    # =================================

    def jacobian_scipy(self, param_ar, sweep_vec, pl_val):
        """Evaluates jacobian of fitmodel in format expected by scipy least_squares"""

        for i, fn in enumerate(self.fn_chain):
            this_fn_params = param_ar[fn.this_fn_param_indices]
            if not i:
                val = fn.grad_fn(sweep_vec, *this_fn_params)
            else:
                val = np.hstack((val, fn.grad_fn(sweep_vec, *this_fn_params)))
        return val


# ====================================================================================


def get_param_defn(fit_model):
    """Returns list of parameters in fit_model"""
    param_defn_ar = []
    for fn in fit_model.fn_chain:
        param_defn_ar.extend(fn.param_defn)
    return param_defn_ar


# =================================


# get ordered dict of key: param name, val: param unit, for all parameters in chain
def get_param_odict(fit_model):
    """
    get ordered dict of key: param_key (param_name), val: param_unit for all parameters in fit_model
    """
    param_dict = OrderedDict()
    for fn in fit_model.fn_chain:
        for i in range(len(fn.param_defn)):
            param_name = fn.param_defn[i] + "_0"
            param_unit = fn.param_units[fn.param_defn[i]]
            # ensure no overlapping param names
            while param_name in param_dict.keys():
                param_name = param_name[:-1] + str(int(param_name[-1]) + 1)

            param_dict[param_name] = param_unit
    return param_dict


# =================================


def get_param_unit(fit_model, param_name, param_number):
    """
    Get unit for a given param_key (given by param_name + "_" + param_number)
    """
    param_dict = get_param_odict(fit_model)
    return param_dict[param_name + "_" + str(param_number)]


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
        Argument
        --------
        param_indices : np array
            where the parameters for this fitfunc are located within broader fitmodel param array
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
        return np.empty(np.shape(x)).fill(c)

    # =================================

    @staticmethod
    @njit
    def grad_fn(x, c):
        """Compute the grad of the residue, excluding PL as a param
        {output shape: (len(x), 1)}
        """
        J = np.empty((x.shape[0], 1))
        J[:, 0] = 0
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
        {output shape: (len(x), 1)}
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

    param_defn = ["rabi_freq", "pos_circ", "amp_circ"]
    param_units = {"rabi_freq": "Nu (Hz)", "pos_circ": "Tau (s)", "amp_circ": "Amp (a.u.)"}

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
# ==========================================================================

# TODO add exponentials... consult T1, T2 people etc.
# careful -> don't want overlapping param definitions!!!
AVAILABLE_FNS = {
    "lorentzian": Lorentzian,
    "lorentzian_hyperfine_14": Lorentzian_hyperfine_14,
    "lorentzian_hyperfine_15": Lorentzian_hyperfine_15,
    "gaussian": Gaussian,
    "gaussian_hyperfine_14": Gaussian_hyperfine_14,
    "gaussian_hyperfine_15": Gaussian_hyperfine_15,
    "constant": Constant,
    "linear": Linear,
    "circular": Circular,
}
"""
Dictionary that defines fit functions available for use.

Add any functions you define here so you can use them.

Try not to overlap function parameter names.
"""


# ==========================================================================
# ==========================================================================

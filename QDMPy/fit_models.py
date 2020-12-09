# -*- coding: utf-8 -*-

"""
Module docstring
"""

# ============================================================================

__author__ = "Sam Scholten"

# ============================================================================

import numpy as np
from numba import njit, jit
from collections import OrderedDict

# ============================================================================


# TODO... might want to go through and speed-test njit, jit again :)


# ================================================================================================
# ================================================================================================
#
# FitFunc Class
#
# ================================================================================================
# ================================================================================================


class FitModel:
    # this model isn't used for gpufit
    def __init__(self, fit_functions):
        # fit_functions format: {"linear": 1, "lorentzian": 8} etc., i.e. options["fit_functions"]

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

        out = np.zeros(np.shape(sweep_vec))
        for fn in self.fn_chain:
            this_fn_params = param_ar[fn.this_fn_param_indices]
            out += fn.eval(sweep_vec, *this_fn_params)

        return out

    # =================================

    def residuals_scipy(self, param_ar, sweep_vec, pl_val):
        return self.__call__(param_ar, sweep_vec) - pl_val

    # =================================

    def jacobian_scipy(self, param_ar, sweep_vec, pl_val):

        for i, fn in enumerate(self.fn_chain):
            this_fn_params = param_ar[fn.this_fn_param_indices]
            if not i:
                val = fn.grad_fn(sweep_vec, *this_fn_params)
            else:
                val = np.hstack((val, fn.grad_fn(sweep_vec, *this_fn_params)))
        return val


# ====================================================================================


def get_param_defn(fit_model):
    param_defn_ar = []
    for fn in fit_model.fn_chain:
        param_defn_ar.extend(fn.param_defn)
    return param_defn_ar


# =================================


# get ordered dict of key: param name, val: param unit, for all parameters in chain
def get_param_odict(fit_model):
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
    param_dict = get_param_odict(fit_model)
    return param_dict[param_name + "_" + str(param_number)]


# ====================================================================================
# ====================================================================================


class FitFunc:

    # init should probably tell us where the params exist in the chain
    def __init__(self, param_indices):
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

    param_defn = ["c"]
    param_units = {"c": "Amplitude (a.u.)"}

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
    """Constant"""

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
    Call instance as instance(x, inital_paramater_list) where
    inital_parameter_list is a list of inital guesses for each Gaussian's
    _, _, _ in that order, i.e. [].
    fwhm, pos, amp in that order, i.e [fwhm, pos, amp, fwhm2, pos2, amp2...].
    """

    param_defn = ["rabi_freq", "pos", "amp"]
    param_units = {"rabi_freq": "Nu (Hz)", "pos": "Tau (s)", "amp": "Amp (a.u.)"}

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
    """
    Sum of abitary number of Gaussian Class.
    Define number of Gaussians on instansiation.
    Call instance as instance(x, inital_paramater_list) where
    inital_parameter_list is a list of inital guesses for each Gaussian's
    fwhm, pos, amp in that order, i.e [fwhm, pos, amp, fwhm2, pos2, amp2...].
    """

    param_defn = ["fwhm", "pos", "amp"]
    param_units = {"fwhm": "Freq (MHz)", "pos": "Freq (MHz)", "amp": "Amp (a.u.)"}

    # =================================

    @staticmethod
    @njit(fastmath=True)
    def eval(x, fwhm, pos, amp):
        return amp * np.exp(-SCALE_SIGMA * (x - pos) ** 2 / fwhm ** 2)


class Gaussian_hyperfine_14(FitFunc):

    param_defn = [
        "pos",
        "amp_1_hyp",
        "amp_2_hyp",
        "amp_3_hyp",
        "fwhm_1_hyp",
        "fwhm_2_hyp",
        "fwhm_3_hyp",
    ]
    param_units = {
        "pos": "Frequency (MHz)",
        "amp_1_hyp": "Amplitude (a.u.)",
        "amp_2_hyp": "Amplitude (a.u.)",
        "amp_3_hyp": "Amplitude (a.u.)",
        "fwhm_1_hyp": "Frequency (MHz)",
        "fwhm_2_hyp": "Frequency (MHz)",
        "fwhm_3_hyp": "Frequency (MHz)",
    }
    fn_type = "feature"

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

    param_defn = ["pos", "amp_1_hyp", "amp_2_hyp", "fwhm_1_hyp", "fwhm_2_hyp"]
    param_units = {
        "pos": "Frequency (MHz)",
        "amp_1_hyp": "Amplitude (a.u.)",
        "amp_2_hyp": "Amplitude (a.u.)",
        "fwhm_1_hyp": "Frequency (MHz)",
        "fwhm_2_hyp": "Frequency (MHz)",
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
    """
    Sum of abitary number of lorentzian Class.
    Define number of lorentzians on instansiation.
    Call instance as instance(x, inital_paramater_list) where
    inital_parameter_list is a list of inital guesses for each lorenzian's
    fwhm, pos, amp in that order, i.e [fwhm, pos, amp, fwhm2, pos2, amp2...].
    """

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
    """
    Sum of abitary number of lorentzian Class.
    Define number of lorentzians on instansiation.
    Call instance as instance(x, inital_paramater_list) where
    inital_parameter_list is a list of inital guesses for each lorenzian's
    fwhm, pos, amp in that order, i.e [fwhm, pos, amp, fwhm2, pos2, amp2...].
    """

    param_defn = [
        "pos",
        "amp_1_hyp",
        "amp_2_hyp",
        "amp_3_hyp",
        "fwhm_1_hyp",
        "fwhm_2_hyp",
        "fwhm_3_hyp",
    ]
    param_units = {
        "pos": "Frequency (MHz)",
        "amp_1_hyp": "Amplitude (a.u.)",
        "amp_2_hyp": "Amplitude (a.u.)",
        "amp_3_hyp": "Amplitude (a.u.)",
        "fwhm_1_hyp": "Frequency (MHz)",
        "fwhm_2_hyp": "Frequency (MHz)",
        "fwhm_3_hyp": "Frequency (MHz)",
    }

    # def __init__(self, num_peaks):
    #     super().__initt_(num_peaks)t

    # =================================

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
    """Sum of abitary number of lorentzian Class.
    Define number of lorentzians on instansiation.
    Call instance as instance(x, inital_paramater_list) whereg
    inital_parameter_list is a list of inital guesses for each lorenzian's
    fwhm, pos, amp in that order, i.e [fwhm, pos, amp, fwhm2, pos2, amp2...].
    """

    param_defn = ["pos", "amp_1_hyp", "amp_2_hyp", "fwhm_1_hyp", "fwhm_2_hyp"]
    param_units = {
        "pos": "Frequency (MHz)",
        "amp_1_hyp": "Amplitude (a.u.)",
        "amp_2_hyp": "Amplitude (a.u.)",
        "fwhm_1_hyp": "Frequency (MHz)",
        "fwhm_2_hyp": "Frequency (MHz)",
    }

    # def __init__(self, num_peaks):
    #     super().__init__(num_peaks)

    # =================================

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

# ==========================================================================
# ==========================================================================

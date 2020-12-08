# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, jit

# TODO... might want to go through and speed-test njit, jit again :)

# ================================================================================================
# ================================================================================================
#
# FitFunc Class
#
# ================================================================================================
# ================================================================================================


class FitFunc:
    """
    Parent class for arbitary fit functions
    num_fns is the number of (base) functions in this FitFunc
    i.e. 8 lorentzians, each with independent params, bounds, guesses
    """

    param_defn = []

    def __init__(self, num_fns, chain_fitfunc=None):
        self.num_fns = num_fns
        if chain_fitfunc is None:
            self.chain_param_len = 0
            self.chain_fitfunc = ChainTerminator()
        else:
            self.chain_fitfunc = chain_fitfunc
            self.chain_param_len = len(chain_fitfunc.get_param_defn())

    # =================================

    def __call__(self, sweep_vec, fit_params):
        """
        Returns the value of the fit function at sweep_val (i.e. freq, tau)
        for given fit_options. Vectorised (sweep_vec may be vector or number).
        """
        chain_params, these_params = np.split(fit_params, [self.chain_param_len])
        newoptions = these_params.reshape(self.num_fns, len(self.param_defn))

        outx = np.zeros(np.shape(sweep_vec))
        for f_params in newoptions:
            outx += self.base_fn(sweep_vec, *f_params)

        return outx + self.chain_fitfunc(sweep_vec, chain_params)

    # =================================

    def jacobian(self, sweep_vec, fit_params):
        """
        Returns the value of the fit function's jacobian at sweep_vals for
        given fit_params.
        shape: (len(sweep_val), num_fns*len(param_defn))
        """

        chain_params, params = np.split(fit_params, [self.chain_param_len])
        new_params = params.reshape(self.num_fns, len(self.param_defn))

        try:
            ftype = self.fn_type
        except AttributeError:
            # Just so we can check for chain termination (could be done more neatly)
            raise AttributeError("You need to define the type of your function.")

        for i, f_params in enumerate(new_params):
            if not i:
                output = self.grad_fn(sweep_vec, *f_params)
            else:
                # stack on next fn's grad to the jacobian
                output = np.hstack((output, self.grad_fn(sweep_vec, *f_params)))

        # chain terminator here adds on PL jacobian term
        # this is the recursive exit case
        # stop hstacking onto jac matrix
        if self.chain_fitfunc.fn_type == "terminator":
            return output

        # stack on the next fit functions jacobian (recursively)
        # NOTE the chain fitfuncs have to be added at the start as that's how its defined
        # in fit_model (gen_fit_params),
        return np.hstack((self.chain_fitfunc.jacobian(sweep_vec, chain_params), output))

    # =================================

    def get_param_defn(self):
        """
        Returns the chained parameter defintions.  Not sure if used and
        should be considered for removal or renaming as it is confusinigly similar
        to the static member variable param.defn which does not include chained
        functions.
        """
        try:
            return self.param_defn + self.chain_fitfunc.get_param_defn()
        except (AttributeError):
            return self.param_defn

    # =================================

    @staticmethod
    def base_fn(sweep_vec, *fit_params):
        raise NotImplementedError(
            "You shouldn't be here, go away. You MUST override base_fn, check your spelling."
        )

    # =================================

    @staticmethod
    def grad_fn(sweep_vec, *fit_params):
        """ if you want to use a grad_fn override this in the subclass """
        return None


# ============================================================================


class ChainTerminator(FitFunc):
    """
    Ends the chain of arbitrary fit functions. This needs to be here as we don't want
    circular dependencies.
    """

    param_defn = []
    param_units = {}
    fn_type = "terminator"
    chain_fitfunc = None

    # override the init for FitFunc
    def __init__(self):
        self.chain_param_len = 0
        self.num_fns = 0

    def __call__(self, *anything):
        """ contributes nothing to the residual """
        return 0

    @staticmethod
    def base_fn(*anything):
        raise NotImplementedError("you shouldn't be here")

    @staticmethod
    def grad_fn(sweep_vec, *anything):
        """ hstack the PL term onto the jacobian """
        return -np.ones(sweep_vec.shape[0], dtype=np.float32).reshape(-1, 1)


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
    fn_type = "bground"

    # =================================

    @staticmethod
    @njit(fastmath=True)
    def base_fn(x, c):
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
    fn_type = "bground"

    #    def __init__(self, num_peaks):
    #        super().__init__(num_peaks)

    # =================================

    # speed tested, marginally faster with fastmath off (idk why)
    @staticmethod
    @njit(fastmath=False)
    def base_fn(x, c, m):
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
    fn_type = "feature"

    @staticmethod
    @njit
    def base_fn(x, rabi_freq, pos, amp):
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
    fn_type = "feature"

    # =================================

    @staticmethod
    @njit(fastmath=True)
    def base_fn(x, fwhm, pos, amp):
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
    def base_fn(x, pos, amp_1_hyp, amp_2_hyp, amp_3_hyp, fwhm_1_hyp, fwhm_2_hyp, fwhm_3_hyp):
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
    fn_type = "feature"

    # A15 para = 3.03 MHz
    @staticmethod
    @njit(fastmath=True)
    def base_fn(x, pos, amp_1_hyp, amp_2_hyp, fwhm_1_hyp, fwhm_2_hyp):
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
    fn_type = "feature"

    # =================================

    # fastmath gives a ~10% speed up on my testing
    @staticmethod
    @njit(fastmath=True)
    def base_fn(x, fwhm, pos, amp):
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
    fn_type = "feature"

    # def __init__(self, num_peaks):
    #     super().__initt_(num_peaks)t

    # =================================

    # A14 para = -2.14 MHz
    @staticmethod
    @njit(fastmath=True)
    def base_fn(x, pos, amp_1_hyp, amp_2_hyp, amp_3_hyp, fwhm_1_hyp, fwhm_2_hyp, fwhm_3_hyp):
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
    fn_type = "feature"

    # def __init__(self, num_peaks):
    #     super().__init__(num_peaks)

    # =================================

    # A15 para = 3.03 MHz
    @staticmethod
    @njit(fastmath=True)
    def base_fn(x, pos, amp_1_hyp, amp_2_hyp, fwhm_1_hyp, fwhm_2_hyp):
        hwhmsqr1 = fwhm_1_hyp ** 2 / 4
        hwhmsqr2 = fwhm_2_hyp ** 2 / 4
        return amp_1_hyp * hwhmsqr1 / (
            (x - pos - 1.515) ** 2 + hwhmsqr1
        ) + amp_2_hyp * hwhmsqr2 / ((x - pos + 1.515) ** 2 + hwhmsqr2)

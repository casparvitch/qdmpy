# -*- coding: utf-8 -*-

__author__ = "Sam Scholten"

# NOTE
# ----
# This is the meat of where we want to do more work.
# importantly: this doesn't abstract nicely to gpu_fit
# do we want to have a loop on our gpu?

# ==========================================================================


import numpy as np

import fit_functions

# ==========================================================================


available_fns = {
    "lorentzian": fit_functions.Lorentzian,
    "lorentzian_hyperfine_14": fit_functions.Lorentfitzian_hyperfine_14,
    "lorentzian_hyperfine_15": fit_functions.Lorentzian_hyperfine_15,
    "gaussian": fit_functions.Gaussian,
    "gaussian_hyperfine_14": fit_functions.Gaussian_hyperfine_14,
    "gaussian_hyperfine_15": fit_functions.Gaussian_hyperfine_15,
    "constant": fit_functions.Constant,
    "linear": fit_functions.Linear,
    "circular": fit_functions.Circular,
}

# ==========================================================================

# data would usually be sig_norm
def fit_roi(data, sweep_list):
    pass


# ==========================================================================

# TODO this is where we need to make adjustments to bg function etc.
# {basically abstract this part here where there's a list, to being broader...?}
def define_fit_model(options):
    guess_dict, bound_dict = gen_init_guesses(options)
    fit_param_ar, fit_param_bound_ar = gen_fit_params(options, guess_dict, bound_dict)

    fit_model = FitModel(
        [options["lineshape"], options["bg_function"]],
        [options["num_peaks"], 1],
        guess_dict,
        bound_dict,
        fit_param_ar,
        fit_param_bound_ar,
    )

    options["fit_param_ar"] = fit_model.peaks.param_defn
    options["fit_parameter_unit"] = fit_model.peaks.parameter_unit

    return fit_model


# ==========================================================================


# TODO make 'background' be a part of this normal process
# --> "functions": {"linear": 1, "lorentzian": 8} etc. can handle num_peaks too then.
# TODO probably want to just pass in options, init can handle the rest


class FitModel:
    def __init__(
        self,
        fn_names,
        num_fns,
        guess_dict,
        bound_dict,
        fit_param_ar,
        fit_param,  # ?
        fit_param_bound_ar,
    ):
        # deal with user passing single fns not as single lists
        try:
            self.functions = available_fns[fn_names](num_fns)
            self.fn_names = [fn_names]
            self.num_fns = [num_fns]
        except TypeError:
            peak_chain = None
            self.param_keys = []
            # Loading these in reverse fixed some issue but I don't remember what.
            # It shouldn't matter *shrug*
            for single_fn, single_num_fns in zip(fn_names[::-1], num_fns[::-1]):
                peak_chain = available_fns[single_fn](single_num_fns, peak_chain)
            self.peaks = peak_chain
            # we reverse these for simplicity for other users, as the chain is reversed.
            self.fns = fns[::-1]
            self.num_fns = num_fns[::-1]

        self.init_guesses = guess_dict
        self.init_bounds = bound_dict
        self.fit_param_ar = fit_param_ar
        self.fit_param_bound_ar = fit_param_bound_ar

    def residuals_scipy(self, fit_param_ar, sweep_val, pl_val):
        return self.peaks(sweep_val, fit_param_ar) - pl_val

    def jacobian_scipy(self, fit_param_ar, sweep_val, pl_val):
        return self.peaks.jacobian(sweep_val, fit_param_ar)


class FitModel:
    def __init__(self, fns, guess_dict, bound_dict, fit_param_ar, fit_param_bound_ar):
        # what is input form? dict?
        # i.e. single function fit
        if len(fns) == 1:
            self.fns = available_fns[fns]()


# ==========================================================================

# TODO these below need to be adjusted + improved


def gen_init_guesses(options):
    guess_dict = {}
    bound_dict = {}
    peaks = available_fns[options["lineshape"]](options["num_peaks"])
    for param_key in peaks.param_defn:
        # < TODO > add an auto guess for the peak positions?
        # use scipy.signal.find_peaks?
        guess = options[param_key + "_guess"]
        val = guess
        if param_key + "_range" in options:
            if type(guess) is list and len(guess) > 1:
                val_b = [
                    [
                        x - options[param_key + "_range"],
                        x + options[param_key + "_range"],
                    ]
                    for x in guess
                ]
            else:
                # print(guess)
                # print(options[param_key + "_range"])
                val_b = [
                    guess - options[param_key + "_range"],
                    guess + options[param_key + "_range"],
                ]
        elif param_key + "_bounds" in options:
            val_b = options[param_key + "_bounds"]
        else:
            val_b = [[0, 0]]
        if val is not None:
            guess_dict[param_key] = val
            bound_dict[param_key] = np.array(val_b)
        else:
            raise RuntimeError(
                "I'm not sure what this means... I know "
                + "it's bad though... Don't put 'None' as "  # noqa: W503
                + "a param guess."  # noqa: W503
            )
    return guess_dict, bound_dict


# =================================


def gen_fit_params(options, guess_dict, bound_dict):

    num_fns = options["num_peaks"]
    fn = options["lineshape"]
    peaks = available_fns[fn](num_fns)

    fit_param_ar = np.array([])
    fit_param_bound_ar = np.array([])

    bg_function_name = options["bg_function"]
    bg_params = options["bg_parameter_guess"]
    bg_bounds = options["bg_parameter_bounds"]

    # TODO generalise for arbitary number of BG functions
    for n in range(len(available_fns[bg_function_name].param_defn)):
        fit_param_ar = np.append(fit_param_ar, bg_params[n])
        fit_param_bound_ar = np.append(fit_param_bound_ar, bg_bounds[n])

    for n in range(num_fns):
        to_append = np.zeros(len(peaks.param_defn))
        to_append_bounds = np.zeros((len(peaks.param_defn), 2))
        for position, key in enumerate(peaks.param_defn):
            try:
                to_append[position] = guess_dict[key][n]
            except (TypeError, KeyError):
                to_append[position] = guess_dict[key]
            if len(bound_dict[key].shape) == 2:
                to_append_bounds[position] = bound_dict[key][n]
            else:
                to_append_bounds[position] = bound_dict[key]

        fit_param_ar = np.append(fit_param_ar, to_append)
        fit_param_bound_ar = np.append(fit_param_bound_ar, to_append_bounds)
    # This is a messy way to deal with this, why not keep its shape as you append.
    fit_param_bound_ar = fit_param_bound_ar.reshape(
        len(bg_params) + options["num_peaks"] * len(peaks.param_defn), 2
    )

    return fit_param_ar, fit_param_bound_ar

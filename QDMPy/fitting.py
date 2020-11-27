# -*- coding: utf-8 -*-

__author__ = "Sam Scholten"

# NOTE
# ----
# This is the meat of where we want to do more work.
# importantly: this doesn't abstract nicely to gpu_fit
# -> write differently for gpu (including ROI fit etc.! different file completely...)

# ==========================================================================


import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm
import psutil
import os
import concurrent.futures
from itertools import repeat
import warnings

import systems
import fit_functions

# ==========================================================================

# TODO add exponentials... consult T1, T2 people etc.
AVAILABLE_FNS = {
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


class FitResultROI:
    """Just an object to hold values, and a place to define their names..."""

    def __init__(
        self,
        fit_options,
        pl_roi,
        sweep_vector,
        best_fit_result,
        scipy_best_fit,
        init_fit,
        fit_sweep_vector,
    ):
        self.fit_options = fit_options  # options passed to scipy least_squares
        self.pl_roi = pl_roi  # pl data summed over FOV, 1D (as fn of sweep_vec)
        self.sweep_vector = sweep_vector  # affine parameter i.e. tau or freq
        self.best_fit_result = best_fit_result  # scipy solution fit_params
        self.scipy_best_fit = scipy_best_fit  # scipy best fit (array of PL vals)
        self.init_fit = init_fit  # initial fit (array of PL vals)

        # the values (sweep vals) that correspond to x coordinate of above PL vals in
        # scipy_best_fit and init_fit
        self.fit_sweep_vector = fit_sweep_vector


# ==========================================================================


def limit_cpu():
    """is called at every process start"""
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


# ==========================================================================


# simple generator to shape data as expected by least_squares in concurrent method
def my_gen(our_array):
    len_z, len_x, len_y = np.shape(our_array)
    for x in range(len_x):
        for y in range(len_y):
            yield [x, y, our_array[:, x, y]]


# ==========================================================================


def to_squares_wrapper(fun, p0, sweep_val, pl_val, kwargs={}):
    return ((pl_val[0], pl_val[1]), least_squares(fun, p0, args=(sweep_val, pl_val[2]), **kwargs))


# ==========================================================================


def fit_roi(options, sig_norm, sweep_list, fit_model):
    systems.clean_options(options)

    # fit *all* pl data (i.e. summing over FOV)
    # collapse to just pl_ar (as function of sweep, 1D)
    pl_roi = np.nansum(np.nansum(sig_norm, 2), 1)
    pl_roi = (pl_roi / np.max(pl_roi)).copy()  # .copy() untested 2020-11-27

    sweep_vector = np.linspace(min(sweep_list), max(sweep_list))  # not sure how this is useful?

    init_guess = fit_model.fit_param_ar
    fit_bounds = (fit_model.fit_param_bound_ar[:, 0], fit_model.fit_param_bound_ar[:, 1])

    fit_options = {
        "method": options["fit_method"],
        "verbose": options["fit_gtol"],
        "xtol": options["fit_xtol"],
        "ftol": options["fit_ftol"],
        "loss": options["loss_fn"],
    }

    if options["fit_method"] != "lm":
        fit_options["bounds"] = fit_bounds
        fit_options["verbose"] = options["verbose_fitting"]

    if options["scale_x"]:
        fit_options["x_scale"] = "jac"
    else:
        options["scale_x"] = False

    # define jacobian option for least_squares fitting
    if fit_model.jacobian_scipy is None or not options["use_analytic_jac"]:
        fit_options["jac"] = options["fit_jac_acc"]
    else:
        fit_options["jac"] = fit_model.jacobian_scipy

    fitting_results = least_squares(
        fit_model.residuals_scipy, init_guess, args=(sweep_list, pl_roi), **fit_options
    )

    best_fit_result = fitting_results.x
    fit_sweep_vector = np.linespace(np.min(sweep_vector), np.max(sweep_vector), 10000)
    scipy_best_fit = fit_model.fn_chain(fit_sweep_vector, best_fit_result)
    init_fit = fit_model.fn_chain(fit_sweep_vector, init_guess)

    return FitResultROI(
        fit_options,
        pl_roi,
        sweep_vector,
        best_fit_result,
        scipy_best_fit,
        init_fit,
        fit_sweep_vector,
    )


# ==========================================================================

# FIXME option to scramble pixels
# just operate on sig_norm, there should be a method in numpy only on specific axes (last 2)

# ar = ...
# rng = np.random.default_rng()
# x_shuf = rng.permutation(ar.shape[1])
# x_unshuf = np.argsort(x_shuf)
# y_shuf = rng.permutation(ar.shape[2])
# y_unshuf = np.argsort(y_shuf)
# ok how do we use it now...

# ar_shuffled_in_x = ar[:, shuf_x, :]
# ar_shuffled_in_xy = ar_shuffled_in_x[:, :, shuf_y]
# can then go backwards with same process...
# does that mean we need to store fit info in the same way?
# But that's ok as we _do_ store the info (after get_pixel_fitting_results) in image arrays!!!
# NOTE ok this will actually work :o


def fit_pixels(options, sig_norm, sweep_list, fit_model, roi_fit_result):
    systems.clean_options(options)

    sweep_ar = np.array(sweep_list)
    threads = options["threads"]
    data_length = np.shape(sig_norm)[1] * np.shape(sig_norm)[2]

    # this makes low binning work (idk why), else do chunksize = 1
    chunksize = int(data_length / (threads * 100))

    if not chunksize:
        warnings.warn("chunksize was 0, setting to 1")
        chunksize = 1

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=threads, initializer=limit_cpu
    ) as executor:
        fit_results = list(
            tqdm(
                executor.map(
                    to_squares_wrapper,
                    repeat(fit_model.residuals_scipy),
                    repeat(roi_fit_result.best_fit_result),
                    repeat(sweep_ar),
                    my_gen(sig_norm),
                    repeat(roi_fit_result.fit_options),
                    chunksize=chunksize,
                ),
                ascii=True,
                mininterval=1,
                total=data_length,
                unit=" PX",
                disable=(not options["show_progressbar"]),
            )
        )

    # again, roi_shape here untested...
    return get_pixel_fitting_results(fit_model, fit_results, (sig_norm[1], sig_norm[2]))


# ==========================================================================


def fit_AOIs(options, sig_norm, sweep_list, fit_model, AOIs, roi_fit_result):
    systems.clean_options(options)

    sweep_ar = np.array(sweep_list)
    threads = options["threads"]
    chunksize = 1

    fit_results_list = []

    for AOI in tqdm(
        AOIs,
        ascii=True,
        mininterval=1,
        unit=" AOI",
        disable=(not options["show_progressbar"]),
    ):
        aoi_sig_norm = sig_norm[:, AOI[0], AOI[1]].copy()
        # ok now fit that AOI region

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=threads, initializer=limit_cpu
        ) as executor:
            fit_results_list.append(
                list(
                    executor.map(
                        to_squares_wrapper,
                        repeat(fit_model.residuals_scipy),
                        repeat(roi_fit_result.best_fit_result),
                        repeat(sweep_ar),
                        my_gen(aoi_sig_norm),
                        repeat(roi_fit_result.fit_options),
                        chunksize=chunksize,
                    )
                )
            )

    AOI_fit_params = []
    for i, AOI_res in enumerate(fit_results_list):
        # NOTE AOIs[i] as region_shape below NOT TESTED {~~~hopefully it works~~~}
        AOI_fit_params.append(get_pixel_fitting_results(fit_model, AOI_res, AOIs[i]))

    return AOI_fit_params


# ==========================================================================


def get_pixel_fitting_results(fit_model, fit_results, roi_shape):
    """
    Take the fit result data and back it down into a dictionary of arrays
    with keys representing the peak parameters (i.e. fwhm, pos, amp).  Each
    array is 3D [z,x,y] with the z dimension representing the peak number and
    x and y the lateral map .  Any added functions (background ect are
    handled by this in the same way.  I.e. with a linear background there will
    be 'm' and 'c' keys with a shape (1,x,y).
    """
    fit_image_results = {}
    # Populate the dictionary with the correct size empty arrays using np.zeros.

    for fn_type, num in fit_model.fns.items():
        fn_params = AVAILABLE_FNS[fn_type].param_defn
        for param_key in fn_params:
            fit_image_results[param_key] = np.zeros((num, roi_shape[0], roi_shape[1])) * np.nan
    # Fill the arrays element-wise from the results function, which returns a
    # 1D array of flattened best-fit parameters.
    for (x, y), result in fit_results:
        # num tracks position in the 1D array, as we loop over different fns
        # and parameters.
        num = 0
        for fn_type, num in fit_model.fns.items():
            fn_params = AVAILABLE_FNS[fn_type].param_defn
            for idx in range(num):
                for parameter_key in fn_params:
                    fit_image_results[parameter_key][idx, x, y] = result.x[num]
                    num += 1
    return fit_image_results


# ==========================================================================


def define_fit_model(options):

    fit_model = FitModel(options)

    options["fit_param_ar"] = fit_model.peaks.param_defn
    options["fit_parameter_unit"] = fit_model.peaks.parameter_unit

    return fit_model


# ==========================================================================


class FitModel:
    # this model isn't used for gpufit
    def __init__(self, options):

        systems.clean_options(options)
        self.options = options

        self._gen_init_guesses()
        self._gen_fit_params()

        fns = self.options["fit_functions"]  # format: {"linear": 1, "lorentzian": 8} etc.
        fn_chain = None
        self.param_keys = []
        for fn_type in fns:
            fn_chain = AVAILABLE_FNS[fn_type](fns[fn_type], fn_chain)
        self.fn_chain = fn_chain[::-1]  # reverse for simplicity, as chain is reversed

    # =================================

    # TODO decompose into smaller fns
    def _gen_init_guesses(self):
        self.init_guesses = {}
        self.init_bounds = {}

        for fn_type, num in self.options["fit_functions"].items():
            fit_func = AVAILABLE_FNS[fn_type](num)
            for param_key in fit_func.param_defn:
                guess = self.options[param_key + "_guess"]
                if param_key + "_range" in self.options:
                    rang = self.options[param_key + "_range"]
                    if type(guess) is list and len(guess) > 1:

                        # separate bounds for each fn of this type
                        if type(rang) is list and len(rang) > 1:
                            bounds = [
                                [each_guess - each_range, each_guess + each_range]
                                for each_guess, each_range in zip(guess, rang)
                            ]
                        # separate guess for each fn of this type, all with same range
                        else:
                            bounds = [
                                [
                                    each_guess - rang,
                                    each_guess + rang,
                                ]
                                for each_guess in guess
                            ]
                    else:
                        # FIXME handle this in options cleaner
                        if type(rang) is list:
                            if len(rang) == 1:
                                rang = rang[0]
                            else:
                                raise RuntimeError("param range len should match guess len")
                        # param guess and range are just single vals (easy!)
                        else:
                            bounds = [
                                guess - rang,
                                guess + rang,
                            ]
                elif param_key + "_bounds" in self.options:
                    # assumes bounds are passed in with correct formatatting
                    bounds = self.options[param_key + "_bounds"]
                else:
                    # FIXME again, handle in options cleaner.
                    raise RuntimeError("Provide bounds for this param")

                if guess is not None:
                    self.init_guesses[param_key] = guess
                    self.init_bounds[param_key] = np.array(bounds)
                else:
                    raise RuntimeError("Not sure why your guess is None?")

    # =================================

    def _gen_fit_params(self):
        param_lst = []
        bound_lst = []

        for fn_type, num in self.options["fit_functions"].items():
            for n in range(num):

                for pos, key in enumerate(fn_type.param_defn):
                    try:
                        param_lst.append(self.init_guesses[key][n])
                    except (TypeError, KeyError):
                        param_lst.append(self.init_guesses[key])
                    if len(self.init_bounds[key].shape) == 2:
                        bound_lst.append(self.init_bounds[key][n])
                    else:
                        bound_lst.append(self.init_bounds[key])

        self.fit_param_ar = np.array(param_lst)  # shape: num_params
        self.fit_param_bound_ar = np.array(bound_lst)  # shape: num_params, 2

    # =================================

    def residuals_scipy(self, fit_param_ar, sweep_val, pl_val):
        return self.fn_chain(sweep_val, fit_param_ar) - pl_val

    # =================================

    def jacobian_scipy(self, fit_param_ar, sweep_val, pl_val):
        return self.fn_chain.jacobian(sweep_val, fit_param_ar)


# ================================================================================================


def save_param_fit_images(options, fit_model, fit_image_results):
    """Save fit param arrays as txt files"""

    systems.clean_options(options)

    # NOTE untested
    for fn_type, num in options["fit_functions"].items():
        fit_func = AVAILABLE_FNS[fn_type](num)
        for param_key in fit_func.param_defn:
            for idx in range(num):
                np.savetxt(
                    options["output_dir"] + "/fn_" + str(param_key) + "_" + str(idx + 1) + ".txt",
                    fit_image_results[param_key][idx, :, :],
                )

# -*- coding: utf-8 -*-
"""
Module docstring
"""

# ============================================================================

__author__ = "Sam Scholten"

# ==========================================================================


import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm
import psutil
import os
import concurrent.futures
from itertools import repeat
import warnings
from sys import platform

# ============================================================================


import systems
import fit_models
import data_loading


# ==========================================================================

# Notes
# -----
# This is the meat of where we want to do more work.
# importantly: this doesn't abstract nicely to gpu_fit
# -> write differently for gpu (including ROI fit etc.! different file completely...)


class FitResultROIAvg:
    """Just an object to hold values, and a place to define their names..."""

    def __init__(
        self,
        fit_options,
        pl_roi,
        sweep_list,
        best_fit_result,
        best_fit_pl_vals,
        scipy_best_fit,
        init_fit,
        fit_sweep_vector,
    ):
        self.fit_options = fit_options  # options passed to scipy least_squares

        self.pl_roi = pl_roi  # pl data summed over FOV, 1D (as fn of sweep_vec)
        self.sweep_list = sweep_list  # affine parameter i.e. tau or freq

        self.best_fit_result = best_fit_result  # scipy solution fit_params
        self.best_fit_pl_vals = best_fit_pl_vals  # fit_model(best_fit_result, sweep_list)

        # below: a higher res linspace of the above data.
        self.scipy_best_fit = scipy_best_fit  # scipy best fit (array of PL vals)
        self.init_fit = init_fit  # initial fit (array of PL vals)

        # the values (sweep vals) that correspond to x coordinate of above PL vals in
        # scipy_best_fit and init_fit
        self.fit_sweep_vector = fit_sweep_vector


# ==========================================================================


def fit_ROI_avg(options, sig_norm, sweep_list, fit_model):
    systems.clean_options(options)

    # fit *all* pl data (i.e. summing over FOV)
    # collapse to just pl_ar (as function of sweep, 1D)
    pl_roi = np.nansum(np.nansum(sig_norm, 2), 1)
    pl_roi = (pl_roi / np.max(pl_roi)).copy()  # .copy() untested 2020-11-27

    # this is just constructing the initial parameter guesses and bounds in the right format
    fit_param_ar, fit_param_bound_ar = _gen_fit_params(options, *_gen_init_guesses(options))
    init_guess = fit_param_ar
    fit_bounds = (fit_param_bound_ar[:, 0], fit_param_bound_ar[:, 1])

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
    fit_sweep_vector = np.linspace(np.min(sweep_list), np.max(sweep_list), 10000)
    scipy_best_fit = fit_model(best_fit_result, fit_sweep_vector)
    init_fit = fit_model(init_guess, fit_sweep_vector)

    return FitResultROIAvg(
        fit_options,
        pl_roi,
        sweep_list,
        best_fit_result,
        fit_model(best_fit_result, sweep_list),
        scipy_best_fit,
        init_fit,
        fit_sweep_vector,
    )


# ==========================================================================


def fit_single_pixel(options, pixel_pl_ar, sweep_list, fit_model, roi_avg_fit_result):
    " fit single pixel and return best_fit_resul t"

    fit_opts = {}
    for key, val in roi_avg_fit_result.fit_options.items():
        fit_opts[key] = val

    sweep_ar = np.array(sweep_list)

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        guess_params = roi_avg_fit_result.best_fit_result.copy()
    else:
        # this is just constructing the initial parameter guesses and bounds in the right format
        fit_param_ar, fit_param_bound_ar = _gen_fit_params(options, *_gen_init_guesses(options))
        guess_params = fit_param_ar.copy()

    fitting_results = least_squares(
        fit_model.residuals_scipy, guess_params, args=(sweep_list, pixel_pl_ar), **fit_opts
    )
    return fitting_results.x


# ==========================================================================


def fit_AOIs(options, sig_norm, sweep_list, fit_model, AOIs, roi_avg_fit_result):
    # TODO: implement pixel scrambler here too.

    systems.clean_options(options)

    fit_opts = {}
    for key, val in roi_avg_fit_result.fit_options.items():
        fit_opts[key] = val

    sweep_ar = np.array(sweep_list)

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        guess_params = roi_avg_fit_result.best_fit_result.copy()
    else:
        # this is just constructing the initial parameter guesses and bounds in the right format
        fit_param_ar, fit_param_bound_ar = _gen_fit_params(options, *_gen_init_guesses(options))
        guess_params = fit_param_ar.copy()

    AOI_avg_best_fit_results_lst = []

    for AOI in AOIs:
        aoi_sig_norm = sig_norm[:, AOI[0], AOI[1]]
        aoi_avg = np.nansum(np.nansum(aoi_sig_norm, 2), 1)
        aoi_avg = (aoi_avg / np.max(aoi_avg)).copy()

        fitting_results = least_squares(
            fit_model.residuals_scipy, guess_params, args=(sweep_list, aoi_avg), **fit_opts
        )
        AOI_avg_best_fit_results_lst.append(fitting_results.x)

    return AOI_avg_best_fit_results_lst


# ==========================================================================


def limit_cpu():
    """is called at every process start"""
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use p.nice(19)
    if platform.startswith("linux"):  # linux
        p.nice(19)
    elif platform.startswith("win32"):  # windows
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    elif platform.startswith("darwin"):  # macOS
        warnings.warn("Not sure what to use for macOS... skipping cpu limitting")
    else:  # 'freebsd', 'aix', 'cygwin'...
        warnings.warn(f"Not sure what to use for your OS: {platform}... skipping cpu limitting")


# ==========================================================================


# simple generator to shape data as expected by least_squares in concurrent method
# allows us to keep track of *where* (i.e. which pixel location) each result corresponds to
# (see also: to_squares wrapper)
def my_gen(our_array):
    len_z, len_x, len_y = np.shape(our_array)
    for x in range(len_x):
        for y in range(len_y):
            yield [x, y, our_array[:, x, y]]


# ==========================================================================


def to_squares_wrapper(fun, p0, sweep_val, shaped_data, kwargs={}):
    # shaped_data: [x, y, pl]
    # output: (x, y), result_params
    return (
        (shaped_data[0], shaped_data[1]),
        least_squares(fun, p0, args=(sweep_val, shaped_data[2]), **kwargs).x,
    )


# ==========================================================================


def shuffle_pixels(data_3d):
    rng = np.random.default_rng()

    x_shuf = rng.permutation(data_3d.shape[1])
    x_unshuf = np.argsort(x_shuf)
    y_shuf = rng.permutation(data_3d.shape[2])
    y_unshuf = np.argsort(y_shuf)

    # are these the correct indices? i.e. is zeroth index the freq. sweep?
    shuffled_in_x = data_3d[:, x_shuf, :]
    shuffled_in_xy = shuffled_in_x[:, :, y_shuf]

    # return shuffled pixels, and arrays to unshuffle
    return shuffled_in_xy.copy(), (x_unshuf, y_unshuf)


# =================================


def unshuffle_pixels(data_2d, unshuffler):
    x_unshuf, y_unshuf = unshuffler
    unshuffled_in_x = data_2d[x_unshuf, :]
    unshuffled_in_xy = unshuffled_in_x[:, y_unshuf]
    return unshuffled_in_xy.copy()


# =================================


def unshuffle_fit_results(fit_result_dict, unshuffler):
    for key, array in fit_result_dict.items():
        fit_result_dict[key] = unshuffle_pixels(array, unshuffler)
    return fit_result_dict


# ==========================================================================


def fit_pixels(options, sig_norm, sweep_list, fit_model, roi_avg_fit_result):
    systems.clean_options(options)

    sweep_ar = np.array(sweep_list)
    threads = options["threads"]
    data_length = np.shape(sig_norm)[1] * np.shape(sig_norm)[2]

    # this makes low binning work (idk why), else do chunksize = 1
    chunksize = int(data_length / (threads * 100))

    # randomize order of fitting pixels (will un-scramble later) so ETA is more correct
    if options["scramble_pixels"]:
        pixel_data, unshuffler = shuffle_pixels(sig_norm)
    else:
        pixel_data = sig_norm

    if not chunksize:
        warnings.warn("chunksize was 0, setting to 1")
        chunksize = 1

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        init_params = roi_avg_fit_result.best_fit_result.copy()
    else:
        # this is just constructing the initial parameter guesses and bounds in the right format
        fit_param_ar, fit_param_bound_ar = _gen_fit_params(options, *_gen_init_guesses(options))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=threads, initializer=limit_cpu
    ) as executor:
        fit_results = list(
            tqdm(
                executor.map(
                    to_squares_wrapper,
                    repeat(fit_model.residuals_scipy),
                    repeat(init_params),
                    repeat(sweep_ar),
                    my_gen(pixel_data),
                    repeat(roi_avg_fit_result.fit_options),
                    chunksize=chunksize,
                ),
                ascii=True,
                mininterval=1,
                total=data_length,
                unit=" PX",
                disable=(not options["show_progressbar"]),
            )
        )

    roi_shape = np.shape(sig_norm)
    res = get_pixel_fitting_results(fit_model, fit_results, (roi_shape[1], roi_shape[2]))
    if options["scramble_pixles"]:
        # I don't think this is required as we keep track of posn' with to_squares_wrapper
        return unshuffle_fit_results(res, unshuffler)
    else:
        return res


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
    # NOTE look closely at fit_results, this seems like a slow way to extract?

    # initialise dictionary with key: val = param_name: param_units
    fit_image_results = fit_models.get_param_odict(fit_model)

    # override with correct size empty arrays using np.zeros.
    for key in fit_image_results.keys():
        fit_image_results[key] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan

    # Fill the arrays element-wise from the results function, which returns a
    # 1D array of flattened best-fit parameters.

    for (x, y), result in fit_results:
        filled_params = {}  # keep track of index, i.e. pos_0, for this pixel
        for fn in fit_model.fn_chain:
            for param_num, param_name in enumerate(fn.param_defn):

                # keep track of what index we're up to, i.e. pos_1
                if param_name not in filled_params.keys():
                    key = param_name + "_0"
                    filled_params[param_name] = 1
                else:
                    key = param_name + "_" + str(filled_params[param_name])
                    filled_params[param_name] += 1

                fit_image_results[key][x, y] = result[fn.this_fn_param_indices[param_num]]

    return fit_image_results


# ================================================================================================


def load_prev_fit_results(options, fit_model):
    """load results from previous fit"""

    prev_options = data_loading.get_prev_options(options)

    a_fit_func = prev_options["fit_functions"].keys()[0]
    a_fit_param = fit_models.AVAILABLE_FNS[a_fit_func].param_defn[0]

    # read a random param to get the shape
    shape = read_processed_param(options, a_fit_func, a_fit_param + "_0").shape

    # init the results dict
    # FIXME idx doesn't do anything now! -> follow get_pixel_fitting_results
    fit_param_res_dict = {}
    for fn_type, num in prev_options["fit_functions"].items():
        for idx in range(num):
            params = fit_models.AVAILABLE_FNS[fn_type].param_defn
            for param_name in params:
                fit_param_res_dict[param_name + "_" + str(idx)] = np.zeros(
                    (num, shape[0], shape[1])
                )

    # Read in the previous fitted data
    for fn_type, num in prev_options["fit_functions"].items():
        for idx in num:
            params = fit_models.AVAILABLE_FNS[fn_type].param_defn
            for param_name in params:
                fit_param_res_dict[param_name][idx, :, :] = read_processed_param(
                    options, fn_type, param_name + "_" + str(idx)
                )


# ================================================================================================


def read_processed_param(options, fn_type, fit_param):
    return np.loadtxt(options["data_dir"] + "/" + fn_type + "_" + fit_param + ".txt")


# ================================================================================================


def define_fit_model(options):

    fit_functions = options["fit_functions"]

    fit_model = fit_models.FitModel(fit_functions)

    # for the record
    options["fit_param_defn"] = fit_models.get_param_odict(fit_model)

    return fit_model


# ==========================================================================


def _gen_init_guesses(options):
    init_guesses = {}
    init_bounds = {}

    for fn_type, num in options["fit_functions"].items():
        fit_func = fit_models.AVAILABLE_FNS[fn_type](num)
        for param_key in fit_func.param_defn:
            guess = options[param_key + "_guess"]
            if param_key + "_range" in options:
                bounds = _bounds_from_range(options, param_key)
            elif param_key + "_bounds" in options:
                # assumes bounds are passed in with correct formatatting
                bounds = options[param_key + "_bounds"]
            else:
                raise RuntimeError(
                    "Provide bounds for this param."
                    + "\n"  # noqa: W503
                    + "Note to maintainer: this should be handled in options cleaner"  # noqa: W503
                )

            if guess is not None:
                init_guesses[param_key] = guess
                init_bounds[param_key] = np.array(bounds)
            else:
                raise RuntimeError("Not sure why your guess is None?")

    return init_guesses, init_bounds


# =================================


def _bounds_from_range(options, param_key):
    guess = options[param_key + "_guess"]
    rang = options[param_key + "_range"]
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
        # handle this in options cleaner?
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
    return bounds


# =================================


def _gen_fit_params(options, init_guesses, init_bounds):
    param_lst = []
    bound_lst = []

    for fn_type, num in options["fit_functions"].items():
        for n in range(num):

            for pos, key in enumerate(fit_models.AVAILABLE_FNS[fn_type].param_defn):
                try:
                    param_lst.append(init_guesses[key][n])
                except (TypeError, KeyError):
                    param_lst.append(init_guesses[key])
                if len(init_bounds[key].shape) == 2:
                    bound_lst.append(init_bounds[key][n])
                else:
                    bound_lst.append(init_bounds[key])

    fit_param_ar = np.array(param_lst)  # shape: num_params
    fit_param_bound_ar = np.array(bound_lst)  # shape: num_params, 2
    return fit_param_ar, fit_param_bound_ar


# ================================================================================================


def save_param_fit_images(options, fit_model, fit_image_results):
    """Save fit param arrays as txt files"""

    systems.clean_options(options)

    # NOTE untested
    # FIXME idx doesn't do anything now, rewrite this bad boi
    for fn_type, num in options["fit_functions"].items():
        fit_func = fit_models.AVAILABLE_FNS[fn_type](num)
        for param_key in fit_func.param_defn:
            for idx in range(num):
                # TODO check this is consistent with reading in prev. data (and makes sense etc...)
                np.savetxt(
                    options["output_dir"]
                    + "/"
                    + fn_type
                    + "_"
                    + str(param_key)
                    + "_"
                    + str(idx + 1)
                    + ".txt",
                    fit_image_results[param_key][idx, :, :],
                )

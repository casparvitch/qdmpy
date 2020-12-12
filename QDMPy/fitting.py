# -*- coding: utf-8 -*-
"""
This module holds tools for fitting raw data, as loaded/reshaped by data_loading.

Classes
-------
 - `QDMPy.fitting.FitResultROIAvg`

Functions
---------
 - `QDMPy.fitting.prepare_fit_options`
 - `QDMPy.fitting.fit_ROI_avg`
 - `QDMPy.fitting.fit_single_pixel`
 - `QDMPy.fitting.fit_AOIs`
 - `QDMPy.fitting.limit_cpu`
 - `QDMPy.fitting.my_gen`
 - `QDMPy.fitting.to_squares_wrapper`
 - `QDMPy.fitting.shuffle_pixels`
 - `QDMPy.fitting.unshuffle_pixels`
 - `QDMPy.fitting.unshuffle_fit_results`
 - `QDMPy.fitting.fit_pixels`
 - `QDMPy.fitting.get_pixel_fitting_results`
 - `QDMPy.fitting.load_prev_fit_results`
 - `QDMPy.fitting.load_fit_param`
 - `QDMPy.fitting.define_fit_model`
 - `QDMPy.fitting.gen_init_guesses`
 - `QDMPy.fitting.bounds_from_range`
 - `QDMPy.fitting.gen_fit_params`
"""

# ============================================================================

__author__ = "Sam Scholten"

# ==========================================================================


import numpy as np
from scipy.optimize import least_squares
from tqdm.autonotebook import tqdm  # auto detects jupyter
import psutil
import os
import concurrent.futures
from itertools import repeat
import warnings
from sys import platform

# ============================================================================


import QDMPy.systems as systems
import QDMPy.fit_models as fit_models
import QDMPy.data_loading as data_loading
import QDMPy.misc as misc

# ==========================================================================


class FitResultROIAvg:
    """Just an object to hold values, and a place to define their names."""

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
        """
        Arguments
        ---------
        fit_options : dict
            Options passed to scipy least_squares.

        pl_roi : np array, 1D
            PL data summed over FOV, as fn of sweep_vec.

        sweep_list : np array, 1D
            Affine parameter i.e. tau or frequency.

        best_fit_result : np array, 1D
            Scipy solution fit parameters array.

        best_fit_pl_vals : np array, 1D
            fit_model(best_fit_result, sweep_list).

        scipy_best_fit : np array, 1D
            Array of PL vals for scipy's best fit. Higher res (longer) than best_fit_pl_vals.

        init_fit : np array, 1D
            Initial guess of fit model, i.e. an array of PL vals (same length as scipy_best_fit).

        fit_sweep_vector : np array, 1D
            The values (sweep values) as x coordinate corresponding to above two arrays.
        """
        self.fit_options = fit_options
        """
        fit_options : dict
            Options passed to scipy least_squares.
        """

        self.pl_roi = pl_roi
        self.sweep_list = sweep_list

        self.best_fit_result = best_fit_result
        self.best_fit_pl_vals = best_fit_pl_vals

        self.scipy_best_fit = scipy_best_fit
        self.init_fit = init_fit
        self.fit_sweep_vector = fit_sweep_vector

    def savejson(self, filename, dir):
        """ Save all attributes as a json file in dir/filename, via `misc.dict_to_json` """

        output_dict = {
            "pl_roi": self.pl_roi,
            "sweep_list": self.sweep_list,
            "best_fit_result": self.best_fit_result,
            "best_fit_pl_vals": self.best_fit_pl_vals,
            "scipy_best_fit": self.scipy_best_fit,
            "init_fit": self.init_fit,
            "fit_sweep_vector": self.fit_sweep_vector,
        }
        misc.dict_to_json(output_dict, filename, dir)


# ==========================================================================


def prepare_fit_options(options, fit_model):
    """
    General options dict -> fit_options, init_guesses in format that scipy least_squares expects.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    fit_model : `fit_models.FitModel` object.

    Returns
    -------
    fit_options : dict
        Dictionary with options that scipy.optimize.least_squares expects, specific to fitting.

    init_guess : np array, 1D (shape: num_params)
        Array of parameter values as 'initial guess' of fit model.
    """

    # this is just constructing the initial parameter guesses and bounds in the right format
    fit_param_ar, fit_param_bound_ar = gen_fit_params(options, *gen_init_guesses(options))
    init_guess = fit_param_ar
    fit_bounds = (fit_param_bound_ar[:, 0], fit_param_bound_ar[:, 1])

    # see scipy.optimize.least_squares
    fit_options = {
        "method": options["fit_method"],
        "verbose": options["verbose_fitting"],
        "gtol": options["fit_gtol"],
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
    return fit_options, init_guess


# ==========================================================================


def fit_ROI_avg(options, sig_norm, sweep_list, fit_model):
    """
    Fit the average of the measurement over the region of interest specified.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    sig_norm : np array, 3D
        Normalised measurement array, shape: [sweep_list, x, y].

    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)

    fit_model : `QDMPy.fit_models.FitModel` object.

    Returns
    -------
    `QDMPy.fitting.FitResultROIAvg` object containing the fit result (see class specifics)
    """

    systems.clean_options(options)

    # fit *all* pl data (i.e. summing over FOV)
    # collapse to just pl_ar (as function of sweep, 1D)
    pl_roi = np.nansum(np.nansum(sig_norm, 2), 1)
    pl_roi = (pl_roi / np.max(pl_roi)).copy()  # .copy() untested 2020-11-27

    fit_options, init_guess = prepare_fit_options(options, fit_model)

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
    """
    Fit Single pixel and return best_fit_result.x (i.e. the optimal fit parameters)

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    pixel_pl_ar : np array, 1D
        Normalised PL as function of sweep_list for a single pixel.

    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)

    fit_model : `QDMPy.fit_models.FitModel` object.

    roi_avg_fit_result : `QDMPy.fitting.FitResultROIAvg`
        `QDMPy.fitting.FitResultROIAvg` object, to pull fit_options from.

    Returns
    -------
    `fitting.FitResultROIAvg` object containing the fit result (see class specifics)
    """

    fit_opts = {}
    for key, val in roi_avg_fit_result.fit_options.items():
        fit_opts[key] = val

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        guess_params = roi_avg_fit_result.best_fit_result.copy()
    else:
        # this is just constructing the initial parameter guesses and bounds in the right format
        fit_param_ar, fit_param_bound_ar = gen_fit_params(options, *gen_init_guesses(options))
        guess_params = fit_param_ar.copy()

    fitting_results = least_squares(
        fit_model.residuals_scipy, guess_params, args=(sweep_list, pixel_pl_ar), **fit_opts
    )
    return fitting_results.x


# ==========================================================================


def fit_AOIs(options, sig_norm, sweep_list, fit_model, AOIs, roi_avg_fit_result):
    """
    Fit Single pixel and return best_fit_result.x (i.e. the optimal fit parameters)

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    sig_norm : np array, 3D
        Normalised measurement array, shape: [sweep_list, x, y].

    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)

    fit_model : `fit_models.FitModel` object.

    AOIs : list
        List of AOI specifications - each a length-2 iterable that can be used to directly index
        into sig_norm to return that AOI region, e.g. sig_norm[:, AOI[0], AOI[1]].

    roi_avg_fit_result : `fitting.FitResultROIAvg`
        `QDMPy.fitting.FitResultROIAvg` object, to pull `QDMPy.fitting.FitResultROIAvg.fit_options`
        from.

    Returns
    -------
    AOI_avg_best_fit_results_lst : list
        List of fit_result.x arrays (i.e. list of best fit parameters)
    """

    systems.clean_options(options)

    fit_opts = {}
    for key, val in roi_avg_fit_result.fit_options.items():
        fit_opts[key] = val

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        guess_params = roi_avg_fit_result.best_fit_result.copy()
    else:
        # this is just constructing the initial parameter guesses and bounds in the right format
        fit_param_ar, fit_param_bound_ar = gen_fit_params(options, *gen_init_guesses(options))
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
    """Called at every process start, to reduce the priority of this process"""
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


def my_gen(our_array):
    """
    Simple generator to shape data as expected by least_squares in concurrent method.

    Also allows us to track *where* (i.e. which pixel location) each result corresponds to.
    See also: `fitting.to_squares_wrapper`

    Arguments
    ---------
    our_array : np array, 3D
        Shape: [sweep_list, x, y]

    Returns
    -------
    [x, y, our_array[:, x, y]] : generator (yielded)
    """
    len_z, len_x, len_y = np.shape(our_array)
    for x in range(len_x):
        for y in range(len_y):
            yield [x, y, our_array[:, x, y]]


# ==========================================================================


def to_squares_wrapper(fun, p0, sweep_vec, shaped_data, kwargs={}):
    """
    Simple wrapper of scipy.optimize.least_squares to allow us to keep track of which
    solution is which (or where).

    Arguments
    ---------
    fun : function
        Function object acting as residual (fit model minus pl value)

    p0 : np array
        Initial guess: array of parameters

    sweep_vec : np array
        Array (or I guess single value, anything iterable) of affine parameter (tau/freq)

    shaped_data : list (3 elements)
        array returned by `fitting.my_gen`: [x, y, sig_norm[:, x, y]]

    kwargs : dict
        Other options (dict) passed to least_squares, i.e. fit_options

    Returns
    -------
    (x, y), least_squares(...).x
        I.e. the position of the fit result, and then the fit result parameters array.
    """
    # shaped_data: [x, y, pl]
    # output: (x, y), result_params
    return (
        (shaped_data[0], shaped_data[1]),
        least_squares(fun, p0, args=(sweep_vec, shaped_data[2]), **kwargs).x,
    )


# ==========================================================================


def shuffle_pixels(data_3d):
    """
    Simple shuffler

    Arguments
    ---------
    data_3d : np array, 3D
        i.e. sig_norm data, [affine param, x, y].

    Returns
    -------
    shuffled_in_xy : np array, 3D
        data_3d shuffled in 2nd, 3rd axis.

    (x_unshuf, y_unshuf) : Both np arrays
        Can be used to unshuffle shuffled_in_xy, i.e. through `fitting.unshuffled_pixels`.
    """

    rng = np.random.default_rng()

    x_shuf = rng.permutation(data_3d.shape[1])
    x_unshuf = np.argsort(x_shuf)
    y_shuf = rng.permutation(data_3d.shape[2])
    y_unshuf = np.argsort(y_shuf)

    shuffled_in_x = data_3d[:, x_shuf, :]
    shuffled_in_xy = shuffled_in_x[:, :, y_shuf]

    # return shuffled pixels, and arrays to unshuffle
    return shuffled_in_xy.copy(), (x_unshuf, y_unshuf)


# =================================


def unshuffle_pixels(data_2d, unshuffler):
    """
    Simple shuffler

    Arguments
    ---------
    data_2d : np array, 2D
        i.e. 'image' of a single fit parameter, all shuffled up!

    unshuffler : (x_unshuf, y_unshuf)
        Two arrays returned by fitting.shuffle_pixels that allow unshuffling of data_2d.

    Returns
    -------
    unshuffled_in_xy: np array, 2D
        data_2d but the inverse operation of `fitting.shuffle_pixels` has been applied
    """
    x_unshuf, y_unshuf = unshuffler
    unshuffled_in_x = data_2d[x_unshuf, :]
    unshuffled_in_xy = unshuffled_in_x[:, y_unshuf]
    return unshuffled_in_xy.copy()


# =================================


def unshuffle_fit_results(fit_result_dict, unshuffler):
    """
    Simple shuffler

    Arguments
    ---------
    fit_result_dict : dict
        Dictionary, key: param_names, val: image (2D) of param values across FOV. Each image
        requires reshuffling (which this function achieves).

    unshuffler : (x_unshuf, y_unshuf)
        Two arrays returned by fitting.shuffle_pixels that allow unshuffling of data_2d.

    Returns
    -------
    fit_result_dict : same as input, but each fit parameter has been unshuffled.
    """
    for key, array in fit_result_dict.items():
        fit_result_dict[key] = unshuffle_pixels(array, unshuffler)
    return fit_result_dict


# ==========================================================================


def fit_pixels(options, sig_norm, sweep_list, fit_model, roi_avg_fit_result):
    """
    Fits each pixel and returns dictionary of param_name -> param_image.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    sig_norm : np array, 3D
        Normalised measurement array, shape: [sweep_list, x, y].

    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)

    fit_model : `fit_models.FitModel` object.

    roi_avg_fit_result : `fitting.FitResultROIAvg`
        `fitting.FitResultROIAvg` object, to pull fit_options from.

    Returns
    -------
    fit_result_dict : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    """
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
        fit_param_ar, fit_param_bound_ar = gen_fit_params(options, *gen_init_guesses(options))

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
    if options["scramble_pixels"]:
        # I don't think this is required as we keep track of posn' with to_squares_wrapper
        return unshuffle_fit_results(res, unshuffler)
    else:
        return res


# ==========================================================================


def get_pixel_fitting_results(fit_model, fit_results, roi_shape):
    """
    Take the fit result data from scipy and back it down to a dictionary of arrays.

    Each array is 2D, representing the values for each parameter (specified by the dict key).


    Arguments
    ---------

    fit_model : `fit_models.FitModel` object.

    fit_results : list of (x, y), result values (see `fitting.to_squares_wrapper`)
        A list of each pixels parameter array: fit_result.x (as well as position in image
        denoted by (x, y), returned by the concurrent mapper in `fitting.fit_pixels`.

    roi_shape : iterable, length 2
        Shape of the region of interest, to allow us to generate the right shape empty arrays.
        This is probably a little useless, we could extract from any of the results.

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    """
    # initialise dictionary with key: val = param_name: param_units
    fit_image_results = fit_models.get_param_odict(fit_model)

    # override with correct size empty arrays using np.zeros
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


def load_prev_fit_results(options):
    """Load (all) parameter fit results from previous processing."""

    prev_options = data_loading.get_prev_options(options)

    fit_param_res_dict = {}

    for fn_type, num in prev_options["fit_functions"].items():
        for param_name in fit_models.AVAILABLE_FNS[fn_type].param_defn:
            for n in range(num):
                param_key = param_name + "_" + str(n)
                fit_param_res_dict[param_key] = load_fit_param(options, param_key)
    return fit_param_res_dict


# ================================================================================================


def load_fit_param(options, param_key):
    """Load a previously fit param, of name 'param_key'."""
    return np.loadtxt(options["data_dir"] / (param_key + ".txt"))


# ================================================================================================


def define_fit_model(options):
    """Define (and return) fit_model object, from options dictionary."""

    fit_functions = options["fit_functions"]

    fit_model = fit_models.FitModel(fit_functions)

    # for the record
    options["fit_param_defn"] = fit_models.get_param_odict(fit_model)

    return fit_model


# ==========================================================================


def gen_init_guesses(options):
    """
    Generate initial guesses (and bounds) in fit parameters from options dictionary.

    Both are returned as dictionaries, you need to use `fitting.get_fit_params` to convert to
    the correct (array) format.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    init_guesses : dict
        Dict holding guesses for each parameter, e.g. key -> list of guesses for each independent
        version of that fn_type.

    init_bounds : dict
        Dict holding guesses for each parameter, e.g. key -> list of bounds for each independent
        version of that fn_type.
    """
    init_guesses = {}
    init_bounds = {}

    for fn_type, num in options["fit_functions"].items():
        fit_func = fit_models.AVAILABLE_FNS[fn_type](num)
        for param_key in fit_func.param_defn:
            guess = options[param_key + "_guess"]
            if param_key + "_range" in options:
                bounds = bounds_from_range(options, param_key)
            elif param_key + "_bounds" in options:
                # assumes bounds are passed in with correct formatatting
                bounds = options[param_key + "_bounds"]
            else:
                raise RuntimeError(f"Provide bounds for the {fn_type}.{param_key} param.")

            if guess is not None:
                init_guesses[param_key] = guess
                init_bounds[param_key] = np.array(bounds)
            else:
                raise RuntimeError(f"Not sure why your guess for {fn_type}.{param_key} is None?")

    return init_guesses, init_bounds


# =================================


def bounds_from_range(options, param_key):
    """Generate parameter bounds (list, len 2) when given a range option."""
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


def gen_fit_params(options, init_guesses, init_bounds):
    """
    Generate arrays of initial fit guesses and bounds in correct form for scipy least_squares.

    init_guesses and init_bounds are dictionaries up to this point, we now convert to np arrays,
    that scipy will recognise. In particular, we specificy that each of the 'num' of each 'fn_type'
    have independent parameters, so must have independent init_guesses and init_bounds when
    plugged into scipy.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    init_guesses : dict
        Dict holding guesses for each parameter, e.g. key -> list of guesses for each independent
        version of that fn_type.

    init_bounds : dict
        Dict holding guesses for each parameter, e.g. key -> list of bounds for each independent
        version of that fn_type.

    Returns
    -------
    fit_param_ar : np array, shape: num_params

    fi_param_bound_ar : np array, shape: (num_params, 2)
    """
    param_lst = []
    bound_lst = []

    for fn_type, num in options["fit_functions"].items():
        # extract a guess/bounds for each of the copies of each fn_type (e.g. 8 lorentzians)
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

    fit_param_ar = np.array(param_lst)
    fit_param_bound_ar = np.array(bound_lst)
    return fit_param_ar, fit_param_bound_ar

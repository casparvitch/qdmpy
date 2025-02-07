# -*- coding: utf-8 -*-
"""
This module holds some functions and classes that are shared between different
fitting backends, but are not a part of the user-facing interface.

Classes
-------
 - `qdmpy.pl.common.FitResultCollection`
 - `qdmpy.pl.common.ROIAvgFitResult`

Functions
---------
 - `qdmpy.pl.common.shuffle_pixels`
 - `qdmpy.pl.common.unshuffle_pixels`
 - `qdmpy.pl.common.unshuffle_fit_results`
 - `qdmpy.pl.common.pixel_generator`
 - `qdmpy.pl.common.gen_init_guesses`
 - `qdmpy.pl.common.bounds_from_range`
 - `qdmpy.pl.common.get_pixel_fitting_results`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.pl.common.FitResultCollection": True,
    "qdmpy.pl.common.ROIAvgFitResult": True,
    "qdmpy.pl.common.shuffle_pixels": True,
    "qdmpy.pl.common.unshuffle_pixels": True,
    "qdmpy.pl.common.unshuffle_fit_results": True,
    "qdmpy.pl.common.pixel_generator": True,
    "qdmpy.pl.common.gen_init_guesses": True,
    "qdmpy.pl.common.bounds_from_range": True,
    "qdmpy.pl.common.get_pixel_fitting_results": True,
}

# ============================================================================

import numpy as np
import copy
from scipy.linalg import svd

# ============================================================================

import qdmpy.pl.funcs

# ============================================================================


class FitResultCollection:
    """Object to hold AOI average fit results, and a place to define their names."""

    def __init__(
        self,
        fit_backend,
        roi_avg_fit_result,
        single_pixel_result,
        aoi_fit_results_lst,
    ):
        """
        Arguments
        ---------
        fit_backend : str
            Name of the fit backend (e.g. scipy, gpufit, etc.) used.
        roi_avg_fit_result
            `qdmpy.pl.common.ROIAvgFitResult` object.
        single_pixel_result
            Best (optimal) fit/model parameters for single pixel check.
        aoi_fit_results_lst : list of lists
            List of (list of) best (optimal) parameters, for each AOI region (avg).
        """
        self.fit_backend = fit_backend
        self.ROI_avg_fit_result = roi_avg_fit_result
        self.single_pixel_fit_result = single_pixel_result
        self.AOI_fit_results_lst = aoi_fit_results_lst


# ============================================================================


class ROIAvgFitResult:
    """Object to hold ROI average fit result, and a place to define result names."""

    def __init__(
        self,
        fit_backend,
        fit_options,
        fit_model,
        pl_roi,
        sweep_list,
        best_params,
        init_param_guess,
        init_param_bounds,
    ):
        """
        Arguments
        ---------
        fit_backend : string
            Name of the fit backend (e.g. scipy, gpufit, etc.)
        fit_options : dict
            Options dictionary for this fit method, as will be passed to fitting function.
            E.g. scipy least_squares is handed various options as a dictionary.
        fit_model : `qdmpy.pl.model.FitModel`
            Fit model used. Can construct fit via res.fit_model(res.best_params, res.sweep_list).
        pl_roi : np array, 1D
            pl data summed over FOV, as fn of sweep_vec.
        sweep_list : np array, 1D
            Affine parameter i.e. tau or frequency.
        best_params : np array, 1D
            Solution fit parameters array.
        """

        self.fit_backend = fit_backend
        """
        fit_backend : str
            Name of the fit method (e.g. scipy, gpufit, etc.)
        """

        self.fit_options = fit_options
        """
        fit_options : dict
            Options dictionary for this fit method, as will be passed to fitting function.
            E.g. scipy least_squares is handed various options as a dictionary.
        """
        self.fit_model = fit_model

        self.pl_roi = pl_roi
        self.sweep_list = sweep_list

        self.best_params = best_params
        self.init_param_guess = init_param_guess
        self.init_param_bounds = init_param_bounds

    def savejson(self, filename, directory):
        """
        Save all attributes as a json file in dir/filename, via
        `qdmpy.shared.json2dict.dict_to_json`
        """

        output_dict = {
            "fit_backend": self.fit_backend,
            "pl_roi": self.pl_roi,
            "sweep_list": self.sweep_list,
            "best_params": self.best_params,
            "init_param_guess": self.init_param_guess,
            "init_param_bounds": self.init_param_bounds,
        }
        qdmpy.shared.json2dict.dict_to_json(output_dict, filename, directory)


# ============================================================================


def shuffle_pixels(data_3d):
    """
    Simple shuffler

    Arguments
    ---------
    data_3d : np array, 3D
        i.e. sig_norm data, [affine param, y, x].

    Returns
    -------
    shuffled_in_yx : np array, 3D
        data_3d shuffled in 2nd, 3rd axis.
    unshuffler : (y_unshuf, x_unshuf)
        Both np array. Can be used to unshuffle shuffled_in_yx, i.e. through `qdmpy.pl.common.unshuffle_pixels`.
    """

    rng = np.random.default_rng()

    y_shuf = rng.permutation(data_3d.shape[1])
    y_unshuf = np.argsort(y_shuf)
    x_shuf = rng.permutation(data_3d.shape[2])
    x_unshuf = np.argsort(x_shuf)

    shuffled_in_y = data_3d[:, y_shuf, :]
    shuffled_in_yx = shuffled_in_y[:, :, x_shuf]

    # return shuffled pixels, and arrays to unshuffle
    return shuffled_in_yx.copy(), (y_unshuf, x_unshuf)


# =================================


def unshuffle_pixels(data_2d, unshuffler):
    """
    Simple shuffler

    Arguments
    ---------
    data_2d : np array, 2D
        i.e. 'image' of a single fit parameter, all shuffled up!
    unshuffler : (y_unshuf, x_unshuf)
        Two arrays returned by `qdmpy.pl.common.shuffle_pixels that allow unshuffling of data_2d.

    Returns
    -------
    unshuffled_in_yx: np array, 2D
        data_2d but the inverse operation of `qdmpy.pl.common.shuffle_pixels` has been applied
    """
    y_unshuf, x_unshuf = unshuffler
    unshuffled_in_y = data_2d[y_unshuf, :]
    unshuffled_in_yx = unshuffled_in_y[:, x_unshuf]
    return unshuffled_in_yx.copy()


# =================================


def unshuffle_fit_results(fit_result_dict, unshuffler):
    """
    Simple shuffler

    Arguments
    ---------
    fit_result_dict : dict
        Dictionary, key: param_names, val: image (2D) of param values across FOV. Each image
        requires reshuffling (which this function achieves).
        Also has 'residual' as a key.
    unshuffler : (y_unshuf, x_unshuf)
        Two arrays returned by `qdmpy.pl.common.shuffle_pixels` that allow unshuffling of data_2d.

    Returns
    -------
    fit_result_dict : dict
        Same as input, but each fit parameter has been unshuffled.
    """
    for key, array in fit_result_dict.items():
        fit_result_dict[key] = unshuffle_pixels(array, unshuffler)
    return fit_result_dict


# ============================================================================


def pixel_generator(our_array):
    """
    Simple generator to shape data as expected by to_squares_wrapper in scipy concurrent method.

    Also allows us to track *where* (i.e. which pixel location) each result corresponds to.
    See also: `qdmpy.pl.scipyfit.to_squares_wrapper`, and corresponding gpufit method.

    Arguments
    ---------
    our_array : np array, 3D
        Shape: [sweep_list, y, x]

    Returns
    -------
    generator : list
        [y, x, our_array[:, y, x]] generator (yielded)
    """
    _, len_y, len_x = np.shape(our_array)
    for y in range(len_y):
        for x in range(len_x):
            yield [y, x, our_array[:, y, x]]


# ============================================================================


def gen_init_guesses(options):
    """
    Generate initial guesses (and bounds) in fit parameters from options dictionary.

    Both are returned as dictionaries, you need to use 'gen_{scipy/gpufit/...}_init_guesses'
    to convert to the correct (array) format for each specific fitting backend.


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
        fit_func = qdmpy.pl.funcs.AVAILABLE_FNS[fn_type](num)
        for param_key in fit_func.param_defn:
            guess = options[param_key + "_guess"]
            if param_key + "_range" in options:
                bounds = bounds_from_range(options, param_key, guess)
            elif param_key + "_bounds" in options:
                # assumes bounds are passed in with correct formatatting
                bounds = options[param_key + "_bounds"]
            else:
                raise RuntimeError(
                    f"Provide bounds for the {fn_type}.{param_key} param."
                )

            if guess is not None:
                init_guesses[param_key] = guess
                init_bounds[param_key] = np.array(bounds)
            else:
                raise RuntimeError(
                    f"Not sure why your guess for {fn_type}.{param_key} is None?"
                )

    return init_guesses, init_bounds


# ============================================================================


def bounds_from_range(options, param_key, guess):
    """
    Generate parameter bounds (list, len 2) when given a range option.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    param_key : str
        paramater key, e.g. "pos".
    guess : float/int or array
        guess for param, or list of guesses for a given parameter.

    Returns
    -------
    bounds : list
        bounds for each parameter. Dimension depends on dimension of param guess.
    """
    rang = options[param_key + "_range"]
    if isinstance(guess, (list, tuple)) and len(guess) > 1:
        # separate bounds for each fn of this type
        if isinstance(rang, (list, tuple)) and len(rang) > 1:
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
        if isinstance(rang, (list, tuple)):
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


# ============================================================================


def get_pixel_fitting_results(fit_model, fit_results, pixel_data, sweep_list):
    """
    Take the fit result data from scipyfit/gpufit and back it down to a dictionary of arrays.

    Each array is 2D, representing the values for each parameter (specified by the dict key).


    Arguments
    ---------
    fit_model : `qdmpy.pl.model.FitModel`
        Model we're fitting to.
    fit_results : list of [(y, x), result, jac] objects
        (see `qdmpy.pl.scipyfit.to_squares_wrapper`, or corresponding gpufit method)
        A list of each pixel's parameter array, as well as position in image denoted by (y, x).
    pixel_data : np array, 3D
        Normalised measurement array, shape: [sweep_list, y, x]. i.e. sig_norm.
        May or may not already be shuffled (i.e. matches fit_results).
    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq).

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    sigmas : dict
        Dictionary, key: param_keys, val: image (2D) of param uncertainties across FOV.
    """

    roi_shape = np.shape(pixel_data)[1:]

    # initialise dictionary with key: val = param_name: param_units
    fit_image_results = fit_model.get_param_odict()
    sigmas = copy.copy(fit_image_results)

    # override with correct size empty arrays using np.zeros
    for key in fit_image_results.keys():
        fit_image_results[key] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan
        sigmas[key] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan

    fit_image_results["residual_0"] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan

    # Fill the arrays element-wise from the results function, which returns a
    # 1D array of flattened best-fit parameters.
    for (y, x), result, jac in fit_results:
        filled_params = {}  # keep track of index, i.e. pos_0, for this pixel

        if jac is None: # can't fit this pixel
            fit_image_results["residual_0"][y, x] = np.nan
            perr = np.empty(np.shape(result))
            perr[:] = np.nan
        else:
            # NOTE we decide not to call each backend separately here
            resid = fit_model.residuals_scipyfit(result, sweep_list, pixel_data[:, y, x])
            fit_image_results["residual_0"][y, x] = np.sum(
                np.abs(resid, dtype=np.float64), dtype=np.float64
            )
            # uncertainty (covariance matrix), copied from scipy.optimize.curve_fit (not abs. sigma)
            _, s, vt = svd(jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(jac.shape) * s[0]
            s = s[s > threshold]
            vt = vt[: s.size]
            pcov = np.dot(vt.T / s**2, vt)
            # NOTE using/assuming linear cost fn,
            cost = 2 * 0.5 * np.sum(fit_model(result, sweep_list) ** 2)
            s_sq = cost / (len(resid) - len(result))
            pcov *= s_sq
            perr = np.sqrt(np.diag(pcov))  # array of standard deviations

        for param_num, param_name in enumerate(fit_model.get_param_defn()):
            # keep track of what index we're up to, i.e. pos_1
            if param_name not in filled_params.keys():
                key = param_name + "_0"
                filled_params[param_name] = 1
            else:
                key = param_name + "_" + str(filled_params[param_name])
                filled_params[param_name] += 1

            fit_image_results[key][y, x] = result[param_num]
            sigmas[key][y, x] = perr[param_num]

    return fit_image_results, sigmas

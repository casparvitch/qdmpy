# -*- coding: utf-8 -*-
"""
This module holds some functions and classes that are shared between different
fitting backends, but are not a part of the user-facing interface.

Functions
---------
 - `qdmpy.hamiltonian._shared.gen_init_guesses`
 - `qdmpy.hamiltonian._shared.bounds_from_range`
 - `qdmpy.hamiltonian._shared.pixel_generator`
 - `qdmpy.hamiltonian._shared.shuffle_pixels`
 - `qdmpy.hamiltonian._shared.unshuffle_pixels`
 - `qdmpy.hamiltonian._shared.unshuffle_fit_results`
 - `qdmpy.hamiltonian._shared.get_pixel_fitting_results`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.hamiltonian._shared.gen_init_guesses": True,
    "qdmpy.hamiltonian._shared.bounds_from_range": True,
    "qdmpy.hamiltonian._shared.pixel_generator": True,
    "qdmpy.hamiltonian._shared.shuffle_pixels": True,
    "qdmpy.hamiltonian._shared.unshuffle_pixels": True,
    "qdmpy.hamiltonian._shared.unshuffle_fit_results": True,
    "qdmpy.hamiltonian._shared.get_pixel_fitting_results": True,
}

# ============================================================================

import numpy as np
from scipy.linalg import svd
import copy

# ============================================================================

import qdmpy.hamiltonian._hamiltonians

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
    if options["auto_read_B"]:
        import qdmpy.field as Qfield  # avoid circular import

        bias_x, bias_y, bias_z = Qfield.get_B_bias(options)
        override_guesses = {"Bx": bias_x, "By": bias_y, "Bz": bias_z}

    from qdmpy.constants import AVAILABLE_HAMILTONIANS  # avoid circ. imports

    ham = AVAILABLE_HAMILTONIANS[options["hamiltonian"]]
    for param_key in ham.param_defn:

        if param_key in override_guesses:
            guess = override_guesses[param_key]
        else:
            guess = options[param_key + "_guess"]

        if param_key + "_range" in options:
            bounds = bounds_from_range(options, param_key, guess)
        elif param_key + "_bounds" in options:
            # assumes bounds are passed in with correct formatatting
            bounds = options[param_key + "_bounds"]
        else:
            raise RuntimeError(f"Provide bounds for the {ham}.{param_key} param.")

        if guess is not None:
            init_guesses[param_key] = guess
            init_bounds[param_key] = np.array(bounds)
        else:
            raise RuntimeError(f"Not sure why your guess for {ham}.{param_key} is None?")

    return init_guesses, init_bounds


# ============================================================================


def bounds_from_range(options, param_key, guess):
    """Generate parameter bounds (list, len 2) when given a range option."""
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


# ============================================================================


def pixel_generator(our_array):
    """
    Simple generator to shape data as expected by to_squares_wrapper in scipy concurrent method.

    Also allows us to track *where* (i.e. which pixel location) each result corresponds to.
    See also: `qdmpy.hamiltonian._scipyfit.to_squares_wrapper`.

    Arguments
    ---------
    our_array : np array, 3D
        Shape: [idx, y, x] (idx for each bnv, freq etc.)

    Returns
    -------
    generator : list
        [y, x, our_array[:, y, x]] generator (yielded)
    """
    len_z, len_y, len_x = np.shape(our_array)
    for y in range(len_y):
        for x in range(len_x):
            yield [y, x, our_array[:, y, x]]


# ============================================================================


def shuffle_pixels(data_3d):
    """
    Simple shuffler

    Arguments
    ---------
    data_3d : np array, 3D
        i.e. freqs/bnv data, [idx, y, x].

    Returns
    -------
    shuffled_in_yx : np array, 3D
        data_3d shuffled in 2nd, 3rd axis.
    unshuffler : (y_unshuf, x_unshuf)
        Both np array. Can be used to unshuffle shuffled_in_yx, i.e. through
        `qdmpy.hamiltonian._shared.unshuffle_pixels`.
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
        Two arrays returned by `qdmpy.hamiltonian._shared.shuffle_pixels`
        that allow unshuffling of data_2d.

    Returns
    -------
    unshuffled_in_yx: np array, 2D
        data_2d but the inverse operation of `qdmpy.hamiltonian._shared.shuffle_pixels`
        has been applied.
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
        Two arrays returned by `qdmpy.hamiltonian._shared.shuffle_pixels` that allow
        unshuffling of data_2d.

    Returns
    -------
    fit_result_dict : dict
        Same as input, but each fit parameter has been unshuffled.
    """
    for key, array in fit_result_dict.items():
        fit_result_dict[key] = unshuffle_pixels(array, unshuffler)
    return fit_result_dict


# ============================================================================


def get_pixel_fitting_results(hamiltonian, fit_results, pixel_data):
    """
    Take the fit result data from scipyfit/gpufit and back it down to a dictionary of arrays.

    Each array is 2D, representing the values for each parameter (specified by the dict key).


    Arguments
    ---------
    fit_model : `qdmpy.hamiltonian._hamiltonians.Hamiltonian`
        Model we're fitting to.
    fit_results : list of [(y, x), result, jac] objects
        (see `qdmpy.hamiltonian._scipyfit.to_squares_wrapper`)
        A list of each pixel's parameter array, as well as position in image denoted by (y, x).
    pixel_data : np array, 3D
        Normalised measurement array, shape: [idx, y, x]. i.e. bnvs.
        May or may not already be shuffled (i.e. matches fit_results).

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    sigmas: dict
        As fit_image_results, but containing parameters errors (standard deviations) across FOV.
    """

    roi_shape = np.shape(pixel_data)[1:]

    # initialise dictionary with key: val = param_name: param_units
    fit_image_results = qdmpy.hamiltonian._hamiltonians.get_param_odict(hamiltonian)
    sigmas = copy.copy(fit_image_results)

    # override with correct size empty arrays using np.zeros
    for key in fit_image_results.keys():
        fit_image_results[key] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan
        sigmas[key] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan

    fit_image_results["residual_field"] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan

    # Fill the arrays element-wise from the results function, which returns a
    # 1D array of flattened best-fit parameters.
    for (y, x), result, jac in fit_results:
        resid = hamiltonian.residuals_scipyfit(result, pixel_data[:, y, x])
        fit_image_results["residual_field"][y, x] = np.sum(
            np.abs(resid, dtype=np.float64), dtype=np.float64
        )
        # uncertainty (covariance matrix), copied from scipy.optimize.curve_fit
        _, s, VT = svd(jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[: s.size]
        pcov = np.dot(VT.T / s ** 2, VT)
        perr = np.sqrt(np.diag(pcov))  # array of standard deviations

        for param_num, param_name in enumerate(hamiltonian.param_defn):
            fit_image_results[param_name][y, x] = result[param_num]
            sigmas[param_name][y, x] = perr[param_num]

    return fit_image_results, sigmas


# ============================================================================

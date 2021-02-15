# -*- coding: utf-8 -*-
"""
This module holds some functions and classes that are shared between different
fitting backends, but are not a part of the user-facing interface.


Functions
---------
 - `QDMPy.hamiltonian._shared.gen_init_guesses`
 - `QDMPy.hamiltonian._shared.bounds_from_range`
 - `QDMPy.hamiltonian._shared.shuffle_pixels`
 - `QDMPy.hamiltonian._shared.unshuffle_pixels`
 - `QDMPy.hamiltonian._shared.unshuffle_fit_results`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.hamiltonian._shared.": True,
}

# ============================================================================

import numpy as np

# ============================================================================


from QDMPy.constants import AVAILABLE_HAMILTONIANS


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

    ham = AVAILABLE_HAMILTONIANS[options["hamiltonian"]]
    for param_key in ham.param_defn:
        guess = options[param_key + "_guess"]
        if param_key + "_range" in options:
            bounds = bounds_from_range(options, param_key)
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
        Both np array. Can be used to unshuffle shuffled_in_yx, i.e. through
        `QDMPy.hamiltonian._shared.unshuffle_pixels`.
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
        Two arrays returned by `QDMPy.hamiltonian._shared.shuffle_pixels that allow unshuffling of data_2d.

    Returns
    -------
    unshuffled_in_yx: np array, 2D
        data_2d but the inverse operation of `QDMPy.hamiltonian._shared.shuffle_pixels` has been applied
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
        Two arrays returned by `QDMPy.hamiltonian._shared.shuffle_pixels` that allow unshuffling of data_2d.

    Returns
    -------
    fit_result_dict : dict
        Same as input, but each fit parameter has been unshuffled.
    """
    for key, array in fit_result_dict.items():
        fit_result_dict[key] = unshuffle_pixels(array, unshuffler)
    return fit_result_dict


# ============================================================================

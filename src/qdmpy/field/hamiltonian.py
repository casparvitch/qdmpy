# -*- coding: utf-8 -*-
"""
This module is for... TODO

Functions
---------
 - `qdmpy.field.hamiltonian.define_hamiltonian`
 - `qdmpy.field.hamiltonian._prep_fit_backends`
 - `qdmpy.field.hamiltonian.fit_hamiltonian_pixels`
 - `qdmpy.field.hamiltonian.ham_gen_init_guesses`
 - `qdmpy.field.hamiltonian.ham_bounds_from_range`
 - `qdmpy.field.hamiltonian.ham_pixel_generator`
 - `qdmpy.field.hamiltonian.ham_shuffle_pixels`
 - `qdmpy.field.hamiltonian.ham_unshuffle_pixels`
 - `qdmpy.field.hamiltonian.ham_unshuffle_fit_results`
 - `qdmpy.field.hamiltonian.ham_get_pixel_fitting_results`

Classes
-------
 - `qdmpy.field.hamiltonian.Chooser`
 - `qdmpy.field.hamiltonian.Hamiltonian`
 - `qdmpy.field.hamiltonian.ApproxBxyz`
 - `qdmpy.field.hamiltonian.Bxyz`
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.field.hamiltonian.Chooser": True,
    "qdmpy.field.hamiltonian.define_hamiltonian": True,
    "qdmpy.field.hamiltonian._prep_fit_backends": True,
    "qdmpy.field.hamiltonian.fit_hamiltonian_pixels": True,
    "qdmpy.field.hamiltonian.Hamiltonian": True,
    "qdmpy.field.hamiltonian.ApproxBxyz": True,
    "qdmpy.field.hamiltonian.Bxyz": True,
    "qdmpy.field.hamiltonian.ham_gen_init_guesses": True,
    "qdmpy.field.hamiltonian.ham_bounds_from_range": True,
    "qdmpy.field.hamiltonian.ham_pixel_generator": True,
    "qdmpy.field.hamiltonian.ham_shuffle_pixels": True,
    "qdmpy.field.hamiltonian.ham_unshuffle_pixels": True,
    "qdmpy.field.hamiltonian.ham_unshuffle_fit_results": True,
    "qdmpy.field.hamiltonian.ham_get_pixel_fitting_results": True,
}
# ============================================================================
from typing import Dict, List

import numpy as np
import numpy.linalg as LA  # noqa: N812
from collections import OrderedDict
from scipy.linalg import svd
import copy

# ============================================================================

# ============================================================================


S_MAT_X = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
r"""Spin-1 operator: S{\rm X}"""
S_MAT_Y = np.array([[0, -1j, 0], [1j, 0, 1j], [0, 1j, 0]]) / np.sqrt(2)
r"""Spin-1 operator: S{\rm Y}"""
S_MAT_Z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
r"""Spin-1 operator: S{\rm Z}"""


GAMMA = 2.80  # MHz/G
r"""
The Bohr magneton times the Landé g-factor. See [Doherty2013](https://doi.org/10.1016/j.physrep.2013.02.001)
for details of the g-factor anisotropy.

|                                                                  |                                                               |
|------------------------------------------------------------------|---------------------------------------------------------------|
| \( \gamma_{\rm NV} = \mu_{\rm B} g_e  \)                         |                                                               |
| \( \mu_B = 1.39962449361 \times 10^{10}\ {\rm Hz} \rm{T}^{-1} \) |  [NIST](https://physics.nist.gov/cgi-bin/cuu/Value?mubshhz)   |
| \( \mu_B = 1.399...\ {\rm MHz/G} \)                              |                                                               |
| \( g_e \approx 2.0023 \)                                         |  [Doherty2013](https://doi.org/10.1016/j.physrep.2013.02.001) |
| \( \Rightarrow  \gamma_{\rm NV} \approx 2.80 {\rm MHz/G} \)      |                                                               |

"""

# ============================================================================


class Chooser:
    """Chooser class.

    Is fed a boolean 'chooser_ar' on __init__, of length (len(bnvs) or len(freqs)) that
    is used in call to return only the chosen (i.e. True indices in chooser_ar) indices
    of a given array (some_ar in __call__)
    """

    def __init__(self, chooser_ar):
        self.chooser_ar = chooser_ar

    def __call__(self, some_ar):
        return np.array(
            [some_ar[i] for i, do_use in enumerate(self.chooser_ar) if do_use]
        )


# ============================================================================


def define_hamiltonian(options, chooser_obj, unv_frames):
    """
    Return chosen hamiltonian by parsing options.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    chooser_obj : `qdmpy.field.hamiltonian.Chooser`
        Chooser object
    unv_frames : array-like
        NV reference frames in lab frame (see `qdmpy.shared.geom`)

    Returns
    -------
    ham : `qdmpy.field.hamiltonian.Hamiltonian`
        Hamiltonian model object
    """

    ham = AVAILABLE_HAMILTONIANS[options["hamiltonian"]](
        chooser_obj, unv_frames
    )

    options["ham_param_defn"] = ham.get_param_defn()

    _prep_fit_backends(options, ham)

    return ham


# ============================================================================


def _prep_fit_backends(options, ham):
    """
    Prepare all possible fit backends, checking that everything will work.

    Also attempts to import relevant modules into global scope.

    This is a wrapper around specific functions for each backend. All possible fit
    backends are loaded - these are decided in the config file for this system,
    i.e. system.option_choices("fit_backend")

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    ham : `qdmpy.field.hamiltonian.Hamiltonian`
        Model we're fitting to.
    """
    # only scipyfit supported currently
    global fit_scipyfit
    _temp = __import__("qdmpy.field.ham_scipyfit", globals(), locals())
    fit_scipyfit = _temp.field.ham_scipyfit


# ============================================================================


def fit_hamiltonian_pixels(options, data, hamiltonian):
    """
    Fit all pixels in image with chosen fit backend. We're fitting the hamiltonian
    to our previous fit result (i.e. the ODMR/pl fit result).

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    data : np array, 3D
        Normalised measurement array, shape: [sweep_list, y, x]. E.g. bnvs or freqs
    hamiltonian : `qdmpy.field.hamiltonian.Hamiltonian`
        Model we're fitting to.

    Returns
    -------
    ham_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    sigmas : dict
        As ham_results, but containing standard deviations for each parameter across FOV.
    """

    return fit_scipyfit.fit_hamiltonian_scipyfit(options, data, hamiltonian)


# ============================================================================


class Hamiltonian:

    param_defn: List[str] = []
    param_units: Dict[str, str] = {}
    jac_defined = False

    def __init__(self, chooser_obj, unv_frames):
        """
        chooser_obj is used on __call__ and measured_data to return an array
        of only the required parts.
        """
        self.chooser_obj = chooser_obj
        self.unv_frames = unv_frames
        self.unvs = unv_frames[
            :, 2, :
        ].copy()  # i.e. z axis of each nv ref. frame in lab frame

    # =================================

    def __call__(self, param_ar):
        """
        Evaluates Hamiltonian for given parameter values.

        Arguments
        ---------
        param_ar : np array, 1D
            Array of hamiltonian parameters fed in.
        """
        raise NotImplementedError(
            "You MUST override __call__, check your spelling."
        )

    # =================================

    def grad_fn(self, param_ar):
        """
        Return jacobian, shape: (len(bnvs/freqs), len(param_ar))
        Each column is a partial derivative, with respect to each param in param_ar
            (i.e. rows, or first index, is indexing though the bnvs/freqs.)
        """
        raise NotImplementedError("No grad_fn defined for this Hamiltonian.")

    # =================================

    def residuals_scipyfit(self, param_ar, measured_data):
        """
        Evaluates residual: fit model - measured_data. Returns a vector!
        Measured data must be a np array (of the same shape that __call__ returns),
        i.e. freqs, or bnvs.
        """
        return self.chooser_obj(self.__call__(param_ar)) - self.chooser_obj(
            measured_data
        )

    # =================================

    def jacobian_scipyfit(self, param_ar, measured_data):
        """Evaluates (analytic) jacobian of ham in format expected by scipy least_squares."""

        # need to take out rows (first index) according to chooser_obj.
        keep_rows = self.chooser_obj(list(range(len(measured_data))))
        delete_rows = [
            r for r in range(len(measured_data)) if r not in keep_rows
        ]
        return np.delete(self.grad_fn(param_ar), delete_rows, axis=0)

    # =================================

    def jacobian_defined(self):
        return self.jac_defined

    # =================================

    def get_param_defn(self):
        return self.param_defn

    # =================================

    def get_param_odict(self):
        """
        get ordered dict of key: param_key (param_name), val: param_unit for all parameters in ham
        """
        return OrderedDict(self.param_units)

    # =================================

    def get_param_unit(self, param_key):
        """
        Get unit for a given param_key
        """
        if param_key == "residual_field":
            return (
                "Error: sum( || residual(params) || ) over bnvs/freqs (a.u.)"
            )
        param_dict = self.get_param_odict()
        return param_dict[param_key]


# ============================================================================


class ApproxBxyz(Hamiltonian):
    r"""
    Diagonal Hamiltonian approximation. To calculate the magnetic field only fields aligned with
    the NV are considered and thus a simple dot product can be used.

    Fits to bnvs rather than frequencies, i.e.:
    $$ \overline{\overline{B}}_{\rm NV} = overline{\overline{u}}_{\rm NV} \cdot \overline{B} $$
    Where overline denotes qst-order tensor (vector), double overline denotes 2nd-order tensor
    (matrix).
    """

    param_defn: List[str] = ["Bx", "By", "Bz"]
    param_units = {
        "Bx": "Magnetic field, Bx (G)",
        "By": "Magnetic field, By (G)",
        "Bz": "Magnetic field, Bz (G)",
    }
    jac_defined = True

    def __call__(self, param_ar):
        r"""
        $$ \overline{\overline{B}}_{\rm NV} = overline{\overline{u}}_{\rm NV} \cdot \overline{B} $$
        Where overline denotes qst-order tensor (vector), double overline denotes 2nd-order tensor
        (matrix).
        Fit to bnv rather than frequency positions.

        param_ar = [Bx, By, Bz]
        """
        return np.dot(self.unvs, param_ar)

    def grad_fn(self, param_ar):
        return self.unvs
        # J = np.empty((4, 3))  # size: (len(bnvs), len(param_ar)), both known ahead of time.
        # J[:, 0] = self.unvs[:, 0]
        # J[:, 1] = self.unvs[:, 1]
        # J[:, 2] = self.unvs[:, 2]
        # return J
        # i.e. equiv. to:
        # for bnv in range(4):
        #     J[bnv, 0] = self.unvs[bnv][0]  # i.e. ith NV orientation (bnv), x component
        #     J[bnv, 1] = self.unvs[bnv][1]  # y component
        #     J[bnv, 2] = self.unvs[bnv][2]  # z component
        # should be rather obvious from __call__


# ============================================================================


class Bxyz(Hamiltonian):
    r"""
    $$ H_i = D S_{Z_i}^{2} + \gamma_{\rm{NV}} \bf{B} \cdot \bf{S} $$

    where \( {\bf S}_i = (S_{X_i}, S_{Y_i}, S_{Z_i}) \) are the spin-1 operators.
    Here \( (X_i, Y_i, Z_i) \) is the coordinate system of the NV and \( i = 1,2,3,4 \) labels
    each NV orientation with respect to the lab frame.
    """

    param_defn = ["D", "Bx", "By", "Bz"]
    param_units: Dict[str, str] = {
        "D": "Zero field splitting (MHz)",
        "Bx": "Magnetic field, Bx (G)",
        "By": "Magnetic field, By (G)",
        "Bz": "Magnetic field, Bz (G)",
    }
    jac_defined = False

    def __call__(self, param_ar):
        """
        Hamiltonain of the NV spin using only the zero field splitting D and the magnetic field
        bxyz. Takes the fit_params in the order [D, bx, by, bz] and returns the nv frequencies.

        The spin operators need to be rotated to the NV reference frame. This is achieved
        by projecting the magnetic field onto the unv frame.

        i.e. we use the spin operatores in the NV frame, the unvs in the NV frame,
        and thus project the magnetic field into this frame (from the lab frame) to determine
        the nv frequencies (eigenvalues of hamiltonian).
        """
        nv_frequencies = np.zeros(8)
        # D, Bx, By, Bz = param_ar

        Hzero = param_ar[0] * (S_MAT_Z * S_MAT_Z)  # noqa: N806
        for i in range(4):
            bx_proj_onto_unv = np.dot(param_ar[1:4], self.unv_frames[i, 0, :])
            by_proj_onto_unv = np.dot(param_ar[1:4], self.unv_frames[i, 1, :])
            bz_proj_onto_unv = np.dot(param_ar[1:4], self.unv_frames[i, 2, :])

            HB = GAMMA * (  # noqa: N806
                bx_proj_onto_unv * S_MAT_X
                + by_proj_onto_unv * S_MAT_Y
                + bz_proj_onto_unv * S_MAT_Z
            )
            freq, _ = LA.eig(Hzero + HB)
            freq = np.sort(np.real(freq))
            # freqs: ms=0, ms=+-1 -> transition freq is delta
            nv_frequencies[i] = np.real(freq[1] - freq[0])
            nv_frequencies[7 - i] = np.real(freq[2] - freq[0])
        return nv_frequencies

    # this method didn't work :(
    # def grad_fn(self, param_ar):

    # method here: do partial before calculating eigenvalues

    # from qdmpy.constants import S_MAT_X, S_MAT_Y, S_MAT_Z, GAMMA

    # J = np.empty((8, 4))  # shape: num freqs, num params

    # # p equiv. to D, Bx, By, Bz
    # for p in range(4):
    #     if not p:  # sort out D
    #         dH = S_MAT_Z * S_MAT_Z
    #         for i in range(4):
    #             df, _ = LA.eig(dH)
    #             df = np.sort(np.real(df))
    #             J[i, p] = np.real(df[1] - df[0])
    #             J[7 - i, p] = np.real(df[2] - df[0])
    #     else:  # sort out \vec{B}
    #         dparams = np.zeros(4)
    #         dparams[p] = 1  # no polynomials or anything like that so B_comp -> 1, others -> 0
    #         for i in range(4):

    #             dbx_proj_onto_unv = np.dot(dparams[1:4], self.unv_frames[i, 0, :])
    #             dby_proj_onto_unv = np.dot(dparams[1:4], self.unv_frames[i, 1, :])
    #             dbz_proj_onto_unv = np.dot(dparams[1:4], self.unv_frames[i, 2, :])

    #             dH = GAMMA * (
    #                 dbx_proj_onto_unv * S_MAT_X
    #                 + dby_proj_onto_unv * S_MAT_Y
    #                 + dbz_proj_onto_unv * S_MAT_Z
    #             )
    #             df, _ = LA.eig(dH)
    #             df = np.sort(np.real(df))
    #             # freqs: ms=0, ms=+-1 -> transition freq is delta
    #             J[i, p] = np.real(df[1] - df[0])
    #             J[7 - i, p] = np.real(df[2] - df[0])

    # return J


# ============================================================================


def ham_gen_init_guesses(options):
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
    if options["auto_guess_B"]:
        bias_x, bias_y, bias_z = options["bias_field_cartesian_gauss"]
        override_guesses = {"Bx": bias_x, "By": bias_y, "Bz": bias_z}
    else:
        override_guesses = {}

    ham = AVAILABLE_HAMILTONIANS[options["hamiltonian"]]
    for param_key in ham.param_defn:

        if param_key in override_guesses:
            guess = override_guesses[param_key]
        else:
            guess = options[param_key + "_guess"]

        if param_key + "_range" in options:
            bounds = ham_bounds_from_range(options, param_key, guess)
        elif param_key + "_bounds" in options:
            # assumes bounds are passed in with correct formatatting
            bounds = options[param_key + "_bounds"]
        else:
            raise RuntimeError(
                f"Provide bounds for the {ham}.{param_key} param."
            )

        if guess is not None:
            init_guesses[param_key] = guess
            init_bounds[param_key] = np.array(bounds)
        else:
            raise RuntimeError(
                f"Not sure why your guess for {ham}.{param_key} is None?"
            )

    return init_guesses, init_bounds


# ============================================================================


def ham_bounds_from_range(options, param_key, guess):
    """Generate parameter bounds (list, len 2) when given a range option."""
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


def ham_pixel_generator(our_array):
    """
    Simple generator to shape data as expected by to_squares_wrapper in scipy concurrent method.

    Also allows us to track *where* (i.e. which pixel location) each result corresponds to.
    See also: `qdmpy.field.ham_scipyfit.ham_to_squares_wrapper`.

    Arguments
    ---------
    our_array : np array, 3D
        Shape: [idx, y, x] (idx for each bnv, freq etc.)

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


def ham_shuffle_pixels(data_3d):
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
        `qdmpy.field.hamiltonian.ham_unshuffle_pixels`.
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


def ham_unshuffle_pixels(data_2d, unshuffler):
    """
    Simple shuffler

    Arguments
    ---------
    data_2d : np array, 2D
        i.e. 'image' of a single fit parameter, all shuffled up!
    unshuffler : (y_unshuf, x_unshuf)
        Two arrays returned by `qdmpy.field.hamiltonian.ham_shuffle_pixels`
        that allow unshuffling of data_2d.

    Returns
    -------
    unshuffled_in_yx: np array, 2D
        data_2d but the inverse operation of `qdmpy.field.hamiltonian.ham_shuffle_pixels`
        has been applied.
    """
    y_unshuf, x_unshuf = unshuffler
    unshuffled_in_y = data_2d[y_unshuf, :]
    unshuffled_in_yx = unshuffled_in_y[:, x_unshuf]
    return unshuffled_in_yx.copy()


# =================================


def ham_unshuffle_fit_results(fit_result_dict, unshuffler):
    """
    Simple shuffler

    Arguments
    ---------
    fit_result_dict : dict
        Dictionary, key: param_names, val: image (2D) of param values across FOV. Each image
        requires reshuffling (which this function achieves).
        Also has 'residual' as a key.
    unshuffler : (y_unshuf, x_unshuf)
        Two arrays returned by `qdmpy.field.hamiltonian.ham_shuffle_pixels` that allow
        unshuffling of data_2d.

    Returns
    -------
    fit_result_dict : dict
        Same as input, but each fit parameter has been unshuffled.
    """
    for key, array in fit_result_dict.items():
        fit_result_dict[key] = ham_unshuffle_pixels(array, unshuffler)
    return fit_result_dict


# ============================================================================


def ham_get_pixel_fitting_results(hamiltonian, fit_results, pixel_data):
    """
    Take the fit result data from scipyfit/gpufit and back it down to a dictionary of arrays.

    Each array is 2D, representing the values for each parameter (specified by the dict key).


    Arguments
    ---------
    fit_model : `qdmpy.field.hamiltonian.Hamiltonian`
        Model we're fitting to.
    fit_results : list of [(y, x), result, jac] objects
        (see `qdmpy.field.ham_scipyfit.ham_to_squares_wrapper`)
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
    fit_image_results = hamiltonian.get_param_odict()
    sigmas = copy.copy(fit_image_results)

    # override with correct size empty arrays using np.zeros
    for key in fit_image_results.keys():
        fit_image_results[key] = (
            np.zeros((roi_shape[0], roi_shape[1])) * np.nan
        )
        sigmas[key] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan

    fit_image_results["residual_field"] = (
        np.zeros((roi_shape[0], roi_shape[1])) * np.nan
    )

    # Fill the arrays element-wise from the results function, which returns a
    # 1D array of flattened best-fit parameters.
    for (y, x), result, jac in fit_results:
        resid = hamiltonian.residuals_scipyfit(result, pixel_data[:, y, x])
        fit_image_results["residual_field"][y, x] = np.sum(
            np.abs(resid, dtype=np.float64), dtype=np.float64
        )
        # uncertainty (covariance matrix), copied from scipy.optimize.curve_fit
        _, s, vt = svd(jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(jac.shape) * s[0]
        s = s[s > threshold]
        vt = vt[: s.size]
        pcov = np.dot(vt.T / s**2, vt)
        perr = np.sqrt(np.diag(pcov))  # array of standard deviations

        for param_num, param_name in enumerate(hamiltonian.param_defn):
            fit_image_results[param_name][y, x] = result[param_num]
            sigmas[param_name][y, x] = perr[param_num]

    return fit_image_results, sigmas


# ============================================================================
# ============================================================================


AVAILABLE_HAMILTONIANS = {
    "approx_bxyz": ApproxBxyz,
    "bxyz": Bxyz,
}
"""Dictionary that defines hamiltonians available for use.

Add any classes you define here so you can use them.

You do not need to avoid overlapping parameter names as hamiltonian
classes can not be used in combination.
"""

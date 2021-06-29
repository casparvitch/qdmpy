# -*- coding: utf-8 -*-
"""
This module holds tools for pl input/output

Functions
---------
 - `qdmpy.pl.io.load_image_and_sweep`
 - `qdmpy.pl.io.reshape_dataset`
 - `qdmpy.pl.io.save_pl_data`
 - `qdmpy.pl.io._rebin_image`
 - `qdmpy.pl.io._remove_unwanted_data`
 - `qdmpy.pl.io._check_start_end_rectangle`
 - `qdmpy.pl.io.load_prev_pl_fit_results`
 - `qdmpy.pl.io.load_prev_pl_fit_sigmas`
 - `qdmpy.pl.io.load_pl_fit_sigma`
 - `qdmpy.pl.io.load_fit_param`
 - `qdmpy.pl.io.save_pl_fit_sigmas`
 - `qdmpy.pl.io.save_pl_fit_results`
 - `qdmpy.pl.io.load_ref_exp_pl_fit_results`
 - `qdmpy.pl.io._check_if_already_fit`
 - `qdmpy.pl.io._prev_options_exist`
 - `qdmpy.pl.io._options_compatible`
 - `qdmpy.pl.io._prev_pl_fits_exist`
 - `qdmpy.pl.io._prev_pl_sigmas_exist`
 - `qdmpy.pl.io._get_prev_options`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.pl.io.load_image_and_sweep": True,
    "qdmpy.pl.io.reshape_dataset": True,
    "qdmpy.pl.io.save_pl_data": True,
    "qdmpy.pl.io._rebin_image": True,
    "qdmpy.pl.io._remove_unwanted_data": True,
    "qdmpy.pl.io._check_start_end_rectangle": True,
    "qdmpy.pl.io.load_prev_pl_fit_results": True,
    "qdmpy.pl.io.load_prev_pl_fit_sigmas": True,
    "qdmpy.pl.io.load_pl_fit_sigma": True,
    "qdmpy.pl.io.load_fit_param": True,
    "qdmpy.pl.io.save_pl_fit_sigmas": True,
    "qdmpy.pl.io.save_pl_fit_results": True,
    "qdmpy.pl.io.load_ref_exp_pl_fit_results": True,
    "qdmpy.pl.io._check_if_already_fit": True,
    "qdmpy.pl.io._prev_options_exist": True,
    "qdmpy.pl.io._options_compatible": True,
    "qdmpy.pl.io._prev_pl_fits_exist": True,
    "qdmpy.pl.io._prev_pl_sigmas_exist": True,
    "qdmpy.pl.io._get_prev_options": True,
}

# ============================================================================

import numpy as np
import warnings
import pathlib
from rebin import rebin

# ============================================================================

import qdmpy.shared.misc
import qdmpy.pl.funcs
import qdmpy.pl.model

# ============================================================================


def load_image_and_sweep(options):
    """
    Reads raw image data and sweep_list (affine parameters) using system methods

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    image : np array, 3D
        Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has
        not been rebinned. Unwanted sweep values not removed.
    sweep_list : list
        List of sweep parameter values
    """
    image = options["system"].read_image(options["filepath"], options)
    sweep_list = options["system"].read_sweep_list(options["filepath"])
    return image, sweep_list


# ============================================================================


def reshape_dataset(options, image, sweep_list):
    """
    Reshapes and re-bins raw data into more useful format.

    Cuts down to ROI and removes unwanted sweeps.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    image : np array, 3D
        Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has
        not been rebinned. Unwanted sweep values not removed.
    sweep_list : list
        List of sweep parameter values

    Returns
    -------
    pl_image : np array, 2D.
        Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and
        cut down to ROI.
    pl_image_ROI : np array, 2D
        Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and
        cut down to ROI.
    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, y, x]
    ref : np array, 3D
        Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, y, x]
    sig_norm : np array, 3D
        Signal normalised by reference (via subtraction or normalisation, chosen in options),
        reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI.
        Format: [sweep_vals, y, x]
    single_pixel_pl : np array, 1D
        Normalised measurement array, for chosen single pixel check.
    sweep_list : list
        List of sweep parameter values (with removed unwanted sweeps at start/end)
    roi : length 2 list of np meshgrids
        Defines an ROI that can be applied to the 3D image through direct indexing.
        E.g. sig_ROI = sig[:, roi[0], roi[1]] (post rebinning)
    """

    # image = reshape_raw(options, raw_data, sweep_list)
    image_rebinned, sig, ref, sig_norm = _rebin_image(options, image)
    try:
        size_h, size_w = image_rebinned.shape[1:]
    except Exception:
        size_h, size_w = image_rebinned.shape

    options["rebinned_image_shape"] = (size_h, size_w)

    # check options to ensure ROI is in correct format (now we have image size)
    if options["ROI"] != "Full":
        options["ROI_start"], options["ROI_end"] = _check_start_end_rectangle(
            "ROI", *options["ROI_start"], *options["ROI_end"], size_w, size_h
        )  # opposite convention here, [x, y]

    # somewhat important a lot of this isn't hidden, so we can adjust it later
    pl_image, pl_image_roi, sig, ref, sig_norm, sweep_list, roi = _remove_unwanted_data(
        options, image_rebinned, sweep_list, sig, ref, sig_norm
    )  # also cuts sig etc. down to ROI

    # check options to ensure AOI is in correct format (now we have ROI size)
    i = 0
    while True:
        i += 1
        try:
            if options[f"AOI_{i}_start"] is None or options[f"AOI_{i}_end"] is None:
                continue
            options[f"AOI_{i}_start"], options[f"AOI_{i}_end"] = _check_start_end_rectangle(
                f"AOI_{i}",
                *options[f"AOI_{i}_start"],
                *options[f"AOI_{i}_end"],
                *reversed(sig_norm.shape[1:]),  # opposite convention here, [x, y]
            )

        except KeyError:
            break

    # single pixel check
    try:
        single_pixel_pl = sig_norm[
            :, options["single_pixel_check"][1], options["single_pixel_check"][0]
        ]
    except IndexError as e:
        warnings.warn(
            f"Avoiding IndexError for single_pixel_check (setting pixel check to centre of image):\n{e}"
        )
        single_pixel_pl = sig_norm[:, sig_norm.shape[1] // 2, sig_norm.shape[2] // 2]
        options["single_pixel_check"] = (sig_norm.shape[2] // 2, sig_norm.shape[1] // 2)

    return pl_image, pl_image_roi, sig, ref, sig_norm, single_pixel_pl, sweep_list, roi


# ============================================================================


def save_pl_data(options, pl_image, pl_image_roi):
    """Saves pl_image and pl_image_ROI to disk"""
    np.savetxt(options["data_dir"] / "pl_full_rebinned.txt", pl_image)
    np.savetxt(options["data_dir"] / "pl_ROI_rebinned.txt", pl_image_roi)


# ============================================================================


def _rebin_image(options, image):
    """
    Reshapes raw data into more useful shape, according to image size in metadata.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    image : np array, 3D
        Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has
        not been rebinned.

    Returns
    -------
    image_rebinned : np array, 3D
        Format: [sweep values, y, x]. Same as image, but now rebinned (x size and y size
        have changed). Not cut down to ROI.
    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet.
        Format: [sweep_vals, y, x]. Not cut down to ROI.
    ref : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet.
        Not cut down to ROI. Format: [sweep_vals, y, x].
    sig_norm : np array, 3D
        Signal normalised by reference (via subtraction or normalisation, chosen in options),
        reshaped and rebinned.  Unwanted sweeps not removed yet.
        Not cut down to ROI. Format: [sweep_vals, y, x].
    """
    if not options["additional_bins"]:
        options["additional_bins"] = 1
        image_rebinned = image
    else:
        if options["additional_bins"] != 1 and options["additional_bins"] % 2:
            raise RuntimeError("The binning parameter needs to be a multiple of 2.")

        # data_pts = image.shape[0]
        # height = image.shape[1]
        # width = image.shape[2]
        # image_rebinned = (
        #     np.reshape(
        #         image,
        #         [
        #             data_pts,
        #             int(height / options["additional_bins"]),
        #             options["additional_bins"],
        #             int(width / options["additional_bins"]),
        #             options["additional_bins"],
        #         ],
        #     )
        #     .sum(2)
        #     .sum(3)
        # ) # this is old version... moving to rebin package
        image_rebinned = rebin(
            image, factor=(1, options["additional_bins"], options["additional_bins"]), func=np.mean
        )

    # define sig and ref differently if we're using a ref
    if options["used_ref"]:
        sig = image_rebinned[::2, :, :]
        ref = image_rebinned[1::2, :, :]
        if options["normalisation"] == "sub":
            sig_norm = 1 + (sig - ref) / (sig + ref)
        elif options["normalisation"] == "div":
            sig_norm = sig / ref
        else:
            raise KeyError("bad normalisation option, use: ['sub', 'div']")
    else:
        sig = image_rebinned
        ref = image_rebinned
        sig_norm = sig / np.max(sig, 0)

    return image_rebinned, sig, ref, sig_norm


# ============================================================================


def _remove_unwanted_data(options, image_rebinned, sweep_list, sig, ref, sig_norm):
    """
    Removes unwanted sweep values (i.e. freq values or tau values) for all of the data arrays.

    Also cuts data down to ROI.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    image_rebinned : np array, 3D
        Format: [sweep values, y, x]. Same as image, but rebinned (x size and y size
        have changed).
    sweep_list : list
        List of sweep parameter values
    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet.
        Format: [sweep_vals, y, x]
    ref : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet.
        Format: [sweep_vals, y, x]
    sig_norm : np array, 3D
        Signal normalised by reference (via subtraction or normalisation, chosen in options),
        reshaped and rebinned.  Unwanted sweeps not removed yet.
        Format: [sweep_vals, y, x]

    Returns
    -------
    pl_image : np array, 2D
        Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and
        cut down to ROI
    pl_image_ROI : np array, 2D
        Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped and rebinned as
        well as cut down to ROI.
    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Format: [sweep_vals, y, x]
    ref : np array, 3D
        Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Format: [sweep_vals, y, x]
    sig_norm : np array, 3D
        Signal normalised by reference (via subtraction or normalisation, chosen in options),
        reshaped and rebinned. Unwanted sweeps removed.
        Format: [sweep_vals, y, x]
    sweep_list : list
        List of sweep parameter values (with removed unwanted sweeps at start/end)
    roi : length 2 list of np meshgrids
        Defines an ROI that can be applied to the 3D image through direct indexing.
        E.g. sig_ROI = sig[:, roi[0], roi[1]]
    """
    rem_start = options["remove_start_sweep"]
    rem_end = options["remove_end_sweep"]

    roi = qdmpy.shared.misc.define_ROI(options, *options["rebinned_image_shape"])

    if rem_start < 0:
        warnings.warn("remove_start_sweep must be >=0, setting to zero now.")
        rem_start = 0
    if rem_end < 0:
        warnings.warn("remove_end_sweep must be >=0, setting to zero now.")
        rem_end = 0

    pl_image = np.sum(image_rebinned, axis=0)

    pl_image_roi = pl_image[roi[0], roi[1]].copy()
    sig = sig[rem_start : -1 - rem_end, roi[0], roi[1]].copy()  # noqa: E203
    ref = ref[rem_start : -1 - rem_end, roi[0], roi[1]].copy()  # noqa: E203
    sig_norm = sig_norm[rem_start : -1 - rem_end, roi[0], roi[1]].copy()  # noqa: E203
    sweep_list = np.asarray(sweep_list[rem_start : -1 - rem_end]).copy()  # noqa: E203

    return pl_image, pl_image_roi, sig, ref, sig_norm, sweep_list, roi


# ============================================================================


def _check_start_end_rectangle(name, start_x, start_y, end_x, end_y, full_size_w, full_size_h):
    """
    Checks that 'name' rectange (defined by top left corner 'start_x', 'start_y' and bottom
    right corner 'end_x', 'end_y') fits within a larger rectangle of size 'full_size_w',
    'full_size_h'. If there are no good, they're fixed with a warning.

    Arguments
    ---------
    name : str
        The name of the rectangle we're checking (e.g. "ROI", "AOI").
    start_x : int
        x coordinate (relative to origin) of rectangle's top left corner.
    start_y : int
        y coordinate (relative to origin) of rectangle's top left corner.
    end_x : int
        x coordinate (relative to origin) of rectangle's bottom right corner.
    end_y : int
        y coordinate (relative to origin) of rectangle's bottom right corner.
    full_size_w : int
        Full width of image (or image region, e.g. ROI).
    full_size_h : int
        Full height of image (or image region, e.g. ROI).

    Returns
    -------
    start_coords : tuple
        'fixed' start coords: (start_x, start_y)
    end_coords : tuple
        'fixed' end coords: (end_x, end_y)
    """

    if start_x >= end_x:
        warnings.warn(f"{name} Rectangle ends before it starts (in x), swapping them")
        temp = start_x
        start_x = end_x
        end_x = temp
    if start_y >= end_y:
        warnings.warn(f"{name} Rectangle ends before it starts (in y), swapping them")
        temp = start_x
        start_x = end_y
        end_x = temp

    if start_x >= full_size_w:
        warnings.warn(f"{name} Rectangle starts outside image (too large in x), setting to zero.")
        start_x = 0
    elif start_x < 0:
        warnings.warn(f"{name} Rectangle starts outside image (negative in x), setting to zero..")
        start_x = 0

    if start_y >= full_size_h:
        warnings.warn(f"{name} Rectangle starts outside image (too large in y), setting to zero.")
        start_y = 0
    elif start_y < 0:
        warnings.warn(f"{name}  Rectangle starts outside image (negative in y), setting to zero.")
        start_y = 0

    if end_x >= full_size_w:
        warnings.warn(f"{name} Rectangle too big in x, cropping to image.")
        end_x = full_size_w - 1
    if end_y >= full_size_h:
        warnings.warn(f"{name} Rectangle too big in y, cropping to image.")
        end_y = full_size_h - 1

    return (start_x, start_y), (end_x, end_y)


# ============================================================================


def load_prev_pl_fit_results(options):
    """Load (all) parameter pl fits from previous processing."""

    prev_options = qdmpy.io.fit._get_prev_options(options)

    fit_param_res_dict = {}

    for fn_type, num in prev_options["fit_functions"].items():
        for param_name in qdmpy.pl.funcs._AVAILABLE_FNS[fn_type].param_defn:
            for n in range(num):
                param_key = param_name + "_" + str(n)
                fit_param_res_dict[param_key] = load_fit_param(options, param_key)
    fit_param_res_dict["residual_0"] = load_fit_param(options, "residual_0")
    return fit_param_res_dict


# ============================================================================


def load_prev_pl_fit_sigmas(options):
    """Load (all) parameter pl fits from previous processing."""

    prev_options = qdmpy.io.fit._get_prev_options(options)

    sigmas = {}

    for fn_type, num in prev_options["fit_functions"].items():
        for param_name in qdmpy.pl.funcs._AVAILABLE_FNS[fn_type].param_defn:
            for n in range(num):
                key = param_name + "_" + str(n)
                sigmas[key] = load_pl_fit_sigma(options, key)
    return sigmas


# ============================================================================


def load_pl_fit_sigma(options, key):
    """ Load a previous fit sigma result, of name 'param_key' """
    return np.loadtxt(options["data_dir"] / (key + "_sigma.txt"))


# ============================================================================


def load_fit_param(options, param_key):
    """Load a previously fit param, of name 'param_key'."""
    return np.loadtxt(options["data_dir"] / (param_key + ".txt"))


# ============================================================================


def save_pl_fit_sigmas(options, sigmas):
    """
    Saves pixel fit sigmas to disk.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    sigmas : OrderedDict
        Dictionary, key: param_keys, val: image (2D) of param sigmas across FOV.
    """
    if sigmas is not None:
        for key, result in sigmas.items():
            np.savetxt(options["data_dir"] / f"{key}_sigma.txt", result)


# ============================================================================


def save_pl_fit_results(options, pixel_fit_params):
    """
    Saves pl fit results to disk.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    pixel_fit_params : OrderedDict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    """
    if pixel_fit_params is not None:
        for param_key, result in pixel_fit_params.items():
            np.savetxt(options["data_dir"] / f"{param_key}.txt", result)


# ============================================================================


def load_ref_exp_pl_fit_results(ref_options):
    """ref_options dict -> pixel_fit_params dict.

    Provide one of ref_options and ref_options_dir. If both are None, returns None (with a
    warning). If both are supplied, ref_options takes precedence.

    Arguments
    ---------
    ref_options : dict, default=None
        Generic options dict holding all the user options (for the reference experiment).

    Returns
    -------
    fit_result_dict : OrderedDict
        Dictionary, key: param_keys, val: image (2D) of (fit) param values across FOV.
        If no reference experiment is given (i.e. ref_options and ref_options_dir are None) then
        returns None
    """

    # ok now have ref_options dict, time to load params
    if ref_options["found_prev_result"]:
        ref_fit_result_dict = load_prev_pl_fit_results(ref_options)
        ref_sigmas = load_prev_pl_fit_sigmas(ref_options)
        return ref_fit_result_dict, ref_sigmas
    else:
        warnings.warn(
            "Didn't find reference experiment fit results? Reason: "
            + ref_options["found_prev_result_reason"]
        )
        return None, None


# ============================================================================


def _check_if_already_fit(options, loading_ref=False):
    """
    Looks for previous fit result.

    If previous fit result exists, checks for compatibility between option choices.

    loading_ref (bool): skip checks for force_fit etc. and just see if prev pixel results exist.

    Returns nothing.
    """
    if not loading_ref:
        if not options["force_fit"]:
            if not _prev_options_exist(options):  # i.e. look for saved options in output dir
                options["found_prev_result_reason"] = "couldn't find previous options"
                options["found_prev_result"] = False
            elif not (res := _options_compatible(options, _get_prev_options(options)))[0]:
                options["found_prev_result_reason"] = "option not compatible: " + res[1]
                options["found_prev_result"] = False
            elif not (res2 := _prev_pl_fits_exist(options))[0]:
                options["found_prev_result_reason"] = (
                    "couldn't find prev pixel results: " + res2[1]
                )
                options["found_prev_result"] = False
            else:
                options["found_prev_result_reason"] = "found prev result :)"
                options["found_prev_result"] = True
        else:
            options["found_prev_result_reason"] = "option 'force_fit' was True"
            options["found_prev_result"] = False
    elif not (res3 := _prev_pl_fits_exist(options))[0]:
        options["found_prev_result_reason"] = "couldn't find prev pixel results: " + res3[1]
        options["found_prev_result"] = False
    else:
        options["found_prev_result_reason"] = "found prev result :)"
        options["found_prev_result"] = True


# ============================================================================


def _prev_options_exist(options):
    """
    Checks if options file from previous result can be found in default location, returns Bool.
    """
    prev_opt_path = pathlib.Path(options["output_dir"]) / "saved_options.json"
    return prev_opt_path.is_file()


# ============================================================================


def _options_compatible(options, prev_options):
    """
    Checks if option choices are compatible with previously fit options

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    prev_options : dict
        Generic options dict from previous fit result.

    Returns
    -------
    _options_compatible : bool
        Whether or not options are compatible.

    reason : str
        Reason for the above
    """
    if (
        "found_prev_result" in options
        and options["found_prev_result"] is not None
        and not options["found_prev_result"]
    ):
        return False, "already checked for previous fit result and didn't find anything."

    if not (
        options["additional_bins"] == prev_options["additional_bins"]
        or (options["additional_bins"] in [0, 1] and prev_options["additional_bins"] in [0, 1])
    ):
        return False, "different binning"
    for option_name in [
        "normalisation",
        "fit_backend",
        "fit_functions",
        "ROI",
        "ignore_ref",
        "system_name",
        "remove_start_sweep",
        "remove_end_sweep",
        "use_ROI_avg_fit_res_for_all_pixels",
    ]:
        if options[option_name] != prev_options[option_name]:
            return False, f"different option: {option_name}"
    # check relevant ROI params
    if options["ROI"] == "Rectangle" and (
        options["ROI_start"] != prev_options["ROI_start"]
        and options["ROI_end"] != prev_options["ROI_end"]
    ):
        return False, "different ROI rectangle bounds"

    # check relevant param guesses/bounds

    # check relevant fit params
    if options["fit_backend"] == "scipyfit":
        for fit_opt_name in [
            "scipyfit_method",
            "scipyfit_use_analytic_jac",
            "scipyfit_fit_jac_acc",
            "scipyfit_fit_gtol",
            "scipyfit_fit_xtol",
            "scipyfit_fit_ftol",
            "scipyfit_scale_x",
            "scipyfit_loss_fn",
        ]:
            if (
                fit_opt_name not in options
                or fit_opt_name not in prev_options
                or options[fit_opt_name] != prev_options[fit_opt_name]
            ):
                return False, f"scipyfit option different: {fit_opt_name}"
    elif options["fit_backend"] == "gpufit":
        for fit_opt_name in ["gpufit_tolerance", "gpufit_max_iterations", "gpufit_estimator_id"]:
            if options[fit_opt_name] != prev_options[fit_opt_name]:
                return False, f"gpufit option different: {fit_opt_name}"

    # ok now the trickiest one, check parameter guesses & bounds
    unique_params = set(qdmpy.pl.model.FitModel(options["fit_functions"])).get_param_defn()

    for param_name in unique_params:
        if options[param_name + "_guess"] != prev_options[param_name + "_guess"]:
            return False, f"param guess different: {param_name}"

        range_opt = param_name + "_range"
        if range_opt in options and range_opt in prev_options:
            if options[range_opt] != prev_options[range_opt]:
                return False, f"different range options: {param_name}"
            else:
                continue  # this param all g, check others
        # ok range takes precedence over bounds
        if range_opt in options and range_opt not in prev_options:
            return False, f"different range/bound options: {param_name}"
        if range_opt not in options and range_opt in prev_options:
            return False, f"different range/bound options: {param_name}"
        # finally check bounds
        if options[param_name + "_bounds"] != prev_options[param_name + "_bounds"]:
            return False, f"param range different: {param_name}"

    # if all that was ok, return True
    return True, "all g"


# ============================================================================


def _prev_pl_fits_exist(options):
    """Check if the actual fit result files exists.

    Arguments
    ---------
    options : dict
        Generic options dict from (either prev. or current, should be the equiv.) fit result.

    Returns
    -------
    pixels_results_exist : bool
        Whether or not previous pixel result files exist.

    reason : str
        Reason for the above
    """

    for fn_type, num in options["fit_functions"].items():
        for param_name in qdmpy.pl.funcs._AVAILABLE_FNS[fn_type].param_defn:
            for n in range(num):
                param_key = param_name + "_" + str(n)
                if not (pathlib.Path(options["data_dir"]) / (param_key + ".txt")).is_file():
                    return False, f"couldn't find previous param: {param_key}"

    if not (pathlib.Path(options["data_dir"]) / "residual_0.txt").is_file():
        return False, "couldn't find previous residual"
    return True, "found all prev pixel results :)"


# ============================================================================


def _prev_pl_sigmas_exist(prev_options):
    """ as `qdmpy.io.raw._prev_pl_fits_exist` but for sigmas """
    # avoid cyclic imports

    for fn_type, num in prev_options["fit_functions"].items():
        for param_name in qdmpy.pl.funcs._AVAILABLE_FNS[fn_type].param_defn:
            for n in range(num):
                param_key = param_name + "_" + str(n)
                if not (
                    pathlib.Path(prev_options["data_dir"]) / (param_key + "_sigma.txt")
                ).is_file():
                    return False

    return True


# ============================================================================


def _get_prev_options(options):
    """
    Reads options file from previous fit result (.json), returns a dictionary.
    """
    return qdmpy.json2dict.json_to_dict(options["output_dir"] / "saved_options.json")


# ============================================================================

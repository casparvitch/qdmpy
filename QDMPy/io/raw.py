# -*- coding: utf-8 -*-
"""
This module holds tools for loading raw data etc. and reshaping to a usable format.

Also contained here are functions for defining regions/areas of interest within the
larger image dataset, that are then used in consequent functions, as well as the
general options dictionary.

Functions
---------
 - `QDMPy.io.raw.load_options`
 - `QDMPy.io.raw.save_options`
 - `QDMPy.io.raw.load_image_and_sweep`
 - `QDMPy.io.raw.reshape_dataset`
 - `QDMPy.io.raw.save_PL_data`
  - QDMPy.io.raw._define_output_dir
 - `QDMPy.io.raw._interpolate_option_str`
 - `QDMPy.io.raw._rebin_image`
 - `QDMPy.io.raw._remove_unwanted_data`
 - `QDMPy.io.raw._define_ROI`
 - `QDMPy.io.raw._define_area_roi`
 - `QDMPy.io.raw.define_AOIs`
 - `QDMPy.io.raw._recursive_dict_update`
 - `QDMPy.io.raw._check_start_end_rectangle`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.io.raw.load_options": True,
    "QDMPy.io.raw.save_options": True,
    "QDMPy.io.raw.load_image_and_sweep": True,
    "QDMPy.io.raw.reshape_dataset": True,
    "QDMPy.io.raw.save_PL_data": True,
    "QDMPy.io.raw._define_output_dir": True,
    "QDMPy.io.raw._interpolate_option_str": True,
    "QDMPy.io.raw._rebin_image": True,
    "QDMPy.io.raw._remove_unwanted_data": True,
    "QDMPy.io.raw._define_ROI": True,
    "QDMPy.io.raw._define_area_roi": True,
    "QDMPy.io.raw.define_AOIs": True,
    "QDMPy.io.raw._recursive_dict_update": True,
    "QDMPy.io.raw._check_start_end_rectangle": True,
}

# ============================================================================

import numpy as np
import warnings
import os
import pathlib
from collections import OrderedDict  # insertion order is guaranteed for py3.7+, but to be safe!
import collections.abc
import re

# ============================================================================

import QDMPy.io.json2dict
import QDMPy.io.fit
import QDMPy.systems as systems

# ============================================================================


def load_options(
    options_dict=None, options_path=None, check_for_prev_result=False, loading_ref=False
):
    """
    Load and process options (from json file or dict) into generic options dict used everywhere.

    Also handles directory creation etc. to put results in.
    Provide either options_dict OR options_path (must provide one!).

    Note the system default options are loaded in, so you only need to set the things you need.
    In particular, filepath, fit_functions and system_name must be set

    Optional Arguments
    -------------------
    options_dict : dict
        Directly pass in a dictionary of options.
        Default: None

    path : string
        Path to fit options .json file. Can be absolute, or from QDMPy.
        Default: None

    check_for_prev_result : bool
        Check to see if there's a previous fit result for these options.
        Default: False

    loading_ref : bool
        Reloading reference fit result, so ensure we check for previous fit result.
        Passed on to _check_if_already_fit.

    Returns
    -------
    options : dict
        Generic options dict holding all the user options.
    """

    # pass options_dict and/or options_path
    if options_dict is not None and options_path is not None:
        # options_dict takes precedence
        options_path = None
    if options_dict is None and options_path is None:
        raise RuntimeError("pass at least one of options_dict and options_path to load_options")

    if options_path is not None:
        if not os.path.isfile(options_path):
            raise ValueError("options file at `options_path` not found?")
        prelim_options = QDMPy.io.json2dict.json_to_dict(options_path)
    else:
        prelim_options = OrderedDict(options_dict)  # unnescessary py3.7+, leave to describe intent

    required_options = ["filepath", "fit_functions"]
    for key in required_options:
        if key not in prelim_options:
            raise RuntimeError(f"Must provide these options: {required_options}")

    from QDMPy.constants import choose_system  # avoid cyclic imports

    chosen_system = choose_system(prelim_options["system_name"])

    chosen_system.system_specific_option_update(prelim_options)

    options = chosen_system.get_default_options()  # first load in default options
    # now update with what has been decided upon by user
    options = _recursive_dict_update(options, prelim_options)

    chosen_system.determine_binning(options)

    chosen_system.system_specific_option_update(
        options
    )  # do this again on full options to be sure

    options["system"] = chosen_system

    systems.clean_options(options)  # check all the options make sense

    # create output directories
    _define_output_dir(options)

    if not os.path.isdir(options["output_dir"]):
        os.mkdir(options["output_dir"])
    if not os.path.isdir(options["data_dir"]):
        os.mkdir(options["data_dir"])

    # don't always check for prev. results (so we can use this fn in other contexts)
    if check_for_prev_result or loading_ref:
        QDMPy.io.fit._check_if_already_fit(options, loading_ref=loading_ref)
    return options


# ============================================================================


def save_options(options):
    """
    Saves generic options dict to harddrive as json file (in options["output_dir"])

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    """

    keys_to_remove = ["system"]
    save_options = {}

    for key, val in options.items():
        if key.endswith("dir") or key == "filepath":
            val = str(val).replace("\\", "\\\\")
        if key not in keys_to_remove:
            save_options[key] = val
    QDMPy.io.json2dict.dict_to_json(
        save_options, "saved_options.json", path_to_dir=options["output_dir"]
    )


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
    systems.clean_options(options)

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
    PL_image : np array, 2D.
        Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and
        cut down to ROI.

    PL_image_ROI : np array, 2D
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
    """
    systems.clean_options(options)

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
    PL_image, PL_image_ROI, sig, ref, sig_norm, sweep_list = _remove_unwanted_data(
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

    return PL_image, PL_image_ROI, sig, ref, sig_norm, single_pixel_pl, sweep_list


# ============================================================================


def save_PL_data(options, PL_image, PL_image_ROI):
    """Saves PL_image and PL_image_ROI to disk"""
    np.savetxt(options["data_dir"] / "PL_full_rebinned.txt", PL_image)
    np.savetxt(options["data_dir"] / "PL_ROI_rebinned.txt", PL_image_ROI)


# ============================================================================


def _define_output_dir(options):
    """
    Defines output_dir and data_dir in options.
    """
    if options["custom_output_dir"] is not None:
        output_dir = pathlib.PurePosixPath(
            _interpolate_option_str(str(options["custom_output_dir"]), options)
        )
    else:
        output_dir = pathlib.PurePosixPath(str(options["filepath"]))

    if options["custom_output_dir_prefix"] is not None:
        prefix = _interpolate_option_str(str(options["custom_output_dir_prefix"]), options)
    else:
        prefix = ""

    if options["custom_output_dir_suffix"] is not None:
        suffix = _interpolate_option_str(str(options["custom_output_dir_suffix"]), options)
    else:
        suffix = ""

    options["output_dir"] = output_dir.parent.joinpath(prefix + output_dir.stem + suffix)
    options["data_dir"] = options["output_dir"].joinpath("data")


# ============================================================================


def _interpolate_option_str(interp_str, options):
    """
    Interpolates any options between braces in interp_str.
    I.e. "{fit_backend}" -> f"{options['fit_backend']"
    (this is possibly possible directly through f-strings but I didn't want to
    play with fire)

    Arguments
    ---------
    interp_str : str
        String (possibly containing option names between braces) to be interpolated.

    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    interp_str : str
        String, now with interpolated values (option between braces).
    """

    # convert whitespace to underscore
    interp_str = interp_str.replace(" ", "_")
    if not ("{" in interp_str and "}" in interp_str):
        return interp_str
    pattern = r"\{(\w+)\}"  # all text between braces
    match_substr_lst = re.findall(pattern, interp_str)
    locs = []
    for match_substr in match_substr_lst:
        start_loc = interp_str.find(match_substr)
        end_loc = start_loc + len(match_substr) - 1  # both fence posts inclusive
        locs.append((start_loc, end_loc))

    # ok so now we have the option names (keys) to interpolate (match_substr_lst)
    # as well as the locations in the string of the same (locs)
    # first we convert those option names to their variables
    option_lst = []
    for option_name in match_substr_lst:
        try:
            option_lst.append(str(options[option_name]))
        except KeyError as e:
            warnings.warn(
                "\n"
                + "KeyError caught interpolating custom output_dir.\n"
                + f"You gave: {option_name}.\n"
                + "Avoiding this issue by placing 'option_name' in the dir instead. KeyError msg:"
                + f"{e}"
            )
            option_lst.append(option_name)

    # this block is a bit old-school, but does the trick...
    locs_passed = 0  # how many match locations we've passed
    new_str = []  # ok actually a lst but will convert back at end

    for i, c in enumerate(interp_str):
        # don't want braces in output
        if c == "{" or c == "}":
            continue

        #  'inside' a brace -> don't copy these chars
        if i >= locs[locs_passed][0] and i <= locs[locs_passed][1]:
            # when we've got to end of this brace location, copy in option
            if i == locs[locs_passed][1]:
                new_str.append(option_lst[locs_passed])
                locs_passed += 1
        else:
            new_str.append(c)

    return "".join(new_str)


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
    systems.clean_options(options)

    if not options["additional_bins"]:
        options["additional_bins"] = 1
        image_rebinned = image
    else:
        if options["additional_bins"] != 1 and options["additional_bins"] % 2:
            raise RuntimeError("The binning parameter needs to be a multiple of 2.")

        data_pts = image.shape[0]
        height = image.shape[1]
        width = image.shape[2]
        image_rebinned = (
            np.reshape(
                image,
                [
                    data_pts,
                    int(height / options["additional_bins"]),
                    options["additional_bins"],
                    int(width / options["additional_bins"]),
                    options["additional_bins"],
                ],
            )
            .sum(2)
            .sum(3)
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

        sig = ref = image_rebinned
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
    PL_image : np array, 2D
        Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and
        cut down to ROI

    PL_image_ROI : np array, 2D
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
    """
    systems.clean_options(options)

    rem_start = options["remove_start_sweep"]
    rem_end = options["remove_end_sweep"]

    ROI = _define_ROI(options, *options["rebinned_image_shape"])

    if rem_start < 0:
        warnings.warn("remove_start_sweep must be >=0, setting to zero now.")
        rem_start = 0
    if rem_end < 0:
        warnings.warn("remove_end_sweep must be >=0, setting to zero now.")
        rem_end = 0

    PL_image = np.sum(image_rebinned, axis=0)

    PL_image_ROI = PL_image[ROI[0], ROI[1]].copy()
    sig = sig[rem_start : -1 - rem_end, ROI[0], ROI[1]].copy()  # noqa: E203
    ref = ref[rem_start : -1 - rem_end, ROI[0], ROI[1]].copy()  # noqa: E203
    sig_norm = sig_norm[rem_start : -1 - rem_end, ROI[0], ROI[1]].copy()  # noqa: E203
    sweep_list = np.asarray(sweep_list[rem_start : -1 - rem_end]).copy()  # noqa: E203

    return PL_image, PL_image_ROI, sig, ref, sig_norm, sweep_list


# ============================================================================


def _define_ROI(options, full_size_h, full_size_w):
    """
    Defines meshgrids that can be used to slice image into smaller region of interest (ROI).

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    full_size_w : int
        Width of image (after rebin, before ROI cut).

    full_size_h : int
        Height of image (after rebin, before ROI cut).

    Returns
    -------
    ROI : length 2 list of np meshgrids
        Defines an ROI that can be applied to the 3D image through direct indexing.
        E.g. sig_ROI = sig[:, ROI[0], ROI[1]]
    """
    systems.clean_options(options)

    if options["ROI"] == "Full":
        ROI = _define_area_roi(0, 0, full_size_w - 1, full_size_h - 1)
    elif options["ROI"] == "Rectangle":
        start_x, start_y = options["ROI_start"]
        end_x, end_y = options["ROI_end"]
        ROI = _define_area_roi(start_x, start_y, end_x, end_y)

    return ROI


# ============================================================================


def _define_area_roi(start_x, start_y, end_x, end_y):
    """
    Makes a list with of meshgrids that defines the ROI. Rectangular ROI

    Arguments
    ---------
    start_x, start_y, end_x, end_y : int
        Positions of ROI vertices as coordinates of indices.
        Defines a rectangle between (start_x, start_y) and (end_x, end_y).

    Returns
    -------
    ROI : length 2 list of np meshgrids
        Defines an ROI that can be applied to the 3D image through direct indexing.
        E.g. sig_ROI = sig[:, ROI[0], ROI[1]]
    """
    x = np.linspace(start_x, end_x, end_x - start_x + 1, dtype=int)
    y = np.linspace(start_y, end_y, end_y - start_y + 1, dtype=int)
    xv, yv = np.meshgrid(x, y)
    return [yv, xv]  # arrays are indexed in image convention, e.g. sig[swwpp_param, y, x]


# ============================================================================


def define_AOIs(options):
    """
    Defines areas of interest (AOIs).

    Returns list of AOIs that can be used to directly index into image array, e.g.:
    sig_AOI = sig[:, AOI[0], AOI[1]].

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    AOIs : list
        List of AOI regions. Much like ROI object, these are a length-2 list of np meshgrids
        that can be used to directly index into image to provide a view into just the AOI
        part of the image. E.g. sig_AOI = sig[:, AOI[0], AOI[1]]. Returns a list as in
        general we have more than one area of interest.
        I.e. sig_AOI_1 = sig[:, AOIs[1][0], AOIs[1][1]]
    """
    AOIs = []

    i = 0
    while True:
        i += 1
        try:
            start = options["AOI_" + str(i) + "_start"]
            end = options["AOI_" + str(i) + "_end"]

            if start is None or end is None:
                continue
            AOIs.append(_define_area_roi(*start, *end))
        except KeyError:
            break
    return AOIs


# ============================================================================


def _recursive_dict_update(to_be_updated_dict, updating_dict):
    """
    Recursively updates to_be_updated_dict with values from updating_dict (to all dict depths)
    """
    if not isinstance(to_be_updated_dict, collections.abc.Mapping):
        return updating_dict
    for key, val in updating_dict.items():
        if isinstance(val, collections.abc.Mapping):
            # avoids KeyError by returning {}
            to_be_updated_dict[key] = _recursive_dict_update(to_be_updated_dict.get(key, {}), val)
        else:
            to_be_updated_dict[key] = val
    return to_be_updated_dict


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

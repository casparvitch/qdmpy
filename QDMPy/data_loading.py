# -*- coding: utf-8 -*-
"""
This module holds tools for loading raw data etc. and reshaping to a usable format

Functions
---------
 - `QDMPy.data_loading.load_options`
 - `QDMPy.data_loading.save_options`
 - `QDMPy.data_loading.reshape_dataset`
 - `QDMPy.data_loading.prev_options_exist`
 - `QDMPy.data_loading.get_prev_options`
 - `QDMPy.data_loading.check_if_already_processed`
 - `QDMPy.data_loading.check_ROI_compatibility`
 - `QDMPy.data_loading.reshape_raw`
 - `QDMPy.data_loading.rebin_image`
 - `QDMPy.data_loading.define_ROI`
 - `QDMPy.data_loading.define_area_roi`
 - `QDMPy.data_loading.define_area_roi_centre`
 - `QDMPy.data_loading.remove_unwanted_sweeps`
 - `QDMPy.data_loading.define_AOIs`
"""

# ============================================================================

__author__ = "Sam Scholten"

# ============================================================================


import numpy as np
import warnings
import os
import pathlib
from collections import OrderedDict  # insertion order is guaranteed for py3.7+, but to be safe!

# ============================================================================

import QDMPy.misc as misc
import QDMPy.systems as systems

# ============================================================================

DIR_PATH = systems.DIR_PATH

# ============================================================================
#
# =============== OUTWARD-FACING FUNCTIONS
#
# ============================================================================


def load_options(options_dict=None, options_path=None, check_for_prev_result=False):
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

    Returns
    -------
    options : dict
        Generic options dict holding all the user options.
    """

    # only options_dict OR options_path
    if options_dict is not None and options_path is not None:
        raise RuntimeError("pass either options_dict OR options_path to load_options")
    if options_dict is None and options_path is None:
        raise RuntimeError("pass one of options_dict OR options_path to load_options")

    if options_path is not None:
        prelim_options = misc.json_to_dict(options_path)
    else:
        prelim_options = OrderedDict(options_dict)  # unnescessary py3.7+, leave to describe intent

    required_options = ["filepath", "fit_functions"]
    for key in required_options:
        if key not in prelim_options:
            raise RuntimeError(f"Must provide these options: {required_options}")

    sys = systems.choose_system(prelim_options["system_name"])

    sys.system_specific_option_update(prelim_options)

    metadata = sys.read_metadata(prelim_options["filepath"])

    prelim_options["metadata"] = metadata  # add metadata to options dict

    options = sys.get_default_options()  # first load in default options
    options.update(prelim_options)  # now update with what has been decided upon by user

    options["system"] = sys

    systems.clean_options(options)  # check all the options make sense

    options["original_bin"] = int(metadata["Binning"])
    if not int(options["additional_bins"]):
        options["total_bin"] = options["original_bin"]
    else:
        options["total_bin"] = options["original_bin"] * int(options["additional_bins"])

    # create output directories
    output_dir = pathlib.PurePosixPath(
        str(options["filepath"]) + "_processed" + "_bin_" + str(options["total_bin"])
    )
    options["output_dir"] = output_dir
    options["data_dir"] = output_dir / "data"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(options["data_dir"]):
        os.mkdir(options["data_dir"])

    # don't always check for prev. results (so we can use this fn in other contexts)
    if check_for_prev_result:
        check_if_already_processed(options)
    else:
        options["reloaded_prev_fit"] = False
        options["found_prev_result"] = False

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
    misc.dict_to_json(save_options, "saved_options.json", path_to_dir=options["output_dir"])


# ============================================================================


def load_raw_and_sweep(options):
    """
    Reads raw image data and sweep_list (affine parameters) using system methods

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    raw_data : np array, 1D (unshaped)
        Raw unshaped data read from binary file

    sweep_list : list
        List of sweep parameter values
    """
    systems.clean_options(options)

    raw_data = options["system"].read_raw(options["filepath"])
    sweep_list = options["system"].read_sweep_list(options["filepath"])
    return raw_data, sweep_list


# ============================================================================


def reshape_dataset(options, raw_data, sweep_list):
    """
    Reshapes and re-bins raw data into more useful format.

    Cuts down to ROI and removes unwanted sweeps.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    raw_data : np array, 1D (unshaped)
        Raw unshaped data read from binary file

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
        Format: [sweep_vals, x, y]

    ref : np array, 3D
        Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, x, y]

    sig_norm : np array, 3D
        Signal normalised by reference (via subtraction or normalisation, chosen in options),
        reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI.
        Format: [sweep_vals, x, y]

    single_pixel_pl : np array, 1D
        Normalised measurement array, for chosen single pixel check.

    sweep_list : list
        List of sweep parameter values (with removed unwanted sweeps at start/end)
    """
    systems.clean_options(options)

    image = reshape_raw(options, raw_data, sweep_list)
    image_rebinned, sig, ref, sig_norm = rebin_image(options, image)
    try:
        size_h, size_w = image_rebinned.shape[1:]
    except Exception:
        size_h, size_w = image_rebinned.shape
    ROI = define_ROI(options, size_h, size_w)

    # somewhat important a lot of this isn't hidden, so we can adjust it later
    PL_image, PL_image_ROI, sig, ref, sig_norm, sweep_list = remove_unwanted_sweeps(
        options, image_rebinned, sweep_list, sig, ref, sig_norm, ROI
    )  # also cuts sig etc. down to ROI

    # single pixel check
    single_pixel_pl = sig_norm[
        :, options["single_pixel_check"][0], options["single_pixel_check"][1]
    ]

    return PL_image, PL_image_ROI, sig, ref, sig_norm, single_pixel_pl, sweep_list


# ============================================================================
#
# =============== INWARD-FACING FUNCTIONS
#
# ============================================================================


def prev_options_exist(options):
    """
    Checks if options file from previous result can be found in default location, returns Bool.
    """
    prev_opt_path = os.path.normpath(options["output_dir"] / "saved_options.json")
    return os.path.exists(prev_opt_path)


# ============================================================================


def get_prev_options(options):
    """
    Reads options file from previous fit result (.json), returns a dictionary.
    """
    return misc.json_to_dict(options["output_dir"] / "saved_options.json")

    # old version:
    # prev_opt_path = os.path.normpath(options["output_dir"] / "saved_options.json")
    # f = open(prev_opt_path)
    # json_str = f.read()
    # return json.loads(json_str)


# ============================================================================


def check_if_already_processed(options):
    """
    Looks for previous fit result.

    If previous fit result exists, checks for compatibility between ROI option choices.

    Returns nothing.
    """

    if prev_options_exist(options):
        options["found_prev_result"] = True

    # 'fit_pixels' test is there to see if the user actually cares about pixel info
    if not options["force_fit"] and options["found_prev_result"] and options["fit_pixels"]:
        # i.e. we are going to try reloading the prev dataset
        options["reloaded_prev_fit"] = True  # set for the record

        # check ROI regions are the same
        prev_options = get_prev_options(options)
        check_ROI_compatibility(options, prev_options)


# ============================================================================


def check_ROI_compatibility(options, prev_options):
    """
    Check if ROI option in current options and previous options are compatible.

    When re-loading, want to ensure we have the same ROI settings as what the
    previous processing was run under. Allow user to override (i.e. copy old
    ROI settings) with "copy_prev_ROI_options" option.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    prev_options : ndict
        Generic options dict from previous fit result.

    Returns
    -------
    Nothing
    """
    failure_string = """
    Detected previous (similar) fit results. We would skip fitting the pixels
    and load these previous results, but ROI options were not the same. To avoid issues with
    plotting inconsistent PL/param arrays etc. and to remove ambiguity, the previous fit results
    will not be loaded ("force_fit" has been set to True).
     - If you intended to fit the data again (i.e. in different ROI) then don't worry about
       this message.
     - If you want to speed things up a bit and use this automatic reload feature, then try
       setting "auto_match_prev_ROI_options" to True. It will not effect any other part of the
       analysis process.
    """

    ROI_settings = ["ROI", "ROI_halfsize", "ROI_centre", "ROI_rect_size"]
    if options["auto_match_prev_ROI_options"]:
        for key in ROI_settings:
            options[key] = prev_options[key]
    else:
        for key in ROI_settings:
            if options[key] != prev_options[key]:
                warnings.warn(failure_string)
                options["reloaded_prev_fit"] = False
                options["force_fit"] = True
                break


# ============================================================================


def reshape_raw(options, raw_data, sweep_list):
    """
    Reshapes raw data into more useful shape, according to image size in metadata.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    raw_data : np array, 1D (unshaped)
        Raw unshaped data read from binary file

    sweep_list : list
        List of sweep parameter values

    Returns
    -------
    image : np array, 3D
        Format: [sweep values, x, y]. Has not been seperated into sig/ref etc. and has
        not been rebinned. Unwanted sweep values not removed.

    """

    systems.clean_options(options)

    options["used_ref"] = False  # flag for later

    try:
        if not options["ignore_ref"]:
            data_pts = len(sweep_list)
            image = np.reshape(
                raw_data,
                [
                    data_pts,
                    int(options["metadata"]["AOIHeight"]),
                    int(options["metadata"]["AOIWidth"]),
                ],
            )
        else:
            raise ValueError
    except ValueError:
        # if the ref is used then there's 2* the number of sweeps
        # i.e. auto-detect reference existence
        data_pts = 2 * len(sweep_list)
        if options["ignore_ref"]:
            image = np.reshape(
                raw_data,
                [
                    data_pts,
                    int(options["metadata"]["AOIHeight"]),
                    int(options["metadata"]["AOIWidth"]),
                ],
            )[
                ::2
            ]  # hmmm disregard ref -> use every second element
        else:
            image = np.reshape(
                raw_data,
                [
                    data_pts,
                    int(options["metadata"]["AOIHeight"]),
                    int(options["metadata"]["AOIWidth"]),
                ],
            )
            options["used_ref"] = True
    # Transpose the dataset to get the correct x and y orientations
    # will work for non-square images
    return image.transpose([0, 2, 1]).copy()


# ============================================================================


def rebin_image(options, image):
    """
    Reshapes raw data into more useful shape, according to image size in metadata.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    image : np array, 3D
        Format: [sweep values, x, y]. Has not been seperated into sig/ref etc. and has
        not been rebinned.

    Returns
    -------
    image_rebinned : np array, 3D
        Format: [sweep values, x, y]. Same as image, but now rebinned (x size and y size
        have changed). Not cut down to ROI.

    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet.
        Format: [sweep_vals, x, y]. Not cut down to ROI.

    ref : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet.
        Not cut down to ROI. Format: [sweep_vals, x, y].

    sig_norm : np array, 3D
        Signal normalised by reference (via subtraction or normalisation, chosen in options),
        reshaped and rebinned.  Unwanted sweeps not removed yet.
        Not cut down to ROI. Format: [sweep_vals, x, y].
    """
    systems.clean_options(options)

    if not options["additional_bins"]:
        image_rebinned = image
    else:
        if options["additional_bins"] % 2:
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
            sig_norm = 1 + sig - ref
        elif options["normalisation"] == "div":
            sig_norm = sig / ref
        else:
            raise KeyError("bad normalisation option, use: ['sub', 'div']")
    else:

        sig = ref = image_rebinned
        sig_norm = sig / np.max(sig, 0)

    return image_rebinned, sig, ref, sig_norm


# ============================================================================


def define_ROI(options, full_size_h, full_size_w):
    """
    Defines meshgrids that can be used to slice image into smaller region of interest (ROI).

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    full_size_h : int
        Height of image.

    full_size_w : int
        Width of image.

    Returns
    -------
    ROI : length 2 list of np meshgrids.
        Defines an ROI that can be applied to the 3D image through direct indexing.
        E.g. sig_ROI = sig[:, ROI[0], ROI[1]]
    """
    systems.clean_options(options)

    if options["ROI"] == "Full":
        ROI = define_area_roi(0, 0, full_size_w - 1, full_size_h - 1)
    elif options["ROI"] == "Square":
        ROI = define_area_roi_centre(options["ROI_centre"], 2 * options["ROI_halfsize"])
    elif options["ROI"] == "Rectangle":
        start_x = int(options["ROI_centre"][0] - options["ROI_rect_size"][0] / 2)
        start_y = int(options["ROI_centre"][1] - options["ROI_rect_size"][1] / 2)
        end_x = int(options["ROI_centre"][0] + options["ROI_rect_size"][0] / 2)
        end_y = int(options["ROI_centre"][1] + options["ROI_rect_size"][1] / 2)
        ROI = define_area_roi(start_x, start_y, end_x, end_y)

    return ROI


# ============================================================================


def define_area_roi(start_x, start_y, end_x, end_y):
    """
    Makes a list with of meshgrids that defines the ROI. Rectangular ROI

    Arguments
    ---------
    start_x, start_y, end_x, end_y : int
        Positions of ROI vertices as coordinates of indices.
        Defines a rectangle between (start_x, start_y) and (end_x, end_y).

    Returns
    -------
    ROI : length 2 list of np meshgrids.
        Defines an ROI that can be applied to the 3D image through direct indexing.
        E.g. sig_ROI = sig[:, ROI[0], ROI[1]]
    """
    x = [np.linspace(start_x, end_x, end_x - start_x + 1, dtype=int)]
    y = [np.linspace(start_y, end_y, end_y - start_y + 1, dtype=int)]
    xv, yv = np.meshgrid(x, y)
    return [yv, xv]


# ============================================================================


def define_area_roi_centre(centre, size):
    """
    Makes a list with of meshgrids that defines the ROI. Square ROI.

    Arguments
    ---------
    centre, size : int
        Defines centre and size of ROI square.

    Returns
    -------
    ROI : length 2 list of np meshgrids.
        Defines an ROI that can be applied to the 3D image through direct indexing.
        E.g. sig_ROI = sig[:, ROI[0], ROI[1]]
    """
    x = [np.linspace(centre[0] - size / 2, centre[0] + size / 2, size + 1, dtype=int)]
    y = [np.linspace(centre[1] - size / 2, centre[1] + size / 2, size + 1, dtype=int)]
    xv, yv = np.meshgrid(x, y)
    return [yv, xv]


# ============================================================================


def remove_unwanted_sweeps(options, image_rebinned, sweep_list, sig, ref, sig_norm, ROI):
    """
    Removes unwanted sweep values (i.e. freq values or tau values) for all of the data arrays.

    Also cuts data down to ROI.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    image_rebinned : np array, 3D
        Format: [sweep values, x, y]. Same as image, but rebinned (x size and y size
        have changed).

    sweep_list : list
        List of sweep parameter values

    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet.
        Format: [sweep_vals, x, y]

    ref : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet.
        Format: [sweep_vals, x, y]

    sig_norm : np array, 3D
        Signal normalised by reference (via subtraction or normalisation, chosen in options),
        reshaped and rebinned.  Unwanted sweeps not removed yet.
        Format: [sweep_vals, x, y]

    ROI : length 2 list of np meshgrids.
        Defines an ROI that can be applied to the 3D image through direct indexing.
        E.g. sig_ROI = sig[:, ROI[0], ROI[1]]

    Returns
    -------
    PL_image : np array, 2D.
        Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and
        cut down to ROI

    PL_image_ROI : np array, 2D
        Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped and rebinned as
        well as cut down to ROI.

    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Format: [sweep_vals, x, y]

    ref : np array, 3D
        Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Format: [sweep_vals, x, y]

    sig_norm : np array, 3D
        Signal normalised by reference (via subtraction or normalisation, chosen in options),
        reshaped and rebinned. Unwanted sweeps removed.
        Format: [sweep_vals, x, y]

    sweep_list : list
        List of sweep parameter values (with removed unwanted sweeps at start/end)
    """
    systems.clean_options(options)

    # here ensure we have copies, not views
    rem_start = options["remove_start_sweep"]
    rem_end = options["remove_end_sweep"]
    PL_image = np.sum(image_rebinned, axis=0)
    PL_image_ROI = PL_image[ROI[0], ROI[1]].copy()
    sig = sig[rem_start : -1 - rem_end, ROI[0], ROI[1]].copy()  # noqa: E203
    ref = ref[rem_start : -1 - rem_end, ROI[0], ROI[1]].copy()  # noqa: E203
    sig_norm = sig_norm[rem_start : -1 - rem_end, ROI[0], ROI[1]].copy()  # noqa: E203
    sweep_list = np.asarray(sweep_list[rem_start : -1 - rem_end]).copy()  # noqa: E203

    return PL_image, PL_image_ROI, sig, ref, sig_norm, sweep_list


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
            centre = options["area_" + str(i) + "_centre"]
            halfsize = options["area_" + str(i) + "_halfsize"]

            if centre is None or halfsize is None:
                break

            AOIs.append(define_area_roi_centre(centre, 2 * halfsize))
        except KeyError:
            break
    return AOIs


# ============================================================================

# -*- coding: utf-8 -*-

__author__ = "Sam Scholten"


import numpy as np
import warnings
import os
import pathlib
import json

import misc
import systems

# ============================================================================


# ============================================================================
#
# =============== OUTWARD-FACING FUNCTIONS
#
# ============================================================================


def load_options(path="options/fit_options.json", check_for_prev_result=False):
    prelim_options = misc.json_to_dict(path)  # requires main to be dir above opts

    sys = systems.choose_system(prelim_options["system_name"])
    metadata = sys.read_metadata(prelim_options["filepath"])

    prelim_options.update(metadata)  # add metadata to options dict

    options = sys.get_default_options()  # first load in default options
    options.update(prelim_options)  # now update with what has been decided upon by user

    sys.system_specific_option_update(options)

    options["system"] = sys

    systems.clean_options(options)  # check all the options make sense

    options["original_bin"] = int(metadata["Binning"])
    if not int(options["additional_bins"]):
        options["total_bin"] = options["original_bin"]
    else:
        options["total_bin"] = options["original_bin"] * int(options["additional_bins"])

    if check_for_prev_result:
        check_if_already_processed(options)

    return options


# ============================================================================


def load_raw_and_sweep(options):
    systems.clean_options(options)

    raw_data = options["system"].read_raw(options["filepath"])
    sweep_list = options["system"].read_sweep_list(options["filepath"])
    return raw_data, sweep_list


# ============================================================================


def reshape_dataset(options, raw_data, sweep_list):
    systems.clean_options(options)

    # ok now start transforming dataset (again should depend on processed/not processed)
    image = reshape_raw(options, raw_data, sweep_list)
    image_rebinned, sig, ref, sig_norm = rebin_image(options, image)
    ROI = define_roi()

    # somewhat important a lot of this isn't hidden, so we can adjust it later
    image_ROI, sig, ref, sig_norm, sweep_list = remove_unwanted_sweeps(
        options, image_rebinned, sweep_list, sig, ref, sig_norm, ROI
    )  # also cuts sig etc. down to ROI

    return image_ROI, sig, ref, sig_norm, sweep_list


# ============================================================================
#
# =============== INWARD-FACING FUNCTIONS
#
# ============================================================================


def get_prev_options(options):
    prev_opt_path = os.path.normpath(options["output_dir"] + "/saved_options.json")
    f = open(prev_opt_path)
    json_str = f.read()
    return json.loads(json_str)


# ============================================================================


def check_if_already_processed(options):
    output_dir = pathlib.PurePosixPath(
        options["filepath"] + "_processed" + "_bin_" + str(options["total_bin"])
    )
    options["output_dir"] = output_dir
    options["data_dir"] = output_dir / "data"

    if os.path.exists(output_dir):
        options["found_prev_result"] = True
    else:
        os.mkdir(output_dir)
        if not os.path.exists(options["data_dir"]):
            os.mkdir(options["data_dir"])

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
    When re-loading, want to ensure we have the same ROI settings as what the
    previous processing was run under. Allow user to override (i.e. copy old
    ROI settings) with "copy_prev_ROI_options" option.
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

    ROI_settings = ["ROI", "ROI_size", "ROI_centre", "ROI_rec_size"]
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

    systems.clean_options(options)

    options["used_ref"] = False  # flag for later

    try:
        if not options["ignore_ref"]:
            data_pts = len(sweep_list)
            image = np.reshape(
                raw_data,
                [data_pts, int(options["AOIHeight"]), int(options["AOIWidth"])],
            )
        else:
            raise ValueError
    except ValueError:
        # if the ref is used then there's 2* the number of sweeps
        data_pts = 2 * len(sweep_list)
        if options["ignore_ref"]:
            # use every second element
            image = np.reshape(
                raw_data[::2],
                [data_pts, int(options["AOIHeight"]), int(options["AOIWidth"])],
            )
        else:
            image = np.reshape(
                raw_data,
                [data_pts, int(options["AOIHeight"]), int(options["AOIWidth"])],
            )
            warnings.warn(
                "Detected that dataset has reference. "
                + "Continuing processing using the reference."  # noqa: W503
            )
            options["used_ref"] = True
    # Transpose the dataset to get the correct x and y orientations
    # will work for non-square images
    return image.transpose([0, 2, 1]).copy()


# ============================================================================


def rebin_image(options, image):
    systems.clean_options(options)

    if options["additional_bins"] in [0, 1]:
        image_rebinned = image
    else:
        if options["additional_bins"] % 2:
            raise RuntimeError("The binning parameter needs to be a multiple of 2.")
        # TODO add if_processed check here
        if False:
            pass
        else:
            data_pts = image.shape[0]
            height = image.shape[1]
            width = image.shape[2]
            image_rebinned = (
                np.reshape(
                    image,
                    [
                        data_pts,
                        int(height / options["additional_bins"]),
                        data_pts,
                        int(width / options["additional_bins"]),
                        data_pts,
                    ],
                )
                .sum(2)
                .sum(3)
            )
    # define sig and ref differently if we're using a ref
    # this 'True' is only if_processed
    if True:
        if options["used_ref"]:
            sig = image_rebinned[::2, :, :]
            ref = image_rebinned[1::2, :, :]
            if options["normalisation"] == "sub":
                sig_norm = sig - ref
            elif options["normalisation"] == "div":
                sig_norm = sig / ref
            else:
                raise KeyError("bad normalisation option, use: ['sub', 'div']")
        else:

            sig = ref = image_rebinned
            sig_norm = sig / np.max(sig, 0)

    return image_rebinned, sig, ref, sig_norm


# ============================================================================


def define_roi(options, image_rebinned):
    systems.clean_options(options)

    # from old code, not sure what case it handles
    try:
        size_h, size_w = image_rebinned.shape[1:]
    except Exception:
        size_h, size_w = image_rebinned.shape
        warnings.warn("Not sure what this try/except statement is checking. Clarify.")

    if options["ROI"] == "Full":
        ROI = define_area_roi(0, 0, size_w - 1, size_h - 1)
    elif options["ROI"] == "Square":
        ROI = define_area_roi_centre(options["ROI_centre"], 2 * options["ROI_size"])
    elif options["ROI"] == "Rectangle":
        start_x = options["ROI_centre"][0] - options["ROI_rect_size"][0]
        start_y = options["ROI_centre"][1] - options["ROI_rect_size"][1]
        end_x = options["ROI_centre"][0] + options["ROI_rect_size"][0]
        end_y = options["ROI_centre"][1] + options["ROI_rect_size"][1]
        ROI = define_area_roi(start_x, start_y, end_x, end_y)

    return ROI


# ============================================================================


def define_area_roi(start_x, start_y, end_x, end_y):
    """Makes a list with a mesh that defines the an ROI
    This ROI can be simply applied to the 2D image through direct
    indexing, e.g new_image = image(:,ROI[0],ROI[1]) with shink the
    ROI of the image.
    """
    x = [np.linspace(start_x, end_x, end_x - start_x + 1, dtype=int)]
    y = [np.linspace(start_y, end_y, end_y - start_y + 1, dtype=int)]
    xv, yv = np.meshgrid(x, y)
    return [yv, xv]


# ============================================================================


def define_area_roi_centre(centre, size):
    x = [np.linspace(centre[0] - size / 2, centre[0] + size / 2, size + 1, dtype=int)]
    y = [np.linspace(centre[1] - size / 2, centre[1] + size / 2, size + 1, dtype=int)]
    xv, yv = np.meshgrid(x, y)
    return [yv, xv]


# ============================================================================


def remove_unwanted_sweeps(options, image_rebinned, sweep_list, sig, ref, sig_norm, ROI):
    systems.clean_options(options)

    # here ensure we have copies, not views
    rem_start = options["remove_start_sweep"]
    rem_end = options["remove_end_sweep"]
    image_ROI = image_rebinned[:, ROI[0], ROI[1]].copy()
    sig = sig[rem_start : -1 - rem_end, ROI[0], ROI[1]].copy()  # noqa: E203
    ref = ref[rem_start : -1 - rem_end, ROI[0], ROI[1]].copy()  # noqa: E203
    sig_norm = sig_norm[rem_start : -1 - rem_end, ROI[0], ROI[1]].copy()  # noqa: E203
    sweep_list = np.asarray(sweep_list[rem_start : -1 - rem_end]).copy()  # noqa: E203

    return image_ROI, sig, ref, sig_norm, sweep_list


# ============================================================================


def define_AOIs(options):
    AOIs = []

    i = 0
    while True:
        i += 1
        try:
            centre = options["area_" + str(i) + "_centre"]
            size = 2 * options["area_" + str(i) + "_size"]
            AOIs.append(define_area_roi(centre, size))
        except KeyError:
            break


# ============================================================================

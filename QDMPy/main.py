# -*- coding: utf-8 -*-

__author__ = "Sam Scholten"


import misc
import os
import numpy as np
import pathlib
import warnings

import institutes
import systems

# NOTES
# - chuck first section into a data_handling folder (sub-package)
# - separate fns to plot results (these should return results in some format)
# ->> future design = jupyter

# - need modular code: don't produce output unless the user asks for it, so we can test
#   etc. without running everything

# - IDEA: hold important info in struct, ie options + metadata?


def read_processed_data(options, fitted_param=None):
    """ TODO """
    return np.loadtxt(options["filepath_data"] + "/" + fitted_param + ".txt")


def reshape_raw(options, raw_data, sweep_list):
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
                + "Continuing processing using the reference."
            )
    # Transpose the dataset to get the correct x and y orientations
    # will work for non-square images
    return image.transpose([0, 2, 1]).copy()


def rebin_image(options, image):
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


def define_roi(options, image_rebinned):

    # from old code, not sure what case it handles
    try:
        size_h, size_w = image_rebinned.shape[1:]
    except:
        size_h, size_w = image_rebinned.shape
        warnings.warn("Not sure what this try/except statement is checking. Clarify.")

    if options["ROI"] == "Full":
        ROI = define_area_roi(0, 0, size_w - 1, size_h - 1)
    elif options["ROI"] == "Square":
        ROI = define_area_roi_centre(options["ROI_centre"], 2 * options["ROI_radius"])
    # might be better to add ellipse etc. masks later?
    # realistically, not going to do FFT etc. on circular mask, won't save that much time
    # elif options["ROI"] == "Circle":
    #     ROI = define_area_roi_centre(options["ROI_centre"], 2 * options["ROI_radius"])
    elif options["ROI"] == "Rectangle":
        start_x = options["ROI_centre"][0] - options["ROI_rect_size"][0]
        start_y = options["ROI_centre"][1] - options["ROI_rect_size"][1]
        end_x = options["ROI_centre"][0] + options["ROI_rect_size"][0]
        end_y = options["ROI_centre"][1] + options["ROI_rect_size"][1]
        ROI = define_area_roi(start_x, start_y, end_x, end_y)
    else:
        # clean this up, make optionserror nicer + easier to use?
        raise OptionsError(
            "Option 'ROI' not in valid options: 'Full', 'Square', 'Circle', 'Rectangle'."
        )
    return ROI


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


def define_area_roi_centre(centre, size):
    x = [np.linspace(centre[0] - size / 2, centre[0] + size / 2, size + 1, dtype=int)]
    y = [np.linspace(centre[1] - size / 2, centre[1] + size / 2, size + 1, dtype=int)]
    xv, yv = np.meshgrid(x, y)
    return [yv, xv]


def remove_unwanted_sweeps(
    options, image_rebinned, sweep_list, sig, ref, sig_norm, ROI
):
    rem_start = options["remove_start_sweep"]
    rem_end = options["remove_end_sweep"]
    image_ROI = image_rebinned[:, ROI[0], ROI[1]]
    sig = sig[rem_start : -1 - rem_end, ROI[0], ROI[1]]  # noqa: E203
    ref = ref[rem_start : -1 - rem_end, ROI[0], ROI[1]]  # noqa: E203
    sig_norm = sig_norm[rem_start : -1 - rem_end, ROI[0], ROI[1]]  # noqa: E203
    sweep_list = np.asarray(sweep_list[rem_start : -1 - rem_end])  # noqa: E203

    return image_ROI, sig, ref, sig_norm, sweep_list


available_peak_types = {
    # "lorentzian": Lorentzian,
    # "lorentzian_hyperfine_14": Lorentzian_hyperfine_14,
    # "lorentzian_hyperfine_15": Lorentzian_hyperfine_15,
    # "gaussian": Gaussian,
    # "gaussian_hyperfine_14": Gaussian_hyperfine_14,
    # "gaussian_hyperfine_15": Gaussian_hyperfine_15,
    # "constant": Constant,
    # "linear": Linear,
    # "circular": Circular,
}


def gen_init_guesses(options):
    guess_dict = {}
    bound_dict = {}
    peaks = available_peak_types[options["lineshape"]](options["num_peaks"])
    for param_key in peaks.param_defn:
        # < TODO > add an auto guess for the peak positions?
        # yse scipy.signal.find_peaks?
        guess = options[param_key + "_guess"]
        val = guess
        if param_key + "_range" in options:
            if type(guess) is list and len(guess) > 1:
                val_b = [
                    [
                        x - options[param_key + "_range"],
                        x + options[param_key + "_range"],
                    ]
                    for x in guess
                ]
            else:
                # print(guess)
                # print(options[param_key + "_range"])
                val_b = [
                    guess - options[param_key + "_range"],
                    guess + options[param_key + "_range"],
                ]
        elif param_key + "_bounds" in options:
            val_b = options[param_key + "_bounds"]
        else:
            val_b = [[0, 0]]
        if val is not None:
            guess_dict[param_key] = val
            bound_dict[param_key] = np.array(val_b)
        else:
            raise RuntimeError(
                "I'm not sure what this means... I know "
                + "it's bad though... Don't put 'None' as "
                + "a param guess."
            )
    return guess_dict, bound_dict


def define_fit_model(options):
    guess_dict, bound_dict = gen_init_guesses(options)


class FitModel:
    def __init__(self, fn, num_fns, guess_dict, bound_dict, fit_param_ar, fit_param, fit_param_bound_ar):
        

def main(__spec__=None):

    # TODO: build from default dictionary first, then update with what's read in
    # ok maybe have the default dict handled by the choice of system? Ehh
    options = misc.json_to_dict("options/<NAME>.json")

    institute = institutes.choose_institute(options["institute"])
    system = systems.choose_system(options["system"])

    raw_data = institute.read_raw(options["filepath"])
    sweep_list = institute.read_sweep_list(options["filepath"])
    metadata = institute.read_metadata(options["filepath"])
    options.update(metadata)  # add metadata to options dict

    # TODO: check if same thing has already been processed, check all (non-plotting) options
    # are the same. If not, will need to create a different output dir
    # below: assuming haven't processed already. Write into a couple of functions

    options["original_bin"] = int(metadata["Binning"])
    if not int(options["additional_bins"]):
        options["total_bin"] = options["original_bin"]
    else:
        options["total_bin"] = options["original_bin"] * int(options["additional_bins"])

    output_dir = pathlib.PurePosixPath(
        options["filepath"] + "_processed" + "_bin_" + str(options["total_bin"])
    )
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    options["output_dir"] = output_dir

    options["data_dir"] = output_dir / "data"
    if not os.path.exists(options["data_dir"]):
        os.mkdir(options["data_dir"])

    # ok now start transforming dataset (again should depend on processed/not processed)
    # careful here -> ensure copies of array not views etc.
    image = reshape_raw(options, raw_data, sweep_list)
    image_rebinned, sig, ref, sig_norm = rebin_image(options, image)
    ROI = define_roi()
    # mask()
    image_ROI, sig, ref, sig_norm, sweep_list = remove_unwanted_sweeps(
        options, image_rebinned, sweep_list, sig, ref, sig_norm, ROI
    )

    # plot_ROI()

    define_fit_model()

    fit_ROI()

    define_AOIs()
    fit_AOIs()

    # plot_AOI_comparison()

    # plot_ROI_fit()

    # ok stop process here, delete any unneeded data, now moving on to the pixel fitting
    fit_pixels()  # remember to scramble pixels
    plot_fit_results()  # ok need to expand this to more direct functions {params, etc.?}

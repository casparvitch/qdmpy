# -*- coding: utf-8 -*-

__author__ = "Sam Scholten"


import os
import pathlib

import systems
import fitting
import data_loading
import misc

# NOTES
# - separate fns to plot results (these should return results in some format)
# -> plotting file?
# ->> future design = jupyter

# - need modular code: don't produce output unless the user asks for it,
#  so we can test etc. without running everything


# package the below sections into clear functions.
# think in terms of jupyter cells -> load_options, setup_system, reshape_data, etc.


def main(__spec__=None):

    prelim_options = misc.json_to_dict("options/<NAME>.json")  # requires main to be dir above opts

    sys = systems.choose_system(prelim_options["system"])

    # read in some data
    raw_data = sys.read_raw(prelim_options["filepath"])
    sweep_list = sys.read_sweep_list(prelim_options["filepath"])
    metadata = sys.read_metadata(prelim_options["filepath"])

    prelim_options.update(metadata)  # add metadata to options dict

    options = sys.get_default_options()  # first load in default options
    options.update(prelim_options)  # now update with what has been decided upon by user

    options["system"] = sys

    systems.clean_options(options)  # check all the options make sense

    data_loading.check_if_already_processed(options)

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
    image = data_loading.reshape_raw(options, raw_data, sweep_list)
    image_rebinned, sig, ref, sig_norm = data_loading.rebin_image(options, image)
    ROI = data_loading.define_roi()

    # somewhat important a lot of this isn't hidden, so we can adjust it later
    image_ROI, sig, ref, sig_norm, sweep_list = data_loading.remove_unwanted_sweeps(
        options, image_rebinned, sweep_list, sig, ref, sig_norm, ROI
    )  # also cuts sig etc. down to ROI

    fit_model = fitting.define_fit_model()

    # roi_fit_result is an FitResultROI object,
    # see fitting file to see a nice explanation of contents
    roi_fit_result = fitting.fit_ROI(options, sig_norm, sweep_list, fit_model)

    # plot_ROI(sig_norm)

    # plot_ROI_fit() # residual!!!

    # want to do this in a scaled manner? User select with cursor, run in real time etc.?
    AOIs = data_loading.define_AOIs(options)
    AOI_fit_params = fitting.fit_AOIs(options, fit_model, AOIs)

    # plot_AOI_comparison() # residual, compare to ROI!!!

    # move on to the pixel fitting
    fitting.fit_pixels()  # remember to scramble pixels

    # plot_fit_results()  # ok need to expand this to more direct functions {params, etc.?}

    # Note on finishing, need to save options etc. Remember to remove 'system' information

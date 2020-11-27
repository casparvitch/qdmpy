# -*- coding: utf-8 -*-

__author__ = "Sam Scholten"


# should only include our sub-packages really.
import fitting
import data_loading

# NOTES
# - separate fns to plot results (these should return results in some format)
# -> plotting file?
# ->> future design = jupyter

# - need modular code: don't produce output unless the user asks for it,
#  so we can test etc. without running everything


# package the below sections into clear functions.
# think in terms of jupyter cells -> load_options, setup_system, reshape_data, etc.


def main(__spec__=None):

    options = data_loading.load_options()

    raw_data, sweep_list = data_loading.load_raw_and_sweep(options)

    image_ROI, sig, ref, sig_norm, sweep_list = data_loading.reshape_dataset(
        options, raw_data, sweep_list
    )

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

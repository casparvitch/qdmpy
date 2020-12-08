# -*- coding: utf-8 -*-

__author__ = "Sam Scholten"


# should only include our sub-packages really.
import fitting
import data_loading
import fit_plots

import matplotlib
import matplotlib.pyplot as plt

# NOTES
# - separate fns to plot results (these should return results in some format)
# -> plotting file?
# ->> future design = jupyter

# - need modular code: don't produce output unless the user asks for it,
#  so we can test etc. without running everything


# package the below sections into clear functions.
# think in terms of jupyter cells -> load_options, setup_system, reshape_data, etc.
# TODO add widgets etc. for interactive plots


def main(__spec__=None):
    options = data_loading.load_options(check_for_prev_result=True)

    raw_data, prelim_sweep_list = data_loading.load_raw_and_sweep(options)

    PL_image, PL_image_ROI, _, _, sig_norm, sweep_list = data_loading.reshape_dataset(
        options, raw_data, prelim_sweep_list
    )
    fit_model = fitting.define_fit_model(options)

    # roi_fit_result is a FitResultROI object,
    # see fitting file to see a nice explanation of contents
    roi_avg_fit_result = fitting.fit_ROI_avg(options, sig_norm, sweep_list, fit_model)

    fig = fit_plots.plot_ROI_avg_fit(options, roi_avg_fit_result)

    fig2 = fit_plots.plot_ROI_image(options, PL_image)

    AOIs = data_loading.define_AOIs(options)

    fig3 = fit_plots.plot_AOIs(options, PL_image_ROI, AOIs)

    AOI_fit_params = fitting.fit_AOIs(
        options, sig_norm, sweep_list, fit_model, AOIs, roi_avg_fit_result
    )
    print(AOI_fit_params)

    if False:

        # plot_AOI_comparison(options, AOI_fit_params, roi_fit_result) # residual, compare to ROI!!!

        if (options["force_fit"] or not options["found_prev_result"]) and options["fit_pixels"]:
            # move on to the pixel fitting
            if options["fit_pixels"]:
                pixel_fit_params = fitting.fit_pixels()

        else:  # load previous results, start plotting info
            pixel_fit_params = data_loading.load_prev_fit_results(options, fit_model)

    # ok the juice is below
    # plot_fit_results(options, pixel_fit_params)
    # ok need to expand this to more direct functions {params, etc.?}

    # Note on finishing, need to save options etc.
    # Remember to remove 'system' information i.e. all python objects etc.

    plt.show()


if __name__ == "__main__":
    main()

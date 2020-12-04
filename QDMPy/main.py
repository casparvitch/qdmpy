# -*- coding: utf-8 -*-

__author__ = "Sam Scholten"


# should only include our sub-packages really.
import fitting
import data_loading
import matplotlib.pyplot as plt

# NOTES
# - separate fns to plot results (these should return results in some format)
# -> plotting file?
# ->> future design = jupyter

# - need modular code: don't produce output unless the user asks for it,
#  so we can test etc. without running everything


# package the below sections into clear functions.
# think in terms of jupyter cells -> load_options, setup_system, reshape_data, etc.


def main(__spec__=None):

    options = data_loading.load_options(check_for_prev_result=True)
    raw_data, sweep_list = data_loading.load_raw_and_sweep(options)

    image_ROI, sig, ref, sig_norm, sweep_list = data_loading.reshape_dataset(
        options, raw_data, sweep_list
    )

    fit_model = fitting.define_fit_model(options)

    # roi_fit_result is a FitResultROI object,
    # see fitting file to see a nice explanation of contents
    roi_avg_fit_result = fitting.fit_ROI_avg(options, sig_norm, sweep_list, fit_model)

    res = roi_avg_fit_result

    fig = plt.figure(figsize=(8, 6))
    # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    spectrum_frame = fig.add_axes((0.1, 0.3, 0.8, 0.6))

    # ODMR spectrum
    spectrum_frame.plot(
        res.fit_sweep_vector,
        res.init_fit,
        linestyle=(0, (1, 1)),
        label="init guess",
        c="darkgreen",
    )
    spectrum_frame.plot(
        res.fit_sweep_vector,
        res.scipy_best_fit,
        linestyle="--",
        label="scipy best fit",
        c="mediumblue",
    )
    spectrum_frame.plot(
        res.sweep_list,
        res.pl_roi,
        label="raw data",
        ls=" ",
        marker="o",
        ms=3.5,
        mfc="w",
        mec="firebrick",
    )
    spectrum_frame.set_xticklabels([])  # remove from first frame
    spectrum_frame.legend()
    spectrum_frame.grid()

    # residual plot
    residual_frame = fig.add_axes((0.1, 0.1, 0.8, 0.2))
    res_xdata = res.sweep_list
    res_ydata = res.best_fit_pl_vals - res.pl_roi

    residual_frame.plot(
        res_xdata,
        res_ydata,
        label="residual",
        ls="dashed",
        lw=1,
        c="black",
        marker="o",
        ms=3.5,
        mfc="w",
        mec="k",
    )
    residual_frame.legend()
    residual_frame.grid()

    if False:
        # plot_ROI(sig_norm)

        # plot_ROI_avg_fit(options, roi_fit_result) # residual!!!

        # want to do this in a scaled manner? User select with cursor, run in real time etc.?
        AOIs = data_loading.define_AOIs(options)
        AOI_fit_params = fitting.fit_AOIs(options, fit_model, AOIs)

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


if __name__ == "__main__":
    main()

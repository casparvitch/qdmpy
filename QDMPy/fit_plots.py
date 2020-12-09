# -*- coding: utf-8 -*-
"""
Module docstring
"""

# ============================================================================

__author__ = "Sam Scholten"

# ============================================================================

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import numpy as np
import matplotlib.patches as patches

# ============================================================================

import systems
import fit_models
import fitting
import data_loading


"""
NOTES

- MHz has been hardcoded in some places for the plots here, how to generalise?
"""

COLORS = [
    "blue",
    "saddlebrown",
    "darkslategrey",
    "magenta",
    "olive",
    "cyan",
    "purple",
]

# ===========================================================================


def plot_ROI_PL_image(options, PL_image):
    # mmm plot o.g. image with ROI labelled

    # TODO grab all options from plot options etc...

    if options["make_plots"] is False:
        return None

    c_map = "Greys_r"
    c_range = [np.nanmin(PL_image), np.nanmax(PL_image)]

    fig, ax = plot_image(
        PL_image, "PL - ROI", c_map, c_range, "Counts", list(options["figure_size"]), None
    )

    # need to convert back to um?
    # pixel_size = options["system"].get_raw_pixel_size() * options["total_bin"]

    if options["annotate_image_regions"]:
        annotate_ROI_image(options, ax)

    # TODO implement elliptical/circular mask feature
    # --> make object that can be passed around, saved/pickled after being tested etc.

    return fig


# ============================================================================


def add_colourbar(im, fig, ax, aspect=20, pad_fraction=1, **kwargs):
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1.0 / aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cbar = fig.colorbar(im, cax=cax, **kwargs)
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.linewidth = 0.5
    cbar.ax.tick_params(direction="in", labelsize=12, size=5)
    return cbar


# ============================================================================


def add_patch_square_centre(ax, area_c, area_size, label=None, edgecolor="b"):
    """  add the ROI rectangles"""
    rect_corner = [int(area_c[0] - area_size / 2), int(area_c[1] - area_size / 2)]
    rect = patches.Rectangle(
        (rect_corner[0], rect_corner[1]),
        int(area_size),
        int(area_size),
        linewidth=1,
        edgecolor=edgecolor,
        facecolor="none",
    )
    ax.add_patch(rect)
    if label:
        # Add label for the square
        ax.text(
            area_c[0] + 0.95 * area_size,  # label posn.: top right
            area_c[1],
            label,
            {"color": edgecolor, "fontsize": 10, "ha": "center", "va": "center"},
        )


# ============================================================================


def add_patch_rect(ax, rect_corner_x, rect_corner_y, size_x, size_y, label=None, edgecolor="b"):
    """  add the ROI rectangle to an ax and label the rectangle if a label has been given """
    rect = patches.Rectangle(
        (rect_corner_x, rect_corner_y),
        int(size_x),
        int(size_y),
        linewidth=1,
        edgecolor=edgecolor,
        facecolor="none",
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            rect_corner_x + 0.95 * size_x,  # label posn.: top right
            rect_corner_y,
            label,
            {"color": edgecolor, "fontsize": 10, "ha": "center", "va": "bottom"},
        )


# ============================================================================


def annotate_ROI_image(options, ax):
    binning = options["additional_bins"]
    if binning == 0:
        binning = 1
    if options["ROI"] == "Full":
        return None
    elif options["ROI"] == "Square":
        size = options["ROI_radius"] * binning * 2
        corner = [
            options["ROI_centre"][0] * binning - size / 2,
            options["ROI_centre"][1] * binning - size / 2,
        ]

        add_patch_rect(ax, corner[0], corner[1], size, size, label="ROI", edgecolor="r")
    elif options["ROI"] == "Rectangle":
        start_x = options["ROI_centre"][0] - options["ROI_rect_size"][0] / 2
        start_y = options["ROI_centre"][1] - options["ROI_rect_size"][1] / 2
        size_x = options["ROI_rect_size"][0]
        size_y = options["ROI_rect_size"][1]

        add_patch_rect(ax, start_x, start_y, size_x, size_y, label="ROI", edgecolor="r")
    else:
        raise systems.OptionsError(
            "ROI", options["ROI"], options["system"], custom_msg="Unknown ROI encountered."
        )


# ============================================================================


def annotate_AOI_PL_image(options, ax):
    binning = options["additional_bins"]
    if binning == 0:
        binning = 1

    # annotate single pixel check
    corner = (options["single_pixel_check"][0], options["single_pixel_check"][1])
    size = 1
    add_patch_rect(ax, corner[0], corner[1], size, size, label="PX check", edgecolor=COLORS[0])

    i = 0
    while True:
        i += 1
        try:
            centre = options["area_" + str(i) + "_centre"]
            size = options["area_" + str(i) + "_halfsize"]
            if centre is None or size is None:
                break
            centre *= binning
            size *= binning

            corner = [
                centre[0] - size,
                centre[1] - size,
            ]

            add_patch_rect(
                ax,
                corner[0],
                corner[1],
                size,
                size,
                label="AOI " + str(i),
                edgecolor=COLORS[i],
            )
        except KeyError:
            break


# ============================================================================


def plot_AOI_PL_images(options, PL_image_ROI, AOIs):
    # here plot image_ROI cut down, label AOIs on it
    # similar to roi, just more of em ->> loop through all 'i' AOIs given
    if AOIs == []:
        return None

    # TODO grab all options from plot options etc...

    if options["make_plots"] is False:
        return None

    # need to convert back to um? TODO implement some sort of scalebar
    # pixel_size = options["system"].get_raw_pixel_size() * options["total_bin"]

    c_map = "Greys_r"
    c_range = [np.nanmin(PL_image_ROI), np.nanmax(PL_image_ROI)]

    fig, ax = plot_image(
        PL_image_ROI, "PL - AOIs", c_map, c_range, "Counts", list(options["figure_size"]), None
    )

    if options["annotate_image_regions"]:
        annotate_AOI_PL_image(options, ax)

    # TODO implement elliptical/circular mask feature
    # --> make object that can be passed around, saved/pickled after being tested etc.

    return fig


# ============================================================================


def plot_image(image_data, title, c_map, c_range, c_label, figure_size, pixel_size):
    fig, ax = plt.subplots(figsize=figure_size)

    cmap = c_map
    c_range = [np.nanmin(image_data), np.nanmax(image_data)]

    im = ax.imshow(image_data, cmap=c_map, vmin=c_range[0], vmax=c_range[1])

    ax.set_title(title)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    cbar = add_colourbar(im, fig, ax)
    cbar.ax.set_ylabel(c_label, rotation=270)
    return fig, ax


# ============================================================================


def plot_ROI_avg_fit(options, roi_avg_fit_result):
    # options is unused here, but passing it in for future-proofing

    res = roi_avg_fit_result

    fig = plt.figure(figsize=(list(options["figure_size"])))
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
    spectrum_frame.set_ylabel("PL (a.u.)")

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
    residual_frame.set_xlabel("MW Frequency (MHz)")
    residual_frame.set_ylabel("PL (a.u.)")

    return fig


# ============================================================================


def plot_AOI_spectra(options, AOIs, sig, ref, sweep_list):

    # pre-process data to plot
    sig_avgs = []
    ref_avgs = []
    for i, AOI in enumerate(AOIs):
        sig_avg = np.nansum(np.nansum(sig[:, AOI[0], AOI[1]], 2), 1)
        sig_avg = sig_avg / np.max(sig_avg)
        ref_avg = np.nansum(np.nansum(ref[:, AOI[0], AOI[1]], 2), 1)
        ref_avg = ref_avg / np.max(ref_avg)
        sig_avgs.append(sig_avg)
        ref_avgs.append(ref_avg)

    figsize = list(options["figure_size"])
    figsize[0] *= len(AOIs)
    figsize[1] *= 2
    fig, axs = plt.subplots(2, len(AOIs), sharex=True, sharey=False, figsize=figsize)

    for i, AOI in enumerate(AOIs):

        # plot sig
        axs[0, i].plot(
            sweep_list,
            sig_avgs[i],
            label="sig",
            c="blue",
            ls="dashed",
            lw=1,
            marker="o",
            ms=3.5,
            mfc="cornflowerblue",
            mec="mediumblue",
        )
        # plot ref
        axs[0, i].plot(
            sweep_list,
            ref_avgs[i],
            label="ref",
            c="green",
            ls="dashed",
            lw=1,
            marker="o",
            ms=3.5,
            mfc="limegreen",
            mec="darkgreen",
        )

        axs[0, i].legend()
        axs[0, i].grid(True)
        axs[0, i].set_title(
            "AOI " + str(i + 1),
            fontdict={"color": COLORS[i + 1]},
        )
        axs[0, i].set_ylabel("PL (a.u.)")

    linestyles = [
        "--",
        "-.",
        (0, (1, 1)),
        (0, (5, 10)),
        (0, (5, 5)),
        (0, (5, 1)),
        (0, (3, 10, 1, 10)),
        (0, (3, 5, 1, 5)),
        (0, (3, 1, 1, 1)),
        (0, (3, 5, 1, 5, 1, 5)),
        (0, (3, 10, 1, 10, 1, 10)),
        (0, (3, 1, 1, 1, 1, 1)),
    ]

    for i in range(len(AOIs)):
        # plot subtraction norm
        axs[1, 0].plot(
            sweep_list,
            1 + sig_avgs[i] - ref_avgs[i],
            label="AOI " + str(i + 1),
            c=COLORS[i + 1],
            ls=linestyles[i],
            lw=1,
            marker="o",
            ms=3.5,
            mfc="w",
            mec=COLORS[i + 1],
        )
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        axs[1, 0].set_title("Subtraction Normalisation")
        axs[1, 0].set_xlabel("MW Frequency (MHz)")
        axs[1, 0].set_ylabel("PL (a.u.)")

        # plot division norm
        axs[1, 1].plot(
            sweep_list,
            sig_avgs[i] / ref_avgs[i],
            label="AOI " + str(i + 1),
            c=COLORS[i + 1],
            ls=linestyles[i],
            lw=1,
            marker="o",
            ms=3.5,
            mfc="w",
            mec=COLORS[i + 1],
        )

        axs[1, 1].legend()
        axs[1, 1].grid(True)
        axs[1, 1].set_title("Division Normalisation")
        axs[1, 1].set_xlabel("MW Frequency (MHz)")
        axs[1, 1].set_ylabel("PL (a.u.)")

    return fig


# ============================================================================


def plot_AOI_spectra_fit(
    options,
    sig,
    ref,
    sweep_list,
    AOIs,
    AOI_avg_best_fit_results_lst,
    roi_avg_fit_result,
    fit_model,
):
    # from PL version:
    # best_fit_result = fitting_results.x
    # fit_sweep_vector = np.linspace(np.min(sweep_vector), np.max(sweep_vector), 10000)
    # scipy_best_fit = fit_model(best_fit_result, fit_sweep_vector)
    # init_fit = fit_model(init_guess, fit_sweep_vector)

    # rows:
    # ROI avg, single pixel, then each AOI
    # columns:
    # sig & ref, sub & div norm, fit -> compared to ROI {raw, fit, ROI_avg_fit}

    figsize = list(options["figure_size"])
    figsize[0] *= 3  # number of columns
    figsize[1] *= 2 + len(AOIs)  # number of rows
    fig, axs = plt.subplots(2 + len(AOIs), 3, sharex=True, sharey=False, figsize=figsize)

    # pre-process raw data to plot
    sig_avgs = []
    ref_avgs = []
    # add roi data
    sz_h = options["additional_bins"] * int(options["metadata"]["AOIHeight"])
    sz_w = options["additional_bins"] * int(options["metadata"]["AOIWidth"])
    ROI = data_loading.define_ROI(options, sz_h, sz_w)
    roi_avg_sig = np.nansum(np.nansum(sig[:, ROI[0], ROI[1]], 2), 1)
    roi_avg_sig = roi_avg_sig / np.max(roi_avg_sig)
    roi_avg_ref = np.nansum(np.nansum(ref[:, ROI[0], ROI[1]], 2), 1)
    roi_avg_ref = roi_avg_ref / np.max(roi_avg_ref)
    sig_avgs.append(roi_avg_sig)
    ref_avgs.append(roi_avg_ref)
    # add single pixel check
    pixel_sig = sig[:, options["single_pixel_check"][0], options["single_pixel_check"][1]]
    pixel_sig = pixel_sig / np.max(pixel_sig)
    pixel_ref = ref[:, options["single_pixel_check"][0], options["single_pixel_check"][1]]
    pixel_ref = pixel_ref / np.max(pixel_ref)
    sig_avgs.append(pixel_sig)
    ref_avgs.append(pixel_ref)
    for i, AOI in enumerate(AOIs):
        sig_avg = np.nansum(np.nansum(sig[:, AOI[0], AOI[1]], 2), 1)
        sig_avg = sig_avg / np.max(sig_avg)
        ref_avg = np.nansum(np.nansum(ref[:, AOI[0], AOI[1]], 2), 1)
        ref_avg = ref_avg / np.max(ref_avg)
        sig_avgs.append(sig_avg)
        ref_avgs.append(ref_avg)

    # now pre-process fit params
    fit_param_lst = []

    # roi avg
    fit_param_lst.append(roi_avg_fit_result.best_fit_result)

    # single pixel
    if options["normalisation"] == "div":
        pixel_pl_ar = pixel_sig / pixel_ref
    elif options["normalisation"] == "sub":
        pixel_pl_ar = pixel_sig - pixel_ref
    else:
        RuntimeError(f"Not sure what normalisation value {options['normalisation']} is?")
    pixel_fit_params = fitting.fit_single_pixel(
        options, pixel_pl_ar, sweep_list, fit_model, roi_avg_fit_result
    )
    fit_param_lst.append(pixel_fit_params)

    # aois
    for AOI_best_fit_result in AOI_avg_best_fit_results_lst:
        fit_param_lst.append(AOI_best_fit_result)

    # plot sig, ref data as first column
    for i, (sig, ref) in enumerate(zip(sig_avgs, ref_avgs)):

        # plot sig
        axs[i, 0].plot(
            sweep_list,
            sig,
            label="sig",
            c="blue",
            ls="dashed",
            lw=1,
            marker="o",
            ms=3.5,
            mfc="cornflowerblue",
            mec="mediumblue",
        )
        # plot ref
        axs[i, 0].plot(
            sweep_list,
            ref,
            label="ref",
            c="green",
            ls="dashed",
            lw=1,
            marker="o",
            ms=3.5,
            mfc="limegreen",
            mec="darkgreen",
        )

        axs[i, 0].legend()
        axs[i, 0].grid(True)
        if not i:
            axs[i, 0].set_title("ROI avg")
        elif i == 1:
            axs[i, 0].set_title("Single Pixel Check", fontdict={"color": COLORS[0]})
        else:
            axs[i, 0].set_title("AOI " + str(i - 1), fontdict={"color": COLORS[i - 1]})
        axs[i, 0].set_ylabel("PL (a.u.)")
    axs[-1, 0].set_xlabel("MW Frequency (MHz)")

    # plot normalisation as second column
    for i, (sig, ref) in enumerate(zip(sig_avgs, ref_avgs)):
        axs[i, 1].plot(
            sweep_list,
            1 + sig - ref,
            label="subtraction",
            c="firebrick",
            ls="dashed",
            lw=1,
            marker="o",
            ms=3.5,
            mfc="lightcoral",
            mec="maroon",
        )
        axs[i, 1].plot(
            sweep_list,
            sig / ref,
            label="division",
            c="cadetblue",
            ls="dashed",
            lw=1,
            marker="o",
            ms=3.5,
            mfc="powderblue",
            mec="darkslategrey",
        )

        axs[i, 1].legend()
        axs[i, 1].grid(True)
        if not i:
            axs[i, 1].set_title("ROI avg - Normalisation")
        elif i == 1:
            axs[i, 1].set_title(
                "Single Pixel Check - Normalisation", fontdict={"color": COLORS[0]}
            )
        else:
            axs[i, 1].set_title(
                "AOI " + str(i - 1) + " - Normalisation", fontdict={"color": COLORS[i - 1]}
            )
        axs[i, 1].set_ylabel("PL (a.u.)")
    axs[-1, 1].set_xlabel("MW Frequency (MHz)")  # this is meant to be less indented

    # plot fits as third column
    fit_sweep_vector = np.linspace(np.min(sweep_list), np.max(sweep_list), 10000)
    roi_avg_best_fit_ar = fit_model(roi_avg_fit_result.best_fit_result, fit_sweep_vector)

    for i, (fit_param_ar, sig, ref) in enumerate(zip(fit_param_lst, sig_avgs, ref_avgs)):
        if options["normalisation"] == "div":
            sig_norm = sig / ref
        elif options["normalisation"] == "sub":
            sig_norm = sig - ref

        best_fit_ar = fit_model(fit_param_ar, fit_sweep_vector)

        # raw data
        axs[i, 2].plot(
            sweep_list,
            sig_norm,
            label="raw data",
            ls="",
            marker="o",
            ms=3.5,
            mfc="goldenrod",
            mec="k",
        )
        # best fit
        axs[i, 2].plot(fit_sweep_vector, best_fit_ar, label="fit", ls="dashed", lw=1, c="indigo")
        # roi avg fit (as comparison)
        if i:
            axs[i, 2].plot(
                fit_sweep_vector,
                roi_avg_best_fit_ar,
                label="ROI avg fit",
                c="crimson",
                ls="dashed",
                lw=1,
            )
        if not i:
            axs[i, 2].set_title("ROI avg - Fit")
        elif i == 1:
            axs[i, 2].set_title("Single Pixel Check - Fit", fontdict={"color": COLORS[0]})
        else:
            axs[i, 2].set_title("AOI " + str(i - 1) + " - Fit", fontdict={"color": COLORS[i - 1]})

        axs[i, 2].legend()
        axs[i, 2].grid(True)
        axs[i, 2].set_ylabel("PL (a.u.)")
    axs[-1, 2].set_xlabel("MW Frequency (MHz)")  # this is meant to be less indented

    return fig


# ============================================================================


def plot_param_image(options, fit_model, fit_results, param_name, param_number=0):
    # get plotting options from somewhere...

    image = fit_results[param_name + "_" + str(param_number)]
    c_map = "viridis"
    c_range = [np.nanmin(image), np.nanmax(image)]
    c_label = fit_models.get_param_unit(fit_model, param_name, param_number)

    fig, ax = plot_image(
        image,
        param_name + "_" + str(param_number),
        c_map,
        c_range,
        c_label,
        list(options["figure_size"]),
        None,
    )


# TODO now write something to plot multiple images together

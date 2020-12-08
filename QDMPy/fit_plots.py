# -*- coding: utf-8 -*-

__author__ = "Sam Scholten"

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import numpy as np
import matplotlib.patches as patches

import systems


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
        plt.text(
            area_c[0] + 0.95 * area_size,  # label posn.: top right, 5% from square corner
            area_c[1] + 0.05 * area_size,
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
        plt.text(
            rect_corner_x + 0.95 * size_x,  # label posn.: top right, 5% from rectangle corner
            rect_corner_y + 0.05 * size_y,
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

    edgecolors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:purple",
        "tab:pink",
        "tab:olive",
        "tab:cyan",
        "tab:brown",
        "tab:gray",
        "tab:red",
    ]

    i = 0
    while True:
        i += 1
        try:
            centre = options["area_" + str(i) + "_centre"] * binning
            size = options["area_" + str(i) + "_size"] * binning

            corner = [
                centre[0] - size / 2,
                centre[1] - size / 2,
            ]

            add_patch_rect(
                ax,
                corner[0],
                corner[1],
                size,
                size,
                label="AOI " + str(i),
                edgecolor=edgecolors[i],
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

    return fig


# ============================================================================


def plot_param_image(options, fit_model, fit_results, param_name):
    # get plotting options from somewhere...

    image = fit_results[param_name]
    c_map = "viridis"
    c_range = [np.nanmin(image), np.nanmax(image)]
    # c_label = fit_model.fn_chain.parameter_unit[parameter_key]
    c_label = None  # TODO

    fig, ax = plot_image(
        image, param_name, c_map, c_range, c_label, list(options["figure_size"]), None
    )

    # plot_image(image_data, title, c_map, c_range, c_label, figure_size, pixel_size):
    pass

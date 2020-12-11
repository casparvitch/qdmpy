# -*- coding: utf-8 -*-
"""
Module docstring
"""

# ============================================================================

__author__ = "Sam Scholten"

# ============================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import matplotlib.patches as patches
import math
import warnings

# ============================================================================

import QDMPy.systems as systems
import QDMPy.fit_models as fit_models
import QDMPy.fitting as fitting
import QDMPy.data_loading as data_loading
import QDMPy.misc as misc

# ============================================================================

"""
NOTES

- MHz has been hardcoded in some places for the plots here, how to generalise?
"""

# ===========================================================================


def set_mpl_rcparams(options):
    for optn, val in options["mpl_rcparams"].items():
        if type(val) == list:
            val = tuple(val)
        mpl.rcParams[optn] = val


# ===========================================================================


def plot_ROI_PL_image(options, PL_image):
    c_map = options["colormaps"]["PL_images"]
    c_range = get_colormap_range(options["colormap_range_dicts"]["PL_images"], PL_image)

    fig, ax = plot_image(options, PL_image, "PL - ROI", c_map, c_range, "Counts", None)

    if options["show_scalebar"]:
        pixel_size = options["system"].get_raw_pixel_size() * options["total_bin"]
        scalebar = ScaleBar(pixel_size)
        ax.add_artist(scalebar)

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
    tick_locator = mpl.ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.linewidth = 0.5
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
    add_patch_rect(
        ax, corner[0], corner[1], size, size, label="PX check", edgecolor=options["AOI_colors"][0]
    )

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
                edgecolor=options["AOI_colors"][i],
            )
        except KeyError:
            break


# ============================================================================


def plot_AOI_PL_images(options, PL_image_ROI, AOIs):
    # here plot image_ROI cut down, label AOIs on it
    if AOIs == []:
        return None

    # need to convert back to um? TODO implement some sort of scalebar
    # pixel_size = options["system"].get_raw_pixel_size() * options["total_bin"]

    c_map = options["colormaps"]["PL_images"]
    c_range = get_colormap_range(options["colormap_range_dicts"]["PL_images"], PL_image_ROI)

    fig, ax = plot_image(
        options,
        PL_image_ROI,
        "PL - AOIs",
        c_map,
        c_range,
        "Counts",
        None,
    )
    if options["show_scalebar"]:
        pixel_size = options["system"].get_raw_pixel_size() * options["total_bin"]
        scalebar = ScaleBar(pixel_size)
        ax.add_artist(scalebar)

    if options["annotate_image_regions"]:
        annotate_AOI_PL_image(options, ax)

    # TODO implement elliptical/circular mask feature
    # --> make object that can be passed around, saved/pickled after being tested etc.

    return fig


# ============================================================================


def plot_image(options, image_data, title, c_map, c_range, c_label, pixel_size):

    fig, ax = plt.subplots(constrained_layout=True)

    fig, ax = plot_image_on_ax(
        fig, ax, options, image_data, title, c_map, c_range, c_label, pixel_size
    )

    np.savetxt(options["data_dir"] / f"{title}.txt", image_data)
    if options["save_plots"]:
        fig.savefig(options["output_dir"] / (f"{title}." + options["save_fig_type"]))

    return fig, ax


# ============================================================================


def plot_image_on_ax(fig, ax, options, image_data, title, c_map, c_range, c_label, pixel_size):

    im = ax.imshow(image_data, cmap=c_map, vmin=c_range[0], vmax=c_range[1])

    ax.set_title(title)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    cbar = add_colourbar(im, fig, ax)
    cbar.ax.set_ylabel(c_label, rotation=270)

    if options["show_scalebar"]:
        pixel_size = options["system"].get_raw_pixel_size() * options["total_bin"]
        scalebar = ScaleBar(pixel_size)
        ax.add_artist(scalebar)

    return fig, ax


# ============================================================================


def plot_ROI_avg_fit(options, roi_avg_fit_result):
    res = roi_avg_fit_result

    fig = plt.figure(constrained_layout=False)  # constrained doesn't work well here
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
        c="black",
        marker="o",
        mfc="w",
        mec="k",
    )
    residual_frame.legend()
    residual_frame.grid()
    residual_frame.set_xlabel("Sweep parameter")

    roi_avg_fit_result.savejson("ROI_avg_fit.json", options["data_dir"])
    if options["save_plots"]:
        fig.savefig(options["output_dir"] / ("ROI_avg_fit." + options["save_fig_type"]))

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

    figsize = mpl.rcParams["figure.figsize"].copy()
    figsize[0] *= len(AOIs)
    figsize[1] *= 2
    fig, axs = plt.subplots(
        2, len(AOIs), figsize=figsize, sharex=True, sharey=False, constrained_layout=True
    )

    for i, AOI in enumerate(AOIs):

        # plot sig
        axs[0, i].plot(
            sweep_list,
            sig_avgs[i],
            label="sig",
            c="blue",
            ls="dashed",
            marker="o",
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
            marker="o",
            mfc="limegreen",
            mec="darkgreen",
        )

        axs[0, i].legend()
        axs[0, i].grid(True)
        axs[0, i].set_title(
            "AOI " + str(i + 1),
            fontdict={"color": options["AOI_colors"][i + 1]},
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
            c=options["AOI_colors"][i + 1],
            ls=linestyles[i],
            marker="o",
            mfc="w",
            mec=options["AOI_colors"][i + 1],
        )
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        axs[1, 0].set_title("Subtraction Normalisation")
        axs[1, 0].set_xlabel("Sweep parameter")
        axs[1, 0].set_ylabel("PL (a.u.)")

        # plot division norm
        axs[1, 1].plot(
            sweep_list,
            sig_avgs[i] / ref_avgs[i],
            label="AOI " + str(i + 1),
            c=options["AOI_colors"][i + 1],
            ls=linestyles[i],
            marker="o",
            mfc="w",
            mec=options["AOI_colors"][i + 1],
        )

        axs[1, 1].legend()
        axs[1, 1].grid(True)
        axs[1, 1].set_title("Division Normalisation")
        axs[1, 1].set_xlabel("Sweep parameter")
        axs[1, 1].set_ylabel("PL (a.u.)")

    # delete axes that we didn't use
    for i in range(len(AOIs)):
        if i < 2:  # we used these
            continue
        else:  # we didn't use these
            fig.delaxes(axs[1, i])

    output_dict = {}
    for i in range(len(AOIs)):
        output_dict["AOI_sig_avg" + "_" + str(i + 1)] = sig_avgs[i]
        output_dict["AOI_ref_avg" + "_" + str(i + 1)] = ref_avgs[i]

    misc.dict_to_json(output_dict, "AOI_spectra.json", options["data_dir"])

    if options["save_plots"]:
        fig.savefig(options["output_dir"] / ("AOI_spectra." + options["save_fig_type"]))
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
    # rows:
    # ROI avg, single pixel, then each AOI
    # columns:
    # sig & ref, sub & div norm, fit -> compared to ROI {raw, fit, ROI_avg_fit}

    figsize = mpl.rcParams["figure.figsize"].copy()
    figsize[0] *= 3  # number of columns
    # figsize[0] *= 2  # the above was too big
    figsize[1] *= 2 + len(AOIs)  # number of rows

    fig, axs = plt.subplots(
        2 + len(AOIs), 3, figsize=figsize, sharex=True, sharey=False, constrained_layout=True
    )

    # pre-process raw data to plot
    sig_avgs = []
    ref_avgs = []
    # add roi data
    sz_h = int(options["metadata"]["AOIHeight"] / options["additional_bins"])
    sz_w = int(options["metadata"]["AOIWidth"] / options["additional_bins"])
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
    if not options["used_ref"]:
        pixel_pl_ar = pixel_sig
    elif options["normalisation"] == "div":
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
            marker="o",
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
            marker="o",
            mfc="limegreen",
            mec="darkgreen",
        )

        axs[i, 0].legend()
        axs[i, 0].grid(True)
        if not i:
            axs[i, 0].set_title("ROI avg")
        elif i == 1:
            axs[i, 0].set_title("Single Pixel Check", fontdict={"color": options["AOI_colors"][0]})
        else:
            axs[i, 0].set_title(
                "AOI " + str(i - 1) + " avg", fontdict={"color": options["AOI_colors"][i - 1]}
            )
        axs[i, 0].set_ylabel("PL (a.u.)")
    axs[-1, 0].set_xlabel("Sweep parameter")

    # plot normalisation as second column
    for i, (sig, ref) in enumerate(zip(sig_avgs, ref_avgs)):
        axs[i, 1].plot(
            sweep_list,
            1 + sig - ref,
            label="subtraction",
            c="firebrick",
            ls="dashed",
            marker="o",
            mfc="lightcoral",
            mec="maroon",
        )
        axs[i, 1].plot(
            sweep_list,
            sig / ref,
            label="division",
            c="cadetblue",
            ls="dashed",
            marker="o",
            mfc="powderblue",
            mec="darkslategrey",
        )

        axs[i, 1].legend()
        axs[i, 1].grid(True)
        if not i:
            axs[i, 1].set_title("ROI avg - Normalisation")
        elif i == 1:
            axs[i, 1].set_title(
                "Single Pixel Check - Normalisation", fontdict={"color": options["AOI_colors"][0]}
            )
        else:
            axs[i, 1].set_title(
                "AOI " + str(i - 1) + " avg - Normalisation",
                fontdict={"color": options["AOI_colors"][i - 1]},
            )
        axs[i, 1].set_ylabel("PL (a.u.)")
    axs[-1, 1].set_xlabel(
        "Sweep parameter"
    )  # this is meant to be less indentednormalisationnormalisation

    # plot fits as third column
    fit_sweep_vector = np.linspace(np.min(sweep_list), np.max(sweep_list), 10000)
    roi_avg_best_fit_ar = fit_model(roi_avg_fit_result.best_fit_result, fit_sweep_vector)

    for i, (fit_param_ar, sig, ref) in enumerate(zip(fit_param_lst, sig_avgs, ref_avgs)):
        if not options["used_ref"]:
            sig_norm = sig
        elif options["normalisation"] == "div":
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
        axs[i, 2].plot(fit_sweep_vector, best_fit_ar, label="fit", ls="dashed", c="crimson")
        # roi avg fit (as comparison)
        if i:
            axs[i, 2].plot(
                fit_sweep_vector,
                roi_avg_best_fit_ar,
                label="ROI avg fit",
                c="indigo",
                ls="dashed",
            )
        if not i:
            axs[i, 2].set_title("ROI avg - Fit")
        elif i == 1:
            axs[i, 2].set_title(
                "Single Pixel Check - Fit", fontdict={"color": options["AOI_colors"][0]}
            )
        else:
            axs[i, 2].set_title(
                "AOI " + str(i - 1) + " avg - Fit",
                fontdict={"color": options["AOI_colors"][i - 1]},
            )

        axs[i, 2].legend()
        axs[i, 2].grid(True)
        axs[i, 2].set_ylabel("PL (a.u.)")
    axs[-1, 2].set_xlabel("Sweep parameter")  # this is meant to be less indented

    # currently not saving any of the data from this plot (not sure what the user would ever want)

    if options["save_plots"]:
        fig.savefig(options["output_dir"] / ("AOI_spectra_fits." + options["save_fig_type"]))

    return fig


# ============================================================================


def plot_param_image(options, fit_model, pixel_fit_params, param_name, param_number=0):
    # get plotting options from somewhere...

    image = pixel_fit_params[param_name + "_" + str(param_number)]
    c_map = options["colormaps"]["param_images"]
    c_range = get_colormap_range(options["colormap_range_dicts"]["param_images"], image)
    c_label = fit_models.get_param_unit(fit_model, param_name, param_number)

    fig, ax = plot_image(
        options,
        image,
        param_name + "_" + str(param_number),
        c_map,
        c_range,
        c_label,
        None,
    )
    return fig


# ============================================================================


def plot_param_images(options, fit_model, pixel_fit_params, param_name):

    # plot 2 columns wide, as many rows as required

    # first get keys we need
    our_keys = []
    for key in pixel_fit_params:
        if key.startswith(param_name):
            our_keys.append(key)

    # this is an inner function so no one uses it elsewhere/protect namespace
    def param_sorter(param):
        param_name, num = param.split("_")
        return int(num)

    # sort based on number (just in case)
    our_keys.sort(key=param_sorter)

    if len(our_keys) == 1:
        # just one image, so plot normally
        fig = plot_param_image(options, fit_model, pixel_fit_params, param_name, 0)
    else:
        num_columns = 2
        num_rows = math.ceil(len(our_keys) / 2)

        figsize = mpl.rcParams["figure.figsize"].copy()
        figsize[0] *= num_columns
        figsize[1] *= num_rows

        fig, axs = plt.subplots(
            num_rows,
            num_columns,
            figsize=figsize,
            sharex=False,
            sharey=False,
            constrained_layout=True,
        )

        c_map = options["colormaps"]["param_images"]

        # plot 8-lorentzian peaks in a more helpful way (pairs: 0-7, 1-6, etc.)
        if len(our_keys) == 8 and "lorentzian" in options["fit_functions"]:
            param_axis_iterator = zip([0, 7, 1, 6, 2, 5, 3, 4], axs.flatten())
        # otherwise plot in a more conventional order
        else:
            param_axis_iterator = enumerate(axs.flatten())

        for param_number, ax in param_axis_iterator:

            param_key = param_name + "_" + str(param_number)

            try:
                image_data = pixel_fit_params[param_key]
            except KeyError:
                # we have one too many axes (i.e. 7 params, 8 subplots), delete the axs
                fig.delaxes(ax)  # UNTESTED

            c_range = get_colormap_range(
                options["colormap_range_dicts"]["param_images"], image_data
            )
            c_label = fit_models.get_param_unit(fit_model, param_name, param_number)

            plot_image_on_ax(
                fig,
                ax,
                options,
                image_data,
                param_key,
                c_map,
                c_range,
                c_label,
                None,
            )

            np.savetxt(options["data_dir"] / f"{param_key}.txt", image_data)

        if options["save_plots"]:
            fig.savefig(options["output_dir"] / (param_name + "." + options["save_fig_type"]))


# ============================================================================


def get_colormap_range(c_range_dict, image):

    # mostly these are just checking that the input values are valid
    # pretty badly written, I apoligise (there's a reason it's hidden all the way down here...)

    warning_messages = {
        "deviation_from_mean": "Invalid c_range_dict['vals'] encountered. For c_range type 'deviation_from_mean', c_range_dict['vals'] must be a float, between 0 and 1. Changing to 'min_max_symmetric_about_mean' c_range.",
        "strict_range": "Invalid c_range_dict['vals'] encountered. For c_range type 'strict_range', c_range_dict['vals'] must be a float, between 0 and 1. Changing to 'min_max_symmetric_about_mean' c_range.",
        "percentile": "Invalid c_range_dict['vals'] encountered. For c_range type 'percentile', c_range_dict['vals'] must be a list of length 2, with elements (preferably ints) between 0 and 100. Changing to 'min_max_symmetric_about_mean' c_range.",
        "percentile_symmetric_about_zero": "Invalid c_range_dict['vals'] encountered. For c_range type 'percentile', c_range_dict['vals'] must be a list of length 2, with elements (preferably ints) between 0 and 100. Changing to 'min_max_symmetric_about_mean' c_range.",
    }

    c_range_type = c_range_dict["type"]
    if "values" in c_range_dict:
        c_range_values = c_range_dict["values"]
    else:
        c_range_values = None

    range_calculator_dict = {
        "min_max": min_max,
        "deviation_from_mean": deviation_from_mean,
        "min_max_symmetric_about_mean": min_max_sym_mean,
        "min_max_symmetric_about_zero": min_max_sym_zero,
        "percentile": percentile,
        "percentile_symmetric_about_zero": percentile_sym_zero,
        "strict_range": strict_range,
    }

    if c_range_type in ["deviation_from_mean", "strict_range"]:
        if type(c_range_values) != list or len(c_range_values) != 2:
            warnings.warn(warning_messages[c_range_type])
            return min_max_sym_mean(image, c_range_values)

    elif c_range_type.startswith("percentile"):
        if (
            type(c_range_values) != list
            or len(c_range_values) != 2  # noqa: W503
            or c_range_values[0] < 0  # noqa: W503
            or c_range_values[0] >= 100  # noqa: W503
            or c_range_values[1] < 0  # noqa: W503
            or c_range_values[1] >= 100  # noqa: W503
        ):
            warnings.warn(warning_messages[c_range_type])
            return min_max_sym_mean(image, c_range_values)

    else:
        return range_calculator_dict[c_range_type](image, c_range_values)


# ============================


def min_max(image, c_range_values):
    return [np.nanmin(image), np.nanmax(image)]


def strict_range(image, c_range_values):
    return list(c_range_values)


def min_max_sym_mean(image, c_range_values):
    minimum = np.nanmin(image)
    maximum = np.nanmax(image)
    mean = np.mean(image)
    max_distance_from_mean = np.max([abs(maximum - mean), abs(minimum - mean)])
    return [mean - max_distance_from_mean, mean + max_distance_from_mean]


def min_max_sym_zero(image, c_range_values):
    min_abs = np.abs(np.nanmin(image))
    max_abs = np.abs(np.nanmax(image))
    larger = np.nanmax([min_abs, max_abs])
    return [-larger, larger]


def deviation_from_mean(image, c_range_values):
    return ([(1 - c_range_values) * np.mean(image), (1 + c_range_values) * np.mean(image)],)


def percentile(image, c_range_values):
    return [np.nanpercentile(image, c_range_values)]


def percentile_sym_zero(image, c_range_values):
    plow, phigh = np.nanpercentile(image, c_range_values)  # e.g. [10, 90]
    val = max(abs(plow), abs(phigh))
    return [-val, val]


# ============================================================================

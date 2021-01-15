# - coding: utf-8 -*-
"""
This module holds functions for plotting initial processing images and fit results.

Functions
---------
 - `QDMPy.fit_plots.set_mpl_rcparams`
 - `QDMPy.fit_plots.plot_ROI_PL_image`
 - `QDMPy.fit_plots.add_colorbar`
 - `QDMPy.fit_plots.add_patch_square_centre`
 - `QDMPy.fit_plots.add_patch_rect`
 - `QDMPy.fit_plots.annotate_ROI_image`
 - `QDMPy.fit_plots.annotate_AOI_image`
 - `QDMPy.fit_plots.plot_AOI_PL_images`
 - `QDMPy.fit_plots.plot_image`
 - `QDMPy.fit_plots.plot_image_on_ax`
 - `QDMPy.fit_plots.plot_ROI_avg_fits`
 - `QDMPy.fit_plots.plot_AOI_spectra`
 - `QDMPy.fit_plots.plot_AOI_spectra_fit`
 - `QDMPy.fit_plots.plot_param_image`
 - `QDMPy.fit_plots.plot_param_images`
 - `QDMPy.fit_plots.get_colormap_range`
 - `QDMPy.fit_plots.min_max`
 - `QDMPy.fit_plots.strict_range`
 - `QDMPy.fit_plots.min_max_sym_mean`
 - `QDMPy.fit_plots.min_max_sym_zero`
 - `QDMPy.fit_plots.deviation_from_mean`
 - `QDMPy.fit_plots.percentile`
 - `QDMPy.fit_plots.percentile_sym_zero`
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

import QDMPy.data_loading as data_loading
import QDMPy.misc as misc


# ===========================================================================


def set_mpl_rcparams(options):
    """Reads matplotlib-relevant parameters in options and used to define matplotlib rcParams"""
    for optn, val in options["mpl_rcparams"].items():
        if type(val) == list:
            val = tuple(val)
        mpl.rcParams[optn] = val


# ===========================================================================


def plot_ROI_PL_image(options, PL_image):
    """
    Plots full PL image with ROI region annotated.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    PL_image : np array, 2D.
        Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned but
        not cut down to ROI.

    Returns
    -------
    fig : matplotlib Figure object
    """

    c_map = options["colormaps"]["PL_images"]
    c_range = get_colormap_range(options["colormap_range_dicts"]["PL_images"], PL_image)

    fig, ax = plot_image(options, PL_image, "PL - ROI", c_map, c_range, "Counts", None)

    if options["show_scalebar"]:
        pixel_size = options["system"].get_raw_pixel_size() * options["total_bin"]
        scalebar = ScaleBar(pixel_size)
        ax.add_artist(scalebar)

    if options["annotate_image_regions"]:
        annotate_ROI_image(options, ax)

    return fig


# ============================================================================


def add_colorbar(im, fig, ax, aspect=20, pad_fraction=1, **kwargs):
    """
    Adds a colorbar to matplotlib axis

    Arguments
    ---------
    im : image as returned by ax.imshow

    fig : matplotlib Figure object

    ax : matplotlib Axis object


    Optional arguments
    ------------------
    aspect : int
        Reciprocal of aspect ratio passed to new colorbar axis width. Default: 20.

    pad_fraction : int
        Fraction of new colorbar axis width to pad from image. Default: 1.

    **kwargs : other keyword arguments
        Passed to fig.colorbar.

    Returns
    -------
    cbar : matplotlib colorbar object
    """
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
    """
    Annotates square onto image figure.

    Arguments
    ---------
    ax : matplotlib Axis object

    area_c : int
        Location of centre of area you want to annotate.

    area_size : int
        Size of area you want to annotate.


    Optional arguments
    ------------------
    label : str
        Text to label annotated square with. Color is defined by edgecolor. Default: None.

    edgecolor : str
        Color of label and edge of annotation. Default: "b".

    Returns
    -------
    Nothing.
    """
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
    """
    Same as `QDMPy.fit_plots.add_patch_square_centre` but a rectangular annotation.

    Arguments
    ---------
    ax : matplotlib Axis object

    rect_corner_x : int
        Location of top left corner of area you want to annotate, x component.

    rect_corner_y : int
        Location of top left corner of area you want to annotate, y component.

    size_x : int
        Size of area along x (horizontal axis) you want to annotate.
    size_y : int
        Size of area along y (vertical) axis you want to annotate.

    Optional arguments
    ------------------
    label : str
        Text to label annotated square with. Color is defined by edgecolor. Default: None.

    edgecolor : str
        Color of label and edge of annotation. Default: "b".

    Returns
    -------
    Nothing.
    """
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
    """
    Annotates ROI onto a given Axis object. Generally used on a PL image.
    """
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


def annotate_AOI_image(options, ax):
    """
    Annotates AOI onto a given Axis object. Generally used on PL image.
    """
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
    """
    Plots PL image cut down to ROI, with annotated AOI regions.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    PL_image_ROI : np array, 2D.
        Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and
        cut down to ROI.

    AOIs : list
        List of AOI regions. Much like ROI object, these are a length-2 list of np meshgrids
        that can be used to directly index into image to provide a view into just the AOI
        part of the image. E.g. sig_AOI = sig[:, AOI[0], AOI[1]]. Returns a list as in
        general we have more than one area of interest.
        I.e. sig_AOI_1 = sig[:, AOIs[1][0], AOIs[1][1]]

    Returns
    -------
    fig : matplotlib Figure object
    """
    if AOIs == []:
        return None

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
        annotate_AOI_image(options, ax)

    return fig


# ============================================================================


def plot_image(options, image_data, title, c_map, c_range, c_label, pixel_size):
    """
    Plots an image given by image_data. Saves image_data as txt file as well as the figure.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    image_data : np array, 3D
        Data that is plot.

    title : str
        Title of figure, as well as name for save files

    c_map : str
        Colormap object used to map image_data values to a color.

    c_range : str
        Range of values in image_data to map to colors

    c_label : str
        Label for colormap axis

    pixel_size : str
        Size of each pixel in metres, used to define a scalebar.

    Returns
    -------
    fig : matplotlib Figure object

    ax : matplotlib Axis object
    """

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
    """
    Plots an image given by image_data onto given figure and ax.

    Does not save any data.

    Arguments
    ---------
    fig : matplotlib Figure object

    ax : matplotlib Axis object

    options : dict
        Generic options dict holding all the user options.

    image_data : np array, 3D
        Data that is plot.

    title : str
        Title of figure, as well as name for save files

    c_map : str
        Colormap object used to map image_data values to a color.

    c_range : str
        Range of values in image_data to map to colors

    c_label : str
        Label for colormap axis

    pixel_size : str
        Size of each pixel in metres, used to define a scalebar.

    Returns
    -------
    fig : matplotlib Figure object

    ax : matplotlib Axis object
    """

    im = ax.imshow(image_data, cmap=c_map, vmin=c_range[0], vmax=c_range[1])

    ax.set_title(title)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    cbar = add_colorbar(im, fig, ax)
    cbar.ax.set_ylabel(c_label, rotation=270)

    if options["show_scalebar"]:
        pixel_size = options["system"].get_raw_pixel_size() * options["total_bin"]
        scalebar = ScaleBar(pixel_size)
        ax.add_artist(scalebar)

    return fig, ax


# ============================================================================


def plot_ROI_avg_fits(options, backend_ROI_results_lst):
    """
    Plots fit of spectrum averaged across ROI, as well as corresponding residual values.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    backend_ROI_results_lst : list of tuples
        Format: (fit_backend, `QDMPy.fitting.FitResultROIAvg` objects), for each fit_backend

    Returns
    -------
    fig : matplotlib Figure object
    """

    fig = plt.figure(constrained_layout=False)  # constrained doesn't work well here
    # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    spectrum_frame = fig.add_axes((0.1, 0.3, 0.8, 0.6))

    spectrum_frame.plot(
        backend_ROI_results_lst[0].sweep_list,
        backend_ROI_results_lst[0].pl_roi,
        label="raw data",
        ls=" ",
        marker="o",
        mfc="w",
        mec="firebrick",
    )
    high_res_sweep_list = np.linspace(
        np.min(backend_ROI_results_lst[0].sweep_list),
        np.max(backend_ROI_results_lst[0].sweep_list),
        10000,
    )
    high_res_init_fit = backend_ROI_results_lst[0].fit_model(
        backend_ROI_results_lst[0].init_param_guess, high_res_sweep_list
    )
    spectrum_frame.plot(
        high_res_sweep_list,
        high_res_init_fit,
        linestyle=(0, (1, 1)),
        label="init guess",
        c="darkgreen",
    )
    spectrum_frame.set_xticklabels([])  # remove from first frame
    spectrum_frame.grid()
    spectrum_frame.set_ylabel("PL (a.u.)")

    # residual plot
    residual_frame = fig.add_axes((0.1, 0.1, 0.8, 0.2))

    residual_frame.grid()
    residual_frame.set_xlabel("Sweep parameter")

    for res in backend_ROI_results_lst:

        # ODMR spectrum_frame
        high_res_best_fit = res.fit_model(res.best_params, high_res_sweep_list)

        spectrum_frame.plot(
            high_res_sweep_list,
            high_res_best_fit,
            linestyle="--",
            label=f"{res.fit_backend} best fit",
            c=options["fit_backend_colors"][res.fit_backend]["roifit_linecolor"],
        )

        residual_xdata = res.sweep_list
        residual_ydata = res.fit_model(res.best_params, res.sweep_list) - res.pl_roi

        residual_frame.plot(
            residual_xdata,
            residual_ydata,
            label=f"{res.fit_backend} residual",
            ls="dashed",
            c=options["fit_backend_colors"][res.fit_backend]["residual_linecolor"],
            marker="o",
            mfc="w",
            mec=options["fit_backend_colors"][res.fit_backend]["residual_linecolor"],
        )

        res.savejson(f"ROI_avg_fit_{res.fit_backend}.json", options["data_dir"])

    residual_frame.legend()
    spectrum_frame.legend()

    if options["save_plots"]:
        fig.savefig(
            options["output_dir"] / (f"ROI_avg_fit_{res.fit_backend}." + options["save_fig_type"])
        )

    return fig


# ============================================================================


def plot_AOI_spectra(options, AOIs, sig, ref, sweep_list):
    """
    Plots spectra from each AOI, as well as subtraction and division norms.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    AOIs : list
        List of AOI regions. Much like ROI object, these are a length-2 list of np meshgrids
        that can be used to directly index into image to provide a view into just the AOI
        part of the image. E.g. sig_AOI = sig[:, AOI[0], AOI[1]]. Returns a list as in
        general we have more than one area of interest.
        I.e. sig_AOI_1 = sig[:, AOIs[1][0], AOIs[1][1]]

    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, x, y]

    ref : np array, 3D
        Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, x, y]
    sweep_list : list
        List of sweep parameter values (with removed unwanted sweeps at start/end)

    Returns
    -------
    fig : matplotlib Figure object
    """

    # pre-process data to plot
    sig_avgs = []
    ref_avgs = []
    for i, AOI in enumerate(AOIs):
        sig_avg = np.nanmean(np.nanmean(sig[:, AOI[0], AOI[1]], axis=2), axis=1)
        ref_avg = np.nanmean(np.nanmean(ref[:, AOI[0], AOI[1]], axis=2), axis=1)
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
    fit_result_collection_lst,
    backend_ROI_results_lst,
    fit_model,
):
    """
    Plots sig and ref spectra, sub and div normalisation and fit for the ROI average, a single
    pixel, and each of the AOIs. All stacked on top of each other for comparison. The ROI
    average fit is plot against the fit of all of the others for comparison.

    Note here and elsewhere the single pixel check is the first element of the AOI array.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, x, y]

    ref : np array, 3D
        Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, x, y]
    sweep_list : list
        List of sweep parameter values (with removed unwanted sweeps at start/end)


    AOIs : list
        List of AOI regions. Much like ROI object, these are a length-2 list of np meshgrids
        that can be used to directly index into image to provide a view into just the AOI
        part of the image. E.g. sig_AOI = sig[:, AOI[0], AOI[1]]. Returns a list as in
        general we have more than one area of interest.
        I.e. sig_AOI_1 = sig[:, AOIs[1][0], AOIs[1][1]]

    fit_result_collection_lst : list
        List of `QDMPy.fit_shared.FitResultCollection` objects (one for each fit_backend)

    backend_ROI_results_lst : list of `fitting.FitResultROIAvg`
        `QDMPy.fitting.FitResultROIAvg` object, each element for each fit backend

    fit_model : `fit_models.FitModel` object.

    Returns
    -------
    fig : matplotlib Figure object
    """

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
    roi_avg_sig = np.nanmean(np.nanmean(sig[:, ROI[0], ROI[1]], axis=2), axis=1)
    roi_avg_ref = np.nanmean(np.nanmean(ref[:, ROI[0], ROI[1]], axis=2), axis=1)
    sig_avgs.append(roi_avg_sig)
    ref_avgs.append(roi_avg_ref)
    # add single pixel check
    pixel_sig = sig[:, options["single_pixel_check"][0], options["single_pixel_check"][1]]
    pixel_ref = ref[:, options["single_pixel_check"][0], options["single_pixel_check"][1]]
    sig_avgs.append(pixel_sig)
    ref_avgs.append(pixel_ref)
    # add AOI data
    for i, AOI in enumerate(AOIs):
        sig_avg = np.nanmean(np.nanmean(sig[:, AOI[0], AOI[1]], axis=2), axis=1)
        ref_avg = np.nanmean(np.nanmean(ref[:, AOI[0], AOI[1]], axis=2), axis=1)
        sig_avgs.append(sig_avg)
        ref_avgs.append(ref_avg)

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
    )  # this is meant to be less indented than the line above

    high_res_xdata = np.linspace(
        np.min(backend_ROI_results_lst[0].sweep_list),
        np.max(backend_ROI_results_lst[0].sweep_list),
        10000,
    )

    # plot fits as third column
    for fit_backend_number, fit_backend_fit_result in enumerate(fit_result_collection_lst):
        fit_backend_name = fit_backend_fit_result.fit_backend

        fit_params_lst = [
            fit_backend_fit_result.roi_avg_fit_result.best_params,
            fit_backend_fit_result.single_pixel_fit_result,
            *fit_backend_fit_result.AOI_fit_results_lst,
        ]

        for i, (fit_param_ar, sig, ref) in enumerate(zip(fit_params_lst, sig_avgs, ref_avgs)):
            if not options["used_ref"]:
                sig_norm = sig
            elif options["normalisation"] == "div":
                sig_norm = sig / ref
            elif options["normalisation"] == "sub":
                sig_norm = sig - ref

            best_fit_ydata = fit_model(fit_param_ar, high_res_xdata)
            roi_fit_ydata = fit_model(
                fit_backend_fit_result.roi_avg_fit_result.best_params, high_res_xdata
            )

            # first loop over -> plot raw data, add titles
            if not fit_backend_number:

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
                if not i:
                    axs[i, 2].set_title("ROI avg - Fit")
                elif i == 1:
                    axs[i, 2].set_title(
                        "Single Pixel Check - Fit",
                        fontdict={"color": options["AOI_colors"][0]},
                    )
                else:
                    axs[i, 2].set_title(
                        "AOI " + str(i - 1) + " avg - Fit",
                        fontdict={"color": options["AOI_colors"][i - 1]},
                    )
            # roi avg fit (as comparison)
            if i:
                axs[i, 2].plot(
                    high_res_xdata,
                    roi_fit_ydata,
                    label=f"ROI avg fit - {fit_backend_name}",
                    ls="dashed",
                    c=options["fit_backend_colors"][fit_backend_name]["aoi_roi_fit_linecolor"],
                )
            # best fit
            axs[i, 2].plot(
                high_res_xdata,
                best_fit_ydata,
                label=f"fit - {fit_backend_name}",
                ls="dashed",
                c=options["fit_backend_colors"][fit_backend_name]["aoi_best_fit_linecolor"],
            )

            axs[i, 2].legend()
            axs[i, 2].grid(True)
            axs[i, 2].set_ylabel("PL (a.u.)")

    axs[-1, 2].set_xlabel("Sweep parameter")  # this is meant to be less indented than line above

    # currently not saving any of the data from this plot (not sure what the user would ever want)

    if options["save_plots"]:
        fig.savefig(options["output_dir"] / ("AOI_spectra_fits." + options["save_fig_type"]))

    return fig


# ============================================================================


def plot_param_image(options, fit_model, pixel_fit_params, param_name, param_number=0):
    """
    Plots an image corresponding to a single parameter in pixel_fit_params.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    fit_model : `fit_models.FitModel` object.

    pixel_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.

    param_name : str
        Name of parameter you want to plot, e.g. 'fwhm'.


    Optional arguments
    ------------------
    param_number : int
        Which version of the parameter you want. I.e. there might be 8 independent parameters
        in the fit model called 'pos', each labeled 'pos_0', 'pos_1' etc. Default: 0.


    Returns
    -------
    fig : matplotlib Figure object
    """

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
    """
    Plots images for all independent versions of a single parameter type in pixel_fit_params.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    fit_model : `fit_models.FitModel` object.

    pixel_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.

    param_name : str
        Name of parameter you want to plot, e.g. 'fwhm'.

    Returns
    -------
    fig : matplotlib Figure object
    """

    # if no fit completed
    if pixel_fit_params is None:
        warnings.warn(
            "'pixel_fit_params' arg to function 'plot_param_images' is 'None'.\n"
            + "Probably no pixel fitting completed."  # noqa: W503
        )
        return None

    # plot 2 columns wide, as many rows as required

    # first get keys we need
    our_keys = []
    for key in pixel_fit_params:
        if key.startswith(param_name):
            our_keys.append(key)

    # this is an inner function so no one uses it elsewhere/protect namespace
    def param_sorter(param):
        strings = param.split("_")  # e.g. "amp_exp_2" -> ["amp", "exp", "2"]
        # all we need here is the number at the end actually
        # param_name_split = strings[:-1]  # list of 'words', e.g. ["amp", "exp"]
        num = strings[-1]  # grab the number e.g. "2
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
                break

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
    """
    Produce a colormap range to plot image from, using the options in c_range_dict.

    Arguments
    ---------
    c_range_dict : dict
        dictionary with key 'values', used to accompany some of the options below,
        as well as a 'type', one of :
         - "min_max" : map between minimum and maximum values in image.
         - "deviation_from_mean" : requires c_range_dict["values"] be a float
           between 0 and 1 'dev'. Maps between (1 - dev) * mean and (1 + dev) * mean.
         - "min_max_symmetric_about_mean" : map symmetrically about zero, capturing all values
           in image (default).
         - "min_max_symmetric_about_zero" : map symmetrically about zero, capturing all values
           in image.
         - "percentile" : requires c_range_dict["values"] be a list of two numbers between 0 and
           100. Maps the range between those percentiles of the data.
         - "percentile_symmetric_about_zero" : requires c_range_dict["values"] be a list of two
           numbers between 0 and 100. Maps symmetrically about zero, capturing all values between
           those percentile in the data (plus perhaps a bit more to ensure symmety)
         - "strict_range" : requires c_range_dict["values"] be an int of float. Maps colors
           between the values given.
        as well as accompanying 'values' key, used for some of the options below

    Returns
    -------
    c_range : list length 2
        i.e. [min value to map to a color, max value to map to a color]
    """

    # mostly these are just checking that the input values are valid
    # pretty badly written, I apoligise (there's a reason it's hidden all the way down here...)

    warning_messages = {
        "deviation_from_mean": """Invalid c_range_dict['vals'] encountered.
        For c_range type 'deviation_from_mean', c_range_dict['vals'] must be a float,
        between 0 and 1. Changing to 'min_max_symmetric_about_mean' c_range.""",
        "strict_range": """Invalid c_range_dict['vals'] encountered.
        For c_range type 'strict_range', c_range_dict['vals'] must be a a list of length 2,
        with elements that are floats or ints.
        Changing to 'min_max_symmetric_about_mean' c_range.""",
        "percentile": """Invalid c_range_dict['vals'] encountered.
        For c_range type 'percentile', c_range_dict['vals'] must be a list of length 2,
         with elements (preferably ints) between 0 and 100.
         Changing to 'min_max_symmetric_about_mean' c_range.""",
        "percentile_symmetric_about_zero": """Invalid c_range_dict['vals'] encountered.
        For c_range type 'percentile', c_range_dict['vals'] must be a list of length 2,
         with elements (preferably ints) between 0 and 100.
         Changing to 'min_max_symmetric_about_mean' c_range.""",
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

    if c_range_type == "strict_range":
        if (
            type(c_range_values) != list
            or len(c_range_values) != 2  # noqa: W503
            or (type(c_range_values[0]) != float and type(c_range_values[0]) != int)  # noqa: W503
            or (type(c_range_values[1]) != float and type(c_range_values[1]) != int)  # noqa: W503
            or c_range_values[0] > c_range_values[1]  # noqa: W503
        ):
            warnings.warn(warning_messages[c_range_type])
            return min_max_sym_mean(image, c_range_values)
    elif c_range_type == "deviation_from_mean":
        if (
            (type(c_range_values) != float and type(c_range_values) != float)
            or c_range_values < 0  # noqa: W503
            or c_range_values > 1  # noqa: W503
        ):
            warnings.warn(warning_messages[c_range_type])
            return min_max_sym_mean(image, c_range_values)

    elif c_range_type.startswith("percentile"):
        if (
            type(c_range_values) != list
            or len(c_range_values) != 2  # noqa: W503
            or (type(c_range_values[0]) != float and type(c_range_values[0]) != int)  # noqa: W503
            or (type(c_range_values[1]) != float and type(c_range_values[1]) != int)  # noqa: W503
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
    """
    Map between minimum and maximum values in image

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow

    c_range_values : unknown (depends on user settings)
        See `QDMPy.fit_plots.get_colormap_range`
    """
    return [np.nanmin(image), np.nanmax(image)]


def strict_range(image, c_range_values):
    """
    Map between c_range_values

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow

    c_range_values : unknown (depends on user settings)
        See `QDMPy.fit_plots.get_colormap_range`
    """
    return list(c_range_values)


def min_max_sym_mean(image, c_range_values):
    """
    Map symmetrically about mean, capturing all values in image.

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow

    c_range_values : unknown (depends on user settings)
        See `QDMPy.fit_plots.get_colormap_range`
    """
    minimum = np.nanmin(image)
    maximum = np.nanmax(image)
    mean = np.mean(image)
    max_distance_from_mean = np.max([abs(maximum - mean), abs(minimum - mean)])
    return [mean - max_distance_from_mean, mean + max_distance_from_mean]


def min_max_sym_zero(image, c_range_values):
    """
    Map symmetrically about zero, capturing all values in image.

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow

    c_range_values : unknown (depends on user settings)
        See `QDMPy.fit_plots.get_colormap_range`
    """
    min_abs = np.abs(np.nanmin(image))
    max_abs = np.abs(np.nanmax(image))
    larger = np.nanmax([min_abs, max_abs])
    return [-larger, larger]


def deviation_from_mean(image, c_range_values):
    """
    Map a (decimal) deviation from mean, i.e. between (1 - dev) * mean and (1 + dev) * mean

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow

    c_range_values : unknown (depends on user settings)
        See `QDMPy.fit_plots.get_colormap_range`
    """
    return ([(1 - c_range_values) * np.mean(image), (1 + c_range_values) * np.mean(image)],)


def percentile(image, c_range_values):
    """
    Maps the range between two percentiles of the data.

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow

    c_range_values : unknown (depends on user settings)
        See `QDMPy.fit_plots.get_colormap_range`
    """
    return [np.nanpercentile(image, c_range_values)]


def percentile_sym_zero(image, c_range_values):
    """
    Maps the range between two percentiles of the data, but ensuring symmetry about zero

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow

    c_range_values : unknown (depends on user settings)
        See `QDMPy.fit_plots.get_colormap_range`
    """
    plow, phigh = np.nanpercentile(image, c_range_values)  # e.g. [10, 90]
    val = max(abs(plow), abs(phigh))
    return [-val, val]


# ============================================================================

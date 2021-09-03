# - coding: utf-8 -*-
"""
This module holds functions for plotting initial processing images and fit results.

Functions
---------
 - `qdmpy.plot.pl.roi_pl_image`
 - `qdmpy.plot.pl.aoi_pl_image`
 - `qdmpy.plot.pl.roi_avg_fits`
 - `qdmpy.plot.pl.aoi_spectra`
 - `qdmpy.plot.pl.aoi_spectra_fit`
 - `qdmpy.plot.pl.pl_param_image`
 - `qdmpy.plot.pl.pl_param_images`
 - `qdmpy.plot.pl.pl_params_flattened`
 - `qdmpy.plot.pl.other_measurements`
 - `qdmpy.plot.pl._add_patch_rect`
 - `qdmpy.plot.pl._annotate_roi_image`
 - `qdmpy.plot.pl._annotate_aoi_image`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.plot.pl.roi_pl_image": True,
    "qdmpy.plot.pl.aoi_pl_image": True,
    "qdmpy.plot.pl.roi_avg_fits": True,
    "qdmpy.plot.pl.aoi_spectra": True,
    "qdmpy.plot.pl.aoi_spectra_fit": True,
    "qdmpy.plot.pl.pl_param_image": True,
    "qdmpy.plot.pl.pl_param_images": True,
    "qdmpy.plot.pl.pl_params_flattened": True,
    "qdmpy.plot.pl.other_measurements": True,
    "qdmpy.plot.pl._add_patch_rect": True,
    "qdmpy.plot.pl._annotate_roi_image": True,
    "qdmpy.plot.pl._annotate_aoi_image": True,
}

# ============================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.patches as patches
import math
import warnings
from pathlib import Path

# ============================================================================

import qdmpy.plot.common
import qdmpy.shared.json2dict
import qdmpy.shared.misc

# ===========================================================================


def roi_pl_image(options, pl_image):
    """
    Plots full pl image with ROI region annotated.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    pl_image : np array, 2D
        Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned but
        not cut down to ROI.

    Returns
    -------
    fig : matplotlib Figure object
    """

    c_map = options["colormaps"]["pl_images"]
    c_range = qdmpy.plot.common.get_colormap_range(
        options["colormap_range_dicts"]["pl_images"], pl_image
    )

    fig, ax = plt.subplots()

    fig, ax = qdmpy.plot.common.plot_image_on_ax(
        fig, ax, options, pl_image, "pl - Full & Rebinned", c_map, c_range, "Counts"
    )

    if options["annotate_image_regions"]:
        _annotate_roi_image(options, ax)

    if options["save_plots"]:
        fig.savefig(options["output_dir"] / ("pl_full_rebinned." + options["save_fig_type"]))

    return fig


# ============================================================================


def aoi_pl_image(options, pl_image_roi):
    """
    Plots pl image cut down to ROI, with annotated AOI regions.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    pl_image_roi : np array, 2D
        Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and
        cut down to ROI.

    Returns
    -------
    fig : matplotlib Figure object
    """
    fig, ax = plt.subplots()
    c_map = options["colormaps"]["pl_images"]
    c_range = qdmpy.plot.common.get_colormap_range(
        options["colormap_range_dicts"]["pl_images"], pl_image_roi
    )

    fig, ax = qdmpy.plot.common.plot_image_on_ax(
        fig, ax, options, pl_image_roi, "pl - ROI & Rebinned", c_map, c_range, "Counts"
    )

    if options["annotate_image_regions"]:
        _annotate_aoi_image(options, ax)

    if options["save_plots"]:
        fig.savefig(options["output_dir"] / ("pl_ROI_rebinned." + options["save_fig_type"]))

    return fig


# ============================================================================


def roi_avg_fits(options, backend_roi_results_lst):
    """
    Plots fit of spectrum averaged across ROI, as well as corresponding residual values.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    backend_roi_results_lst : list of tuples
        Format: (fit_backend, `qdmpy.pl.common.ROIAvgFitResult` objects), for each fit_backend

    Returns
    -------
    fig : matplotlib Figure object
    """
    figsize = mpl.rcParams["figure.figsize"].copy()
    figsize[0] *= 2
    figsize[1] *= 1.5

    fig = plt.figure(
        figsize=figsize, constrained_layout=False
    )  # constrained doesn't work well here?
    # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    spectrum_frame = fig.add_axes((0.1, 0.3, 0.8, 0.6))

    lspec_names = []
    lspec_lines = []

    spectrum_frame.plot(
        backend_roi_results_lst[0].sweep_list,
        backend_roi_results_lst[0].pl_roi,
        label=f"raw data ({options['normalisation']})",
        ls=" ",
        marker="o",
        mfc="w",
        mec="firebrick",
    )
    lspec_names.append(f"raw data ({options['normalisation']})")
    lspec_lines.append(Line2D([0], [0], ls=" ", marker="o", mfc="w", mec="firebrick"))

    high_res_sweep_list = np.linspace(
        np.min(backend_roi_results_lst[0].sweep_list),
        np.max(backend_roi_results_lst[0].sweep_list),
        10000,
    )
    high_res_init_fit = backend_roi_results_lst[0].fit_model(
        backend_roi_results_lst[0].init_param_guess, high_res_sweep_list
    )
    spectrum_frame.plot(
        high_res_sweep_list,
        high_res_init_fit,
        linestyle=(0, (1, 1)),
        label="init guess",
        c="darkgreen",
    )
    lspec_names.append("init guess")
    lspec_lines.append(Line2D([0], [0], linestyle=(0, (1, 1)), c="darkgreen"))

    spectrum_frame.set_xticklabels([])  # remove from first frame

    spectrum_frame.grid()
    spectrum_frame.set_ylabel("pl (a.u.)")

    # residual plot
    residual_frame = fig.add_axes((0.1, 0.1, 0.8, 0.2))

    lresid_names = []
    lresid_lines = []

    residual_frame.grid()

    residual_frame.set_xlabel("Sweep parameter")
    residual_frame.set_ylabel("Fit - data (a.u.)")

    for res in backend_roi_results_lst:

        # ODMR spectrum_frame
        high_res_best_fit = res.fit_model(res.best_params, high_res_sweep_list)

        spectrum_frame.plot(
            high_res_sweep_list,
            high_res_best_fit,
            linestyle="--",
            label=f"{res.fit_backend} best fit",
            c=options["fit_backend_colors"][res.fit_backend]["ROIfit_linecolor"],
        )
        lspec_names.append(f"{res.fit_backend} best fit")
        lspec_lines.append(
            Line2D(
                [0],
                [0],
                linestyle="--",
                c=options["fit_backend_colors"][res.fit_backend]["ROIfit_linecolor"],
            )
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
        lresid_names.append(f"{res.fit_backend} residual")
        lresid_lines.append(
            Line2D(
                [0],
                [0],
                ls="dashed",
                c=options["fit_backend_colors"][res.fit_backend]["residual_linecolor"],
                marker="o",
                mfc="w",
                mec=options["fit_backend_colors"][res.fit_backend]["residual_linecolor"],
            )
        )

        res.savejson(f"ROI_avg_fit_{res.fit_backend}.json", options["data_dir"])

    # https://jdhao.github.io/2018/01/23/matplotlib-legend-outside-of-axes/
    # https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/linestyles.html
    legend_names = lspec_names.copy()
    legend_names.extend(lresid_names)

    legend_lines = lspec_lines.copy()
    legend_lines.extend(lresid_lines)

    spectrum_frame.legend(
        legend_lines,
        legend_names,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        fontsize="medium",
        ncol=len(legend_names),
        borderaxespad=0,
        frameon=False,
    )
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # spectrum_frame.legend()

    if options["save_plots"]:
        fig.savefig(options["output_dir"] / ("ROI_avg_fits." + options["save_fig_type"]))

    return fig


# ============================================================================


def aoi_spectra(options, sig, ref, sweep_list):
    """
    Plots spectra from each AOI, as well as subtraction and division norms.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, y, x]
    ref : np array, 3D
        Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, y, x]
    sweep_list : list
        List of sweep parameter values (with removed unwanted sweeps at start/end)

    Returns
    -------
    fig : matplotlib Figure object
    """
    aois = qdmpy.shared.misc.define_aois(options)

    # pre-process data to plot
    sig_avgs = []
    ref_avgs = []
    for i, aoi in enumerate(aois):
        sig_avg = np.nanmean(np.nanmean(sig[:, aoi[0], aoi[1]], axis=2), axis=1)
        ref_avg = np.nanmean(np.nanmean(ref[:, aoi[0], aoi[1]], axis=2), axis=1)
        sig_avgs.append(sig_avg)
        ref_avgs.append(ref_avg)

    figsize = mpl.rcParams["figure.figsize"].copy()
    num_wide = 2 if len(aois) < 2 else len(aois)
    figsize[0] *= num_wide
    figsize[1] *= 2
    fig, axs = plt.subplots(
        2, num_wide, figsize=figsize, sharex=True, sharey=False
    )

    for i, aoi in enumerate(aois):

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
        axs[0, i].set_title("AOI " + str(i + 1), fontdict={"color": options["AOI_colors"][i + 1]})
        axs[0, i].set_ylabel("pl (a.u.)")

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

    for i in range(len(aois)):
        # plot subtraction norm
        axs[1, 0].plot(
            sweep_list,
            1 + (sig_avgs[i] - ref_avgs[i]) / (sig_avgs[i] + ref_avgs[i]),
            label="AOI " + str(i + 1),
            c=options["AOI_colors"][i + 1],
            ls=linestyles[i],
            marker="o",
            mfc="w",
            mec=options["AOI_colors"][i + 1],
        )
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        axs[1, 0].set_title(
            "Subtraction Normalisation (Michelson contrast, 1 + (sig - ref / sig + ref) )"
        )
        axs[1, 0].set_xlabel("Sweep parameter")
        axs[1, 0].set_ylabel("pl (a.u.)")

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
        axs[1, 1].set_title("Division Normalisation (Weber contrast, sig / ref)")
        axs[1, 1].set_xlabel("Sweep parameter")
        axs[1, 1].set_ylabel("pl (a.u.)")

    # delete axes that we didn't use
    for i in range(len(aois)):
        if i < len(
            options["system"].option_choices("normalisation")
        ):  # we used these (normalisation)
            continue
        else:  # we didn't use these
            fig.delaxes(axs[1, i])

    if len(aois) == 1:
        # again, didn't use
        fig.delaxes(axs[0, 1])

    output_dict = {}
    for i in range(len(aois)):
        output_dict["AOI_sig_avg" + "_" + str(i + 1)] = sig_avgs[i]
        output_dict["AOI_ref_avg" + "_" + str(i + 1)] = ref_avgs[i]

    qdmpy.shared.json2dict.dict_to_json(output_dict, "AOI_spectra.json", options["data_dir"])

    if options["save_plots"]:
        fig.savefig(options["output_dir"] / ("AOI_spectra." + options["save_fig_type"]))
    return fig


# ============================================================================


def aoi_spectra_fit(options, sig, ref, sweep_list, fit_result_collection_lst, fit_model):
    """
    Plots sig and ref spectra, sub and div normalisation and fit for the ROI average, a single
    pixel, and each of the AOIs. All stacked on top of each other for comparison. The ROI
    average fit is plot against the fit of all of the others for comparison.

    Note here and elsewhere the single pixel check is the first element of the AOI array.

    NOTE this could be faster if we passed in sig_norm as well (backwards-compat. issues tho)

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, y, x]
    ref : np array, 3D
        Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, y, x]
    sweep_list : list
        List of sweep parameter values (with removed unwanted sweeps at start/end)
    fit_result_collection_lst : list
        List of `qdmpy.pl.common.FitResultCollection` objects (one for each fit_backend)
        holding ROI, AOI fit results
    fit_model : `qdmpy.pl.model.FitModel`
        Model we're fitting to.

    Returns
    -------
    fig : matplotlib Figure object
    """

    # rows:
    # ROI avg, single pixel, then each AOI
    # columns:
    # sig & ref, sub & div norm, fit -> compared to ROI {raw, fit, ROI_avg_fit}

    aois = qdmpy.shared.misc.define_aois(options)

    figsize = mpl.rcParams["figure.figsize"].copy()
    figsize[0] *= 3  # number of columns
    figsize[1] *= 2 + len(aois)  # number of rows

    fig, axs = plt.subplots(
        2 + len(aois), 3, figsize=figsize, sharex=True, sharey=False
    )

    #  pre-process raw data to plot -> note some are not averaged yet (will check for this below)
    sigs = []
    refs = []
    sigs.append(sig)
    refs.append(ref)
    # add single pixel check
    pixel_sig = sig[:, options["single_pixel_check"][1], options["single_pixel_check"][0]]
    pixel_ref = ref[:, options["single_pixel_check"][1], options["single_pixel_check"][0]]
    sigs.append(pixel_sig)
    refs.append(pixel_ref)
    # add AOI data
    for i, aoi in enumerate(aois):
        aoi_sig = sig[:, aoi[0], aoi[1]]
        aoi_ref = ref[:, aoi[0], aoi[1]]
        sigs.append(aoi_sig)
        refs.append(aoi_ref)

    # plot sig, ref data as first column
    for i, (s, r) in enumerate(zip(sigs, refs)):
        if len(s.shape) > 1:
            s_avg = np.nanmean(np.nanmean(s, axis=2), axis=1)
            r_avg = np.nanmean(np.nanmean(r, axis=2), axis=1)
        else:
            s_avg = s
            r_avg = r
        # plot sig
        axs[i, 0].plot(
            sweep_list,
            s_avg,
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
            r_avg,
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
        axs[i, 0].set_ylabel("pl (a.u.)")
    axs[-1, 0].set_xlabel("Sweep parameter")

    # plot normalisation as second column
    for i, (s, r) in enumerate(zip(sigs, refs)):
        sub = 1 + (s - r) / (s + r)
        div = s / r

        if len(sub.shape) > 1:
            sub_avg = np.nanmean(np.nanmean(sub, axis=2), axis=1)
            div_avg = np.nanmean(np.nanmean(div, axis=2), axis=1)
        else:
            sub_avg = sub
            div_avg = div

        axs[i, 1].plot(
            sweep_list,
            sub_avg,
            label="subtraction",
            c="firebrick",
            ls="dashed",
            marker="o",
            mfc="lightcoral",
            mec="maroon",
        )
        axs[i, 1].plot(
            sweep_list,
            div_avg,
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
        axs[i, 1].set_ylabel("pl (a.u.)")
    axs[-1, 1].set_xlabel(
        "Sweep parameter"
    )  # this is meant to be less indented than the line above

    high_res_xdata = np.linspace(
        np.min(fit_result_collection_lst[0].ROI_avg_fit_result.sweep_list),
        np.max(fit_result_collection_lst[0].ROI_avg_fit_result.sweep_list),
        10000,
    )

    # loop of fit backends first
    for fit_backend_number, fit_backend_fit_result in enumerate(fit_result_collection_lst):
        fit_backend_name = fit_backend_fit_result.fit_backend

        fit_params_lst = [
            fit_backend_fit_result.ROI_avg_fit_result.best_params,
            fit_backend_fit_result.single_pixel_fit_result,
            *fit_backend_fit_result.AOI_fit_results_lst,
        ]
        # now plot fits as third column
        for i, (fit_param_ar, s, r) in enumerate(zip(fit_params_lst, sigs, refs)):

            if not options["used_ref"]:
                sig_norm = s
            elif options["normalisation"] == "div":
                sig_norm = s / r
            elif options["normalisation"] == "sub":
                sig_norm = 1 + (s - r) / (s + r)

            if len(sig_norm.shape) > 1:
                sig_norm_avg = np.nanmean(np.nanmean(sig_norm, axis=2), axis=1)
            else:
                sig_norm_avg = sig_norm

            best_fit_ydata = fit_model(fit_param_ar, high_res_xdata)
            roi_fit_ydata = fit_model(
                fit_backend_fit_result.ROI_avg_fit_result.best_params, high_res_xdata
            )

            # this is the first loop -> plot raw data, add titles
            if not fit_backend_number:

                # raw data
                axs[i, 2].plot(
                    sweep_list,
                    sig_norm_avg,
                    label=f"raw data ({options['normalisation']})",
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
                        "Single Pixel Check - Fit", fontdict={"color": options["AOI_colors"][0]}
                    )
                else:
                    axs[i, 2].set_title(
                        "AOI " + str(i - 1) + " avg - Fit",
                        fontdict={"color": options["AOI_colors"][i - 1]},
                    )
            # ROI avg fit (as comparison)
            if i:
                axs[i, 2].plot(
                    high_res_xdata,
                    roi_fit_ydata,
                    label=f"ROI avg fit - {fit_backend_name}",
                    ls="dashed",
                    c=options["fit_backend_colors"][fit_backend_name]["AOI_ROI_fit_linecolor"],
                )
            # best fit
            axs[i, 2].plot(
                high_res_xdata,
                best_fit_ydata,
                label=f"fit - {fit_backend_name}",
                ls="dashed",
                c=options["fit_backend_colors"][fit_backend_name]["AOI_best_fit_linecolor"],
            )

            axs[i, 2].legend()
            axs[i, 2].grid(True)
            axs[i, 2].set_ylabel("pl (a.u.)")

    axs[-1, 2].set_xlabel("Sweep parameter")  # this is meant to be less indented than line above

    # currently not saving any of the data from this plot (not sure what the user would ever want)

    if options["save_plots"]:
        fig.savefig(options["output_dir"] / ("AOI_spectra_fits." + options["save_fig_type"]))

    return fig


# ============================================================================


def pl_param_image(
    options, fit_model, pixel_fit_params, param_name, param_number=0, errorplot=False
):
    """
    Plots an image corresponding to a single parameter in pixel_fit_params.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    fit_model : `qdmpy.pl.model.FitModel`
        Model we're fitting to.
    pixel_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    param_name : str
        Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'.

    Optional arguments
    ------------------
    param_number : int
        Which version of the parameter you want. I.e. there might be 8 independent parameters
        in the fit model called 'pos', each labeled 'pos_0', 'pos_1' etc. Default: 0.
    errorplot : bool
        Default: false. Denotes that errors dict has been passed in (e.g. sigmas), so
        ylabel & save names are changed accordingly. Can't be True if param_name='residual'.

    Returns
    -------
    fig : matplotlib Figure object
    """

    image = pixel_fit_params[param_name + "_" + str(param_number)]

    if param_name == "residual":
        c_range = qdmpy.plot.common.get_colormap_range(
            options["colormap_range_dicts"]["residual_images"], image
        )
        c_map = options["colormaps"]["residual_images"]
    elif errorplot:
        c_range = qdmpy.plot.common.get_colormap_range(
            options["colormap_range_dicts"]["sigma_images"], image
        )
        c_map = options["colormaps"]["sigma_images"]
    else:
        c_range = qdmpy.plot.common.get_colormap_range(
            options["colormap_range_dicts"]["param_images"], image
        )
        c_map = options["colormaps"]["param_images"]

    if param_name == "residual" and errorplot:
        warnings.warn("residual doesn't have an error, can't plot residual sigma (ret. None).")
        return None

    if errorplot:
        c_label = "SD: " + fit_model.get_param_unit(param_name, param_number)
    else:
        c_label = fit_model.get_param_unit(param_name, param_number)

    fig, ax = plt.subplots()

    fig, ax = qdmpy.plot.common.plot_image_on_ax(
        fig, ax, options, image, param_name + "_" + str(param_number), c_map, c_range, c_label
    )

    if options["save_plots"]:
        if errorplot:
            path = options["output_dir"] / (
                param_name + "_" + str(param_number) + "_sigma." + options["save_fig_type"]
            )
        else:
            path = options["output_dir"] / (
                param_name + "_" + str(param_number) + "." + options["save_fig_type"]
            )
        fig.savefig(path)

    return fig


# ============================================================================


def pl_param_images(options, fit_model, pixel_fit_params, param_name, errorplot=False):
    """
    Plots images for all independent versions of a single parameter type in pixel_fit_params.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    fit_model : `qdmpy.pl.model.FitModel`
        Model we're fitting to.
    pixel_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    param_name : str
        Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'.
    errorplot : bool
        Default: false. Denotes that errors dict has been passed in (e.g. sigmas), so
        ylabel & save names are changed accordingly. Can't be True if param_name='residual'.

    Returns
    -------
    fig : matplotlib Figure object
    """

    # if no fit completed
    if pixel_fit_params is None:
        warnings.warn(
            "'pixel_fit_params' arg to function 'pl_param_images' is 'None'.\n"
            + "Probably no pixel fitting completed."  # noqa: W503
        )
        return None

    if param_name == "residual" and errorplot:
        warnings.warn("residual doesn't have an error, can't plot residual sigma (ret. None).")
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
    our_keys.sort(key=param_sorter)  # i.e. key = lambda x: int(x.split("_")[-1])
    nk = len(our_keys)

    if nk == 1:
        # just one image, so plot normally
        fig = pl_param_image(options, fit_model, pixel_fit_params, param_name, 0, errorplot)
    else:
        if nk <= 8:
            num_columns = 4
            num_rows = 2
        else:
            num_columns = 2
            num_rows = math.ceil(nk / 2)

        figsize = mpl.rcParams["figure.figsize"].copy()
        figsize[0] *= num_columns
        figsize[1] *= num_rows

        # standardise figsize of output
        # figsize[0] *= 3 / 4
        # figsize[1] *= 3 / 4

        fig, axs = plt.subplots(
            num_rows,
            num_columns,
            figsize=figsize,
            sharex=False,
            sharey=False,
        )
        # plot 8-lorentzian peaks in a more helpful way
        if nk <= 8 and any([f.startswith("lorentzian") for f in options["fit_functions"]]):
            param_nums = []  # [0, 1, 2, 3, 7, 6, 5, 4] etc.
            param_nums.extend(list(range(nk // 2)))
            if nk % 2:
                param_nums.append(nk // 2 + 1)
            if len(param_nums) < 4:
                param_nums.extend([-1 for _ in range(4 - len(param_nums))])  # dummies
            param_nums.extend(list(range(nk - 1, (nk - 1) // 2, -1)))  # range(start, stop, step)
            param_nums.extend([-1 for _ in range(8 - len(param_nums))])  # add on dummies
            param_axis_iterator = zip(param_nums, axs.flatten())
        # otherwise plot in a more conventional order
        else:
            param_axis_iterator = enumerate(axs.flatten())

        for param_number, ax in param_axis_iterator:

            param_key = param_name + "_" + str(param_number)
            try:
                image_data = pixel_fit_params[param_key]
            except KeyError:
                # we have too many axes (i.e. 7 params, 8 subplots), delete the axs
                fig.delaxes(ax)
                continue

            if param_name == "residual":
                c_range = qdmpy.plot.common.get_colormap_range(
                    options["colormap_range_dicts"]["residual_images"], image_data
                )
                c_map = options["colormaps"]["residual_images"]
            elif errorplot:
                c_range = qdmpy.plot.common.get_colormap_range(
                    options["colormap_range_dicts"]["sigma_images"], image_data
                )
                c_map = options["colormaps"]["sigma_images"]
            else:
                c_range = qdmpy.plot.common.get_colormap_range(
                    options["colormap_range_dicts"]["param_images"], image_data
                )
                c_map = options["colormaps"]["param_images"]

            if errorplot:
                c_label = "SD: " + fit_model.get_param_unit(param_name, param_number)
            else:
                c_label = fit_model.get_param_unit(param_name, param_number)

            qdmpy.plot.common.plot_image_on_ax(
                fig, ax, options, image_data, param_key, c_map, c_range, c_label
            )

        if options["save_plots"]:
            if errorplot:
                path = options["output_dir"] / (param_name + "_sigma." + options["save_fig_type"])
            else:
                path = options["output_dir"] / (param_name + "." + options["save_fig_type"])
            fig.savefig(path)

    return fig


# ============================================================================


def pl_params_flattened(
    options,
    fit_model,
    pixel_fit_params,
    roi_avg_fit_result,
    param_name,
    sigmas=None,
    plot_bounds=True,
    plot_sigmas=True,
    errorevery=1,
):
    """
    Compare pixel fits against flattened pixels: initial guess vs ROI fit vs fit result.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    fit_model : `qdmpy.pl.model.FitModel`
        Model we're fitting to.
    pixel_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    roi_avg_fit_result
        `qdmpy.pl.common.ROIAvgFitResult` object.
    param_name : str
        Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'.
    plot_bounds : bool
        Defaults to True, add fit bounds/constraints to plot. (Does nothing for residual plots)
    plot_sigmas : bool
        Defaults to True, add error bars (sigma) to plot. (Does nothing for residual plots)

    Returns
    -------
    fig : matplotlib Figure object

    """
    if pixel_fit_params is None:
        return None

    if not sigmas:
        plot_sigmas = False

    # initial guess vs ROI fit vs pixel fit

    param_keys = sorted([p for p in pixel_fit_params if p.startswith(param_name)])

    figsize = mpl.rcParams["figure.figsize"].copy()

    height = len(param_keys)
    width = 1
    figsize[0] *= 2  # make some extra space...
    figsize[1] = figsize[1] * height / 2 if height > 1 else figsize[1]

    fig, axs = plt.subplots(height, width, sharex=True, figsize=figsize)
    # axs indexed: axs[row, col] (n_cols = 1 here, so index linearly)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    if param_name == "residual":
        param_guesses = [
            fit_model.residuals_scipyfit(
                roi_avg_fit_result.init_param_guess,
                roi_avg_fit_result.sweep_list,
                roi_avg_fit_result.pl_roi,
            ).sum()
        ]

        param_roi_fits = [
            fit_model.residuals_scipyfit(
                roi_avg_fit_result.best_params,
                roi_avg_fit_result.sweep_list,
                roi_avg_fit_result.pl_roi,
            ).sum()
        ]
    else:
        param_guesses = []
        param_roi_fits = []
        param_bounds = []
        for fn_obj in fit_model.fn_chain:
            for param_num, param_root in enumerate(fn_obj.param_defn):
                if param_root == param_name:
                    param_guesses.append(
                        roi_avg_fit_result.init_param_guess[
                            fn_obj.this_fn_param_indices[param_num]
                        ]
                    )
                    param_roi_fits.append(
                        roi_avg_fit_result.best_params[fn_obj.this_fn_param_indices[param_num]]
                    )
                    param_bounds.append(
                        roi_avg_fit_result.init_param_bounds[
                            fn_obj.this_fn_param_indices[param_num]
                        ]
                    )

    axs[-1].set_xlabel("Pixel # (flattened)")
    for ax in axs:
        ax.set_ylabel(fit_model.get_param_unit(param_name, 0))
        ax.grid()

    # uses the default color cycle... i.e.:
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    if param_name != "residual" and plot_bounds:
        legend_names = ["Fit bounds", "Initial guesses", "ROI fits"]
        custom_lines = [
            Line2D([0], [0], color="k", ls=(0, (2, 1)), lw=mpl.rcParams["lines.linewidth"]),
            Line2D([0], [0], color="k", ls=(0, (1, 1)), lw=mpl.rcParams["lines.linewidth"] * 2),
            Line2D([0], [0], color="k", ls=(0, (5, 1)), lw=mpl.rcParams["lines.linewidth"] * 2),
        ]
    else:
        legend_names = ["Initial guesses", "ROI fits"]
        # https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/linestyles.html
        custom_lines = [
            Line2D([0], [0], color="k", ls=(0, (1, 1)), lw=mpl.rcParams["lines.linewidth"] * 2),
            Line2D([0], [0], color="k", ls=(0, (5, 1)), lw=mpl.rcParams["lines.linewidth"] * 2),
        ]

    if param_name != "residual" and plot_bounds:
        for i, bounds in enumerate(param_bounds):
            for b in bounds:
                axs[i].axhline(b, ls=(0, (2, 1)), c="grey", zorder=9)

    for i, guess in enumerate(param_guesses):
        axs[i].axhline(
            guess, ls=(0, (1, 1)), c="k", zorder=10, lw=mpl.rcParams["lines.linewidth"] * 2
        )

    for i, roi_fit in enumerate(param_roi_fits):
        axs[i].axhline(
            roi_fit, ls=(0, (5, 1)), lw=mpl.rcParams["lines.linewidth"] * 2, c="k", zorder=5
        )

    for i, (param_key, color) in enumerate(zip(param_keys, colors)):
        if param_name == "residual" or not plot_sigmas:
            axs[i].plot(
                pixel_fit_params[param_key].flatten(),
                label=param_key,
                marker="o",
                mfc="w",
                ms=mpl.rcParams["lines.markersize"],
                mec=color,
                ls="",
                zorder=20,
            )
        else:
            yvals = pixel_fit_params[param_key].flatten()
            xvals = list(range(len(yvals)))
            axs[i].errorbar(
                xvals,
                yvals,
                xerr=None,
                yerr=sigmas[param_key].flatten(),
                label=param_key,
                marker="o",
                mfc="w",
                ms=mpl.rcParams["lines.markersize"],
                mec=color,
                ecolor=color,
                ls="",
                zorder=20,
                errorevery=errorevery,
            )
        legend_names.append(param_key)
        custom_lines.append(
            Line2D(
                [0],
                [0],
                marker="o",
                mfc="w",
                ms=mpl.rcParams["lines.markersize"] * 2,
                mec=color,
                ls="",
            )
        )

    # https://jdhao.github.io/2018/01/23/matplotlib-legend-outside-of-axes/
    axs[0].legend(
        custom_lines,
        legend_names,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=height + 2,
        borderaxespad=0,
        frameon=False,
        fontsize="medium",
    )

    if options["save_plots"]:
        fig.savefig(
            options["output_dir"] / (f"{param_name}_fit_flattened." + options["save_fig_type"])
        )

    return fig


# ============================================================================


def other_measurements(options, skip_first=0):
    """
    Plot any other tsv/csv datasets (at base_path + s for s in
    options["other_measurement_suffixes"]). Assumes first column is some form of ind. dataset
    """
    suffixes = options["other_measurement_suffixes"]
    if not suffixes:
        return None
    paths = [(s, options["filepath"] + s) for s in suffixes]
    good_paths = [(s, path) for s, path in paths if Path(path).is_file()]
    datasets = {}
    for s, path in good_paths:
        datasets[s] = options["system"].get_headers_and_read_csv(options, path)

    fig = None
    for s, (headers, dset) in datasets.items():

        num_series = len(headers) - 1  # 0th is 'time' array etc. (indep. vals)
        figsize = mpl.rcParams["figure.figsize"].copy()
        figsize[0] *= 2
        figsize[1] *= num_series  # extra height

        fig, axs = plt.subplots(
            num_series, 1, figsize=figsize, sharex=True
        )
        for i, header in enumerate(headers[1:]):
            axs[i].plot(
                dset[skip_first:, 0],
                dset[skip_first:, i + 1],
                label=header,
                marker="o",
                mfc="w",
                ms=mpl.rcParams["lines.markersize"],
                mec="purple",
                ls="",
                zorder=20,
            )
            axs[i].set_xlabel(headers[0])
            axs[i].set_ylabel(headers[i + 1])
            axs[i].legend()
            axs[i].grid()

        if options["save_plots"]:
            fig.savefig(
                options["output_dir"] / (f"other-meas-{Path(s).stem}." + options["save_fig_type"])
            )
    return fig


# ============================================================================


def _add_patch_rect(ax, rect_corner_x, rect_corner_y, size_x, size_y, label=None, edgecolor="b"):
    """
    Adds a rectangular annotation onto ax.

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


def _annotate_roi_image(options, ax):
    """
    Annotates ROI onto a given Axis object. Generally used on a pl image.
    """
    binning = options["additional_bins"]
    if binning == 0:
        binning = 1
    if options["ROI"] == "Full":
        pass
    elif options["ROI"] == "Rectangle":

        # these options are [x, y], opposite to data indexing convention
        start_x, start_y = options["ROI_start"]
        end_x, end_y = options["ROI_end"]

        _add_patch_rect(
            ax,
            start_x,
            start_y,
            end_x - start_x + 1,
            end_y - start_y + 1,
            label="ROI",
            edgecolor="r",
        )
    else:
        raise qdmpy.system.OptionsError(
            "ROI", options["ROI"], options["system"], custom_msg="Unknown ROI encountered."
        )


# ============================================================================


def _annotate_aoi_image(options, ax):
    """
    Annotates AOI onto a given Axis object. Generally used on pl image.
    """
    # NOTE I don't think a binning check is required.
    binning = options["additional_bins"]
    if binning == 0:
        binning = 1

    # annotate single pixel check
    corner_x = options["single_pixel_check"][0]
    corner_y = options["single_pixel_check"][1]
    size = 1
    _add_patch_rect(
        ax, corner_x, corner_y, size, size, label="PX check", edgecolor=options["AOI_colors"][0]
    )

    i = 0
    while True:
        i += 1
        try:
            # these options are [x, y], opposite to data indexing convention
            start = options["AOI_" + str(i) + "_start"]
            end = options["AOI_" + str(i) + "_end"]
            if start is None or end is None:
                continue

            # need to handle binning???
            _add_patch_rect(
                ax,
                *start,
                end[0] - start[0] + 1,
                end[1] - start[1] + 1,
                label="AOI " + str(i),
                edgecolor=options["AOI_colors"][i],
            )
        except KeyError:
            break


# ============================================================================

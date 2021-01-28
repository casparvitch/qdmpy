# - coding: utf-8 -*-
"""
This module holds functions for plotting initial processing images and fit results.

Functions
---------
 - `QDMPy.plot.fits.plot_ROI_PL_image`
 - `QDMPy.plot.fits.plot_AOI_PL_images`
 - `QDMPy.plot.fits.plot_ROI_avg_fits`
 - `QDMPy.plot.fits.plot_AOI_spectra`
 - `QDMPy.plot.fits.plot_AOI_spectra_fit`
 - `QDMPy.plot.fits.plot_param_image`
 - `QDMPy.plot.fits.plot_param_images`
 - `QDMPy.plot.fits._add_patch_rect`
 - `QDMPy.plot.fits._annotate_ROI_image`
 - `QDMPy.plot.fits._annotate_AOI_image`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.plot.fits.plot_ROI_PL_image": True,
    "QDMPy.plot.fits.plot_AOI_PL_images": True,
    "QDMPy.plot.fits.plot_ROI_avg_fits": True,
    "QDMPy.plot.fits.plot_AOI_spectra": True,
    "QDMPy.plot.fits.plot_AOI_spectra_fit": True,
    "QDMPy.plot.fits.plot_param_image": True,
    "QDMPy.plot.fits.plot_param_images": True,
    "QDMPy.plot.fits._add_patch_rect": True,
    "QDMPy.plot.fits._annotate_ROI_image": True,
    "QDMPy.plot.fits._annotate_AOI_image": True,
}

# ============================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import math
import warnings

# ============================================================================

import QDMPy.fit._models as fit_models
import QDMPy.systems
import QDMPy.io.json2dict
import QDMPy.io.raw
import QDMPy.plot.common as plot_common

# ===========================================================================


def plot_ROI_PL_image(options, PL_image):
    """
    Plots full PL image with ROI region annotated.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    PL_image : np array, 2D
        Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned but
        not cut down to ROI.

    Returns
    -------
    fig : matplotlib Figure object
    """

    c_map = options["colormaps"]["PL_images"]
    c_range = plot_common._get_colormap_range(
        options["colormap_range_dicts"]["PL_images"], PL_image
    )

    fig, ax = plt.subplots(constrained_layout=True)

    fig, ax = plot_common.plot_image_on_ax(
        fig,
        ax,
        options,
        PL_image,
        "PL - ROI",
        c_map,
        c_range,
        "Counts",
    )

    if options["annotate_image_regions"]:
        _annotate_ROI_image(options, ax)

    if options["save_plots"]:
        fig.savefig(options["output_dir"] / ("PL - ROI." + options["save_fig_type"]))

    return fig


# ============================================================================


def plot_AOI_PL_images(options, PL_image_ROI):
    """
    Plots PL image cut down to ROI, with annotated AOI regions.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    PL_image_ROI : np array, 2D
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
    fig, ax = plt.subplots(constrained_layout=True)
    c_map = options["colormaps"]["PL_images"]
    c_range = plot_common._get_colormap_range(
        options["colormap_range_dicts"]["PL_images"], PL_image_ROI
    )

    fig, ax = plot_common.plot_image_on_ax(
        fig,
        ax,
        options,
        PL_image_ROI,
        "PL - AOIs",
        c_map,
        c_range,
        "Counts",
    )

    if options["annotate_image_regions"]:
        _annotate_AOI_image(options, ax)

    if options["save_plots"]:
        fig.savefig(options["output_dir"] / ("PL - AOIs." + options["save_fig_type"]))

    return fig


# ============================================================================


def plot_ROI_avg_fits(options, backend_ROI_results_lst):
    """
    Plots fit of spectrum averaged across ROI, as well as corresponding residual values.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    backend_ROI_results_lst : list of tuples
        Format: (fit_backend, `QDMPy.fit._shared.ROIAvgFitResult` objects), for each fit_backend

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
        label=f"raw data ({options['normalisation']})",
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


def plot_AOI_spectra(options, sig, ref, sweep_list):
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
    AOIs = QDMPy.io.raw._define_AOIs(options)

    # pre-process data to plot
    sig_avgs = []
    ref_avgs = []
    for i, AOI in enumerate(AOIs):
        sig_avg = np.nanmean(np.nanmean(sig[:, AOI[0], AOI[1]], axis=2), axis=1)
        ref_avg = np.nanmean(np.nanmean(ref[:, AOI[0], AOI[1]], axis=2), axis=1)
        sig_avgs.append(sig_avg)
        ref_avgs.append(ref_avg)

    figsize = mpl.rcParams["figure.figsize"].copy()
    num_wide = 2 if len(AOIs) < 2 else len(AOIs)
    figsize[0] *= num_wide
    figsize[1] *= 2
    fig, axs = plt.subplots(
        2, num_wide, figsize=figsize, sharex=True, sharey=False, constrained_layout=True
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
        axs[1, 1].set_title("Division Normalisation (Weber contrast, sig / ref)")
        axs[1, 1].set_xlabel("Sweep parameter")
        axs[1, 1].set_ylabel("PL (a.u.)")

    # delete axes that we didn't use
    for i in range(len(AOIs)):
        if i < len(
            options["system"].option_choices("normalisation")
        ):  # we used these (normalisation)
            continue
        else:  # we didn't use these
            fig.delaxes(axs[1, i])

    if len(AOIs) == 1:
        # again, didn't use
        fig.delaxes(axs[0, 1])

    output_dict = {}
    for i in range(len(AOIs)):
        output_dict["AOI_sig_avg" + "_" + str(i + 1)] = sig_avgs[i]
        output_dict["AOI_ref_avg" + "_" + str(i + 1)] = ref_avgs[i]

    QDMPy.io.json2dict.dict_to_json(output_dict, "AOI_spectra.json", options["data_dir"])

    if options["save_plots"]:
        fig.savefig(options["output_dir"] / ("AOI_spectra." + options["save_fig_type"]))
    return fig


# ============================================================================


def plot_AOI_spectra_fit(
    options,
    sig,
    ref,
    sweep_list,
    fit_result_collection_lst,
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
        Format: [sweep_vals, y, x]

    ref : np array, 3D
        Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, y, x]
    sweep_list : list
        List of sweep parameter values (with removed unwanted sweeps at start/end)

    fit_result_collection_lst : list
        List of `QDMPy.fit._shared.FitResultCollection` objects (one for each fit_backend)
        holding ROI, AOI fit results

    fit_model : `QDMPy.fit._models.FitModel`
        Model we're fitting to.

    Returns
    -------
    fig : matplotlib Figure object
    """

    # rows:
    # ROI avg, single pixel, then each AOI
    # columns:
    # sig & ref, sub & div norm, fit -> compared to ROI {raw, fit, ROI_avg_fit}

    AOIs = QDMPy.io.raw._define_AOIs(options)

    figsize = mpl.rcParams["figure.figsize"].copy()
    figsize[0] *= 3  # number of columns
    figsize[1] *= 2 + len(AOIs)  # number of rows

    fig, axs = plt.subplots(
        2 + len(AOIs), 3, figsize=figsize, sharex=True, sharey=False, constrained_layout=True
    )

    # pre-process raw data to plot
    sig_avgs = []
    ref_avgs = []
    # add roi data
    roi_avg_sig = np.nanmean(np.nanmean(sig, axis=2), axis=1)
    roi_avg_ref = np.nanmean(np.nanmean(ref, axis=2), axis=1)
    sig_avgs.append(roi_avg_sig)
    ref_avgs.append(roi_avg_ref)
    # add single pixel check
    pixel_sig = sig[:, options["single_pixel_check"][1], options["single_pixel_check"][0]]
    pixel_ref = ref[:, options["single_pixel_check"][1], options["single_pixel_check"][0]]
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
            1 + (sig - ref) / (sig + ref),
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
        np.min(fit_result_collection_lst[0].roi_avg_fit_result.sweep_list),
        np.max(fit_result_collection_lst[0].roi_avg_fit_result.sweep_list),
        10000,
    )

    # loop of fit backends first
    for fit_backend_number, fit_backend_fit_result in enumerate(fit_result_collection_lst):
        fit_backend_name = fit_backend_fit_result.fit_backend

        fit_params_lst = [
            fit_backend_fit_result.roi_avg_fit_result.best_params,
            fit_backend_fit_result.single_pixel_fit_result,
            *fit_backend_fit_result.AOI_fit_results_lst,
        ]
        # now plot fits as third column
        for i, (fit_param_ar, sig, ref) in enumerate(zip(fit_params_lst, sig_avgs, ref_avgs)):
            if not options["used_ref"]:
                sig_norm = sig
            elif options["normalisation"] == "div":
                sig_norm = sig / ref
            elif options["normalisation"] == "sub":
                sig_norm = 1 + (sig - ref) / (sig + ref)

            best_fit_ydata = fit_model(fit_param_ar, high_res_xdata)
            roi_fit_ydata = fit_model(
                fit_backend_fit_result.roi_avg_fit_result.best_params, high_res_xdata
            )

            # this is the first loop -> plot raw data, add titles
            if not fit_backend_number:

                # raw data
                axs[i, 2].plot(
                    sweep_list,
                    sig_norm,
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

    fit_model : `QDMPy.fit._models.FitModel`
        Model we're fitting to.

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
    c_range = plot_common._get_colormap_range(
        options["colormap_range_dicts"]["param_images"], image
    )
    c_label = fit_models.get_param_unit(fit_model, param_name, param_number)

    fig, ax = plot_common.plot_image(
        options,
        image,
        param_name + "_" + str(param_number),
        c_map,
        c_range,
        c_label,
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

    fit_model : `QDMPy.fit._models.FitModel`
        Model we're fitting to.

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

            c_range = plot_common._get_colormap_range(
                options["colormap_range_dicts"]["param_images"], image_data
            )
            c_label = fit_models.get_param_unit(fit_model, param_name, param_number)

            plot_common.plot_image_on_ax(
                fig,
                ax,
                options,
                image_data,
                param_key,
                c_map,
                c_range,
                c_label,
            )

        if options["save_plots"]:
            fig.savefig(options["output_dir"] / (param_name + "." + options["save_fig_type"]))


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


def _annotate_ROI_image(options, ax):
    """
    Annotates ROI onto a given Axis object. Generally used on a PL image.
    """
    binning = options["additional_bins"]
    if binning == 0:
        binning = 1
    if options["ROI"] == "Full":
        return None
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
        raise QDMPy.systems.OptionsError(
            "ROI", options["ROI"], options["system"], custom_msg="Unknown ROI encountered."
        )


# ============================================================================


def _annotate_AOI_image(options, ax):
    """
    Annotates AOI onto a given Axis object. Generally used on PL image.
    """
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

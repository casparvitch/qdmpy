# - coding: utf-8 -*-
"""
This module holds functions for plotting source (fields).

Functions
---------
 - `qdmpy.plot.source.source_param`
 - `qdmpy.plot.source.current`
 - `qdmpy.plot.source.current_stream`
 - `qdmpy.plot.source.magnetization`
 - `qdmpy.plot.source.divperp_j`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.plot.source.source_param": True,
    "qdmpy.plot.source.current": True,
    "qdmpy.plot.source.current_stream": True,
    "qdmpy.plot.source.magnetization": True,
    "qdmpy.plot.source.divperp_j": True,
}

# ============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import cm
from matplotlib.colors import ListedColormap
import warnings
import copy

# ============================================================================

import qdmpy.plot.common

# ============================================================================


def source_param(
    options,
    param_name,  # "Jx_full_from_bnv" etc.
    source_params,
    c_map=None,
    c_map_type="source_images",
    c_range_type="percentile",
    c_range_vals=(1, 99),
    cbar_label="",
):
    """plot a given source param.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    name : str
        Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames
    source_params : dict
        Dictionary, key: param_keys, val: image (2D) of (source field) param values across FOV.
    c_map : str, default: None
        colormap to use overrides c_map_type and c_range_type
    c_map_type : str, default: "source_images"
        colormap type to search options (options["colormaps"][c_map_type]) for
    c_map_range : str, default: "percentile"
        colormap range option (see `qdmpy.plot.common.get_colormap_range`) to use
    c_range_vals : number or list, default: (1, 99)
        passed with c_map_range to get_colormap_range
    c_bar_label : str, default:""
        label to chuck on ye olde colorbar (z-axis label).

    Returns
    -------
    fig : matplotlib Figure object
    """
    if source_params is None:
        return None
    if param_name not in source_params:
        warnings.warn(f"No param (key) called {param_name} in source_params.")
        return None

    fig, ax = plt.subplots(constrained_layout=True)

    if c_map is None:
        c_range = qdmpy.plot.common.get_colormap_range(
            {"type": c_range_type, "values": c_range_vals}, source_params[param_name]
        )
        c_map = options["colormaps"][c_map_type]

    title = f"{param_name}"

    qdmpy.plot.common.plot_image_on_ax(
        fig, ax, options, source_params[param_name], title, c_map, c_range, cbar_label
    )

    if options["save_plots"]:
        fig.savefig(options["source_dir"] / (f"{param_name}." + options["save_fig_type"]))

    return fig


# ============================================================================


def current(options, source_params, plot_bgrounds=True):
    """Plots current (Jx, Jy, Jnorm). Optionally plot background subtracted.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    source_params : dict
        Dictionary, key: param_keys, val: image (2D) of (source field) param values across FOV.
    plot_bgrounds : {bool}, default: True
        Plot background images

    Returns
    -------
    fig : matplotlib Figure object
    """
    if source_params is None:
        return None

    plot_bgrounds = plot_bgrounds and options["source_bground_method"]

    figsize = mpl.rcParams["figure.figsize"].copy()
    width = 3
    height = len(options["recon_methods"])  # number of rows
    height *= 3 if plot_bgrounds else 1
    figsize[0] *= width  # number of columns
    figsize[1] *= height
    fig, axs = plt.subplots(height, width, figsize=figsize, constrained_layout=True)

    components = {
        "x": "current_vector_images",
        "y": "current_vector_images",
        "norm": "current_norm_images",
    }

    if plot_bgrounds:
        for m_idx, method in enumerate(options["recon_methods"]):
            for row, suffix in enumerate(["_full", "_bground", ""]):
                for col, comp in enumerate(components.keys()):
                    ax = (
                        axs[row, col]
                        if len(options["recon_methods"]) == 1
                        else axs[3 * m_idx + row, col]
                    )

                    ckey = components[comp]
                    name = "J" + comp + suffix + "_" + method

                    if name not in source_params:
                        warnings.warn(f"source param {name} missing from source_params.")
                        continue
                    elif source_params[name] is None:
                        warnings.warn(f"source_param[{name}] was None?")
                        continue

                    jmap = source_params[name]

                    c_range = qdmpy.plot.common.get_colormap_range(
                        options["colormap_range_dicts"][ckey], jmap
                    )
                    c_map = options["colormaps"][ckey]
                    qdmpy.plot.common.plot_image_on_ax(
                        fig, ax, options, jmap, name, c_map, c_range, "J (A/m)"
                    )
    else:
        for m_idx, method in enumerate(options["recon_methods"]):
            for i, comp in enumerate(components.keys()):
                ax = axs[i] if len(options["recon_methods"]) == 1 else axs[m_idx, i]

                ckey = components[comp]
                name = "J" + comp + "_" + method

                if name not in source_params:
                    warnings.warn(f"source param {name} missing from source_params.")
                    continue
                elif source_params[name] is None:
                    warnings.warn(f"source_param[{name}] was None?")
                    continue

                jmap = source_params[name]

                c_range = qdmpy.plot.common.get_colormap_range(
                    options["colormap_range_dicts"][ckey], jmap
                )
                c_map = options["colormaps"][ckey]

                qdmpy.plot.common.plot_image_on_ax(
                    fig, ax, options, jmap, name, c_map, c_range, "J (A/m)"
                )

    if options["save_plots"]:
        fig.savefig(options["source_dir"] / ("J." + options["save_fig_type"]))

    return fig


# ============================================================================


def current_stream(
    options,
    source_params,
    background_image=None,
    probe_image=None,
    probe_color="red",
    probe_alpha=1.0,
    probe_cutoff=0.9,
):
    """Plot current density as a streamplot.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    source_params : dict
        Dictionary, key: param_keys, val: image (2D) of (source field) param values across FOV.
    background_image : numpy array or None, default: None
        If not None, must be pl image of ROI (or other background image), to plot behind streams

    Returns
    -------
    fig : matplotlib Figure object
    """
    if source_params is None:
        return None

    components = ["x", "y", "norm"]

    width = len(options["recon_methods"])
    height = 1
    figsize = mpl.rcParams["figure.figsize"].copy()
    figsize[0] *= width
    fig, axs = plt.subplots(height, width, figsize=figsize, constrained_layout=True)

    for m_idx, method in enumerate(options["recon_methods"]):

        flag = False
        for p in ["J" + comp + "_" + method for comp in components]:
            if p not in source_params:
                warnings.warn(
                    f"bfield param '{p}'' missing from source_params, skipping stream plot."
                )
                flag = True
                break
            elif source_params[p] is None:
                flag = True
                break
        if flag:
            continue

        ax = axs if width == 1 else axs[m_idx]

        if background_image is not None:
            with warnings.catch_warnings():
                # ignore all-NaN slice encountered warning
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ax.imshow(
                    background_image,
                    cmap="Greys_r",
                    vmin=np.nanmin(background_image),
                    vmax=np.nanmax(background_image),
                    alpha=options["streamplot_pl_alpha"],
                )
            if probe_image is not None:
                my_cmap = copy.copy(cm.get_cmap("Reds"))  # doesn't matter _what_ the cmap imshow
                my_cmap.set_under("k", alpha=0)
                my_cmap.set_over(probe_color, alpha=probe_alpha)
                ax.imshow(
                    probe_image,
                    cmap=my_cmap,
                    clim=[
                        probe_cutoff * np.nanmax(probe_image),
                        probe_cutoff * np.nanmax(probe_image) + 0.000001,
                    ],
                )

        shp = source_params["Jx_" + method].shape

        jnorms = source_params["Jnorm_" + method]
        c_range = qdmpy.plot.common.get_colormap_range(
            options["colormap_range_dicts"]["current_norm_images"], jnorms
        )

        c = np.clip(jnorms, a_min=c_range[0], a_max=c_range[1])

        # clear previous cmap if registered already
        if "alpha_cmap" in plt.colormaps():
            cm.unregister_cmap("alpha_cmap")

        cbar_drange = options["streamplot_cbar_options"]["dynamic_range"]
        u = options["streamplot_cbar_options"]["alpha_ramp_factor"]
        h = options["streamplot_cbar_options"]["low_cutoff"]
        f = options["streamplot_cbar_options"]["high_cutoff"]
        base_cmap = cm.get_cmap(options["colormaps"]["current_norm_images"], cbar_drange)

        new_colors = base_cmap(np.linspace(0, 1, cbar_drange))

        xvals = np.linspace(0, 1, cbar_drange)
        cbar_alpha_ramp = np.tanh(u * (xvals - h)) / np.tanh(u * (f - h))
        cbar_alpha_ramp[cbar_alpha_ramp < 0.0] = 0.0
        cbar_alpha_ramp[cbar_alpha_ramp > 1.0] = 1.0

        new_colors[:, 3] = new_colors[:, 3] * cbar_alpha_ramp

        cmap_w_alpha = ListedColormap(new_colors)

        cm.register_cmap(name="alpha_cmap", cmap=cmap_w_alpha)

        strm = ax.streamplot(
            np.arange(shp[1]),
            np.arange(shp[0]),
            source_params["Jx_" + method],
            -source_params["Jy_" + method],
            color=c,
            cmap="alpha_cmap",
            **options["streamplot_options"],
        )

        divider = make_axes_locatable(ax)
        aspect = 20
        pad_fraction = 1
        width = axes_size.AxesY(ax, aspect=1.0 / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)

        cbar = fig.colorbar(strm.lines, cax=cax)

        tick_locator = matplotlib.ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.linewidth = 0.5
        cbar.ax.tick_params(direction="in", labelsize=10, size=2)

        cbar.ax.set_ylabel("Jnorm (A/m)", rotation=270)
        cbar.outline.set_linewidth(0.5)

        ax.set_title("J " + method.replace("_", " "))

        if options["show_scalebar"]:
            pixel_size = options["system"].get_raw_pixel_size(options) * options["total_bin"]
            scalebar = ScaleBar(pixel_size)
            ax.add_artist(scalebar)

        if options["polygon_nodes"] and options["annotate_polygons"]:
            for p in options["polygon_nodes"]:
                ax.add_patch(
                    matplotlib.patches.Polygon(np.array(p), **options["polygon_patch_params"])
                )

            ax.set_facecolor("w")

        if not options["show_tick_marks"]:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        ax.set_aspect("equal")

    if options["save_plots"]:
        fig.savefig(options["source_dir"] / ("Jstream." + options["large_fig_save_type"]))

    return fig


# ============================================================================


def magnetization(options, source_params, plot_bgrounds=True):
    """Plots magnetization. Optionally plot background subtracted.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    source_params : dict
        Dictionary, key: param_keys, val: image (2D) of (source field) param values across FOV.
    plot_bgrounds : {bool}, default: True
        Plot background images

    Returns
    -------
    fig : matplotlib Figure object
    """
    if source_params is None:
        return None

    plot_bgrounds = plot_bgrounds and options["source_bground_method"]

    mag_angle = options["magnetization_angle"]
    if mag_angle is None:
        root_name = "Mz"
    else:
        root_name = "Mpsi"

    figsize = mpl.rcParams["figure.figsize"].copy()
    c_map = options["colormaps"]["magnetization_images"]
    width = 3 if plot_bgrounds else 1
    height = len(options["recon_methods"])  # number of rows
    figsize[0] *= width  # number of columns
    figsize[1] *= height
    fig, axs = plt.subplots(height, width, figsize=figsize, constrained_layout=True)

    # step through recon methods, without bground
    if not plot_bgrounds:
        for m_idx, method in enumerate(options["recon_methods"]):
            ax = axs if width * height == 1 else axs[m_idx]
            name = root_name + "_" + method

            if name not in source_params:
                warnings.warn(f"source param {name} missing from source_params.")
                continue
            elif source_params[name] is None:
                warnings.warn(f"source_param[{name}] was None?")
                continue

            data = source_params[name]
            c_range = qdmpy.plot.common.get_colormap_range(
                options["colormap_range_dicts"]["magnetization_images"], data
            )
            qdmpy.plot.common.plot_image_on_ax(
                fig, ax, options, data, name, c_map, c_range, root_name + " (mu_B/nm^2)"
            )
    # step through recon methods with background
    else:
        for m_idx, method in enumerate(options["recon_methods"]):

            comps = ["_full", "_bground", ""]
            for i, comp in enumerate(comps):
                ax = axs[i] if len(options["recon_methods"]) == 1 else axs[m_idx, i]

                name = root_name + "_" + method + comp

                if name not in source_params:
                    warnings.warn(f"source param {name} missing from source_params.")
                    continue
                elif source_params[name] is None:
                    warnings.warn(f"source_param[{name}] was None?")
                    continue

                data = source_params[name]
                c_range = qdmpy.plot.common.get_colormap_range(
                    options["colormap_range_dicts"]["magnetization_images"], data
                )
                qdmpy.plot.common.plot_image_on_ax(
                    fig, ax, options, data, name, c_map, c_range, root_name + " (mu_B/nm^2)"
                )

    if options["save_plots"]:
        fig.savefig(options["source_dir"] / (root_name + "." + options["save_fig_type"]))

    return fig


# ============================================================================


def divperp_j(options, source_params):
    """plot perpindicular divergence of J, i.e. in-plane divergence (dJ/dx + dJ/dy).

    Parameters
    ----------
    options : dict
        Generic options dict holding all the user options.
    source_params : dict
        Dictionary, key: param_keys, val: image (2D) of source field values across FOV.

    Returns
    -------
    fig : matplotlib Figure object
    """

    if source_params is None:
        return None

    figsize = mpl.rcParams["figure.figsize"].copy()
    width = len(options["recon_methods"])  # * 2  # number of rows (methods + direct/recon)
    figsize[0] *= width
    height = 1
    fig, axs = plt.subplots(height, width, figsize=figsize, constrained_layout=True)

    for m_idx, method in enumerate(options["recon_methods"]):  # # m_idx: each doublet of rows
        if f"divperp_J_{method}" not in source_params:
            warnings.warn("missing recon_method '{method}', skipping.")
            continue
        ax = axs if width == 1 else axs[m_idx]
        data = source_params[f"divperp_J_{method}"]
        title = f"Div perp ( J_{method} )"
        c_range = qdmpy.plot.common.get_colormap_range(
            options["colormap_range_dicts"]["current_div_images"], data
        )
        c_map = options["colormaps"]["current_div_images"]
        qdmpy.plot.common.plot_image_on_ax(
            fig, ax, options, data, title, c_map, c_range, "Div perp J (A/m^2)"
        )

    if options["save_plots"]:
        fig.savefig(options["source_dir"] / ("divperp_J." + options["save_fig_type"]))

    return fig

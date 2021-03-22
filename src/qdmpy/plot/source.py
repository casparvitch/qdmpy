# - coding: utf-8 -*-
"""
This module holds functions for plotting source (fields).

Functions
---------
 - `qdmpy.plot.source.`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {"qdmpy.plot.source.": True}

# ============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib_scalebar.scalebar import ScaleBar
import warnings

# ============================================================================

import qdmpy.plot.common as plot_common

# ============================================================================


def plot_source_param(
    options,
    name,  # i.e. "sig"/"ref" etc.
    param_name,  # "Jx" etc.
    source_params,
    c_map=None,
    c_map_type="source_images",
    c_range_type="percentile",
    c_range_vals=[1, 99],
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
        colormap range option (see `qdmpy.plot.common._get_colormap_range`) to use
    c_range_vals : number or list, default: [5, 95]
        passed with c_map_range to _get_colormap_range
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
        c_range = plot_common._get_colormap_range(
            {"type": c_range_type, "values": c_range_vals}, source_params[param_name]
        )
        c_map = options["colormaps"][c_map_type]

    title = f"{name}"

    plot_common.plot_image_on_ax(
        fig, ax, options, source_params[param_name], title, c_map, c_range, cbar_label
    )

    if options["save_plots"]:
        fig.savefig(options["source_dir"] / (f"{param_name}_{name}." + options["save_fig_type"]))


# ============================================================================


def plot_current(options, source_params):
    """ plots jx, jy, jnorm """
    components = {
        "x": "current_vector_images",
        "y": "current_vector_images",
        "norm": "current_norm_images",
    }
    for p in ["J" + comp for comp in components]:
        if p not in source_params:
            warnings.warn(f"bfield param '{p} missing from source_params, skipping current plot.")
            return None
        elif source_params[p] is None:
            return None

    figsize = mpl.rcParams["figure.figsize"].copy()
    width = 3
    height = 1  # number of rows
    figsize[0] *= width  # number of columns

    fig, ax = plt.subplots(height, width, figsize=figsize, constrained_layout=True)

    for i, comp in enumerate(components.keys()):
        ckey = components[comp]
        jmap = source_params["J" + comp]

        c_range = plot_common._get_colormap_range(options["colormap_range_dicts"][ckey], jmap)
        c_map = options["colormaps"][ckey]

        plot_common.plot_image_on_ax(
            fig, ax[i], options, jmap, "J" + comp, c_map, c_range, "J (A/m)"
        )

    if options["save_plots"]:
        fig.savefig(options["source_dir"] / (f"J{comp}." + options["save_fig_type"]))

    return fig


# ============================================================================


def plot_stream(options, source_params, PL_image_ROI=None):
    components = ["x", "y", "norm"]
    for p in ["J" + comp for comp in components]:
        if p not in source_params:
            warnings.warn(f"bfield param '{p} missing from source_params, skipping stream plot.")
            return None
        elif source_params[p] is None:
            return None

    fig, ax = plt.subplots(constrained_layout=True)

    if PL_image_ROI is not None:
        im = ax.imshow(
            PL_image_ROI,
            cmap="Greys_r",
            vmin=np.nanmin(PL_image_ROI),
            vmax=np.nanmax(PL_image_ROI),
            alpha=options["streamplot_PL_alpha"],
        )

    shp = source_params["Jx"].shape

    strm = ax.streamplot(
        np.arange(shp[1]),
        np.arange(shp[0]),
        source_params["Jx"],
        -source_params["Jy"],
        color=source_params["Jnorm"],
        cmap=options["colormaps"]["current_norm_images"],
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

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    cbar.ax.set_ylabel("J (A/m)", rotation=270)
    cbar.outline.set_linewidth(0.5)

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

    if options["save_plots"]:
        fig.savefig(options["source_dir"] / ("Jstream." + options["save_fig_type"]))

    return fig

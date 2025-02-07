# -*- coding: utf-8 -*-
"""
This module holds tools for loading raw data etc. and reshaping to a usable format.

Also contained here are functions for defining regions/areas of interest within the
larger image dataset, that are then used in consequent functions, as well as the
general options dictionary.

Functions
---------
 - `qdmpy.plot.common.set_mpl_rcparams`
 - `qdmpy.plot.common.plot_image`
 - `qdmpy.plot.common.plot_image_on_ax`
 - `qdmpy.plot.common._add_colorbar`
 - `qdmpy.plot.common.get_colormap_range`
 - `qdmpy.plot.common._min_max`
 - `qdmpy.plot.common._strict_range`
 - `qdmpy.plot.common._min_max_sym_mean`
 - `qdmpy.plot.common._min_max_sym_zero`
 - `qdmpy.plot.common._deviation_from_mean`
 - `qdmpy.plot.common._percentile`
 - `qdmpy.plot.common._percentile_sym_zero`
"""
# ===========================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.plot.common.set_mpl_rcparams": True,
    "qdmpy.plot.common.plot_image": True,
    "qdmpy.plot.common.plot_image_on_ax": True,
    "qdmpy.plot.common._add_colorbar": True,
    "qdmpy.plot.common.get_colormap_range": True,
    "qdmpy.plot.common._min_max": True,
    "qdmpy.plot.common._strict_range": True,
    "qdmpy.plot.common._min_max_sym_mean": True,
    "qdmpy.plot.common._min_max_sym_zero": True,
    "qdmpy.plot.common._deviation_from_mean": True,
    "qdmpy.plot.common._percentile": True,
    "qdmpy.plot.common._percentile_sym_zero": True,
}

# ===========================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import matplotlib.patches

# ===========================================================================

from qdmpy.shared.misc import warn

# ===========================================================================


def set_mpl_rcparams(options):
    """Reads matplotlib-relevant parameters in options and used to define matplotlib rcParams"""
    for optn, val in options["mpl_rcparams"].items():
        if isinstance(val, (list, tuple)):
            val = tuple(val)
        try:
            mpl.rcParams[optn] = val
        except KeyError:
            warn(f"mpl rcparams key '{optn}' not recognised as a valid rc parameter.")


# ===========================================================================


def plot_image(options, image_data, title, c_map, c_range, c_label):
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

    Returns
    -------
    fig : matplotlib Figure object
    ax : matplotlib Axis object
    """

    fig, ax = plt.subplots()

    fig, ax = plot_image_on_ax(
        fig, ax, options, image_data, title, c_map, c_range, c_label
    )

    np.savetxt(options["data_dir"] / f"{title}.txt", image_data)
    if options["save_plots"]:
        fig.savefig(options["output_dir"] / (f"{title}." + options["save_fig_type"]))

    return fig, ax


# ============================================================================


def plot_image_on_ax(fig, ax, options, image_data, title, c_map, c_range, c_label):
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

    Returns
    -------
    fig : matplotlib Figure object
    ax : matplotlib Axis object
    """

    im = ax.imshow(image_data, cmap=c_map, vmin=c_range[0], vmax=c_range[1])

    ax.set_title(title)

    cbar = _add_colorbar(im, fig, ax)
    cbar.ax.set_ylabel(c_label, rotation=270)

    if options["show_scalebar"]:
        pixel_size = (
            options["system"].get_raw_pixel_size(options) * options["total_bin"]
        )
        scalebar = ScaleBar(pixel_size)
        ax.add_artist(scalebar)

    if options["polygon_nodes"] and options["annotate_polygons"]:
        for p in options["polygon_nodes"]:
            # polygons reversed to (x,y) indexing for patch
            ax.add_patch(
                matplotlib.patches.Polygon(
                    np.dstack((p[:, 1], p[:, 0]))[0],
                    **options["polygon_patch_params"],
                )
            )

    if not options["show_tick_marks"]:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    return fig, ax


# ============================================================================


def _add_colorbar(im, fig, ax, aspect=20, pad_fraction=1, **kwargs):
    """
    Adds a colorbar to matplotlib axis

    Arguments
    ---------
    im : image as returned by ax.imshow
    fig : matplotlib Figure object
    ax : matplotlib Axis object

    Returns
    -------
    cbar : matplotlib colorbar object

    Optional Arguments
    ------------------
    aspect : int
        Reciprocal of aspect ratio passed to new colorbar axis width. Default: 20.
    pad_fraction : int
        Fraction of new colorbar axis width to pad from image. Default: 1.

    **kwargs : other keyword arguments
        Passed to fig.colorbar.

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


def get_colormap_range(c_range_dict, image):
    """
    Produce a colormap range to plot image from, using the options in c_range_dict.

    Arguments
    ---------
    c_range_dict : dict
        Dictionary with key 'values', used to accompany some of the options below,
        as well as a 'type', one of :
         - "min_max" : map between minimum and maximum values in image.
         - "deviation_from_mean" : requires c_range_dict["values"] be a float
           between 0 and 1 'dev'. Maps between (1 - dev) * mean and (1 + dev) * mean.
         - "min_max_symmetric_about_mean" : map symmetrically about mean, capturing all values
           in image (default).
         - "min_max_symmetric_about_zero" : map symmetrically about zero, capturing all values
           in image.
         - "percentile" : requires c_range_dict["values"] be a list of two numbers between 0 and
           100. Maps the range between those percentiles of the data.
         - "percentile_symmetric_about_zero" : requires c_range_dict["values"] be a list of two
           numbers between 0 and 100. Maps symmetrically about zero, capturing all values between
           those percentiles in the data (plus perhaps a bit more to ensure symmety)
         - "strict_range" : requires c_range_dict["values"] be list length 2 of floats or ints.
           Maps colors between the values given.
         - "mean_plus_minus" : mean plus or minus this value. c_range_dict["values"] must be an int
           or float.
        as well as accompanying 'values' key, used for some of the options below

    image : array-like
        2D array (image) that fn returns colormap range for.

    Returns
    -------
    c_range : list length 2
        i.e. [min value to map to a color, max value to map to a color]
    """

    # mostly these are just checking that the input values are valid
    # pretty badly written, I apoligise (there's a reason it's hidden all the way down here...)

    warning_messages = {
        "deviation_from_mean": """Invalid c_range_dict['values'] encountered.
        For c_range type 'deviation_from_mean', c_range_dict['values'] must be a float,
        between 0 and 1. Changing to 'min_max_symmetric_about_mean' c_range.""",
        "strict_range": """Invalid c_range_dict['values'] encountered.
        For c_range type 'strict_range', c_range_dict['values'] must be a a list of length 2,
        with elements that are floats or ints.
        Changing to 'min_max_symmetric_about_mean' c_range.""",
        "mean_plus_minus": """Invalid c_range_dict['values'] encountered.
        For c_range type 'mean_plus_minus', c_range_dict['values'] must be an int or float.
        Changing to 'min_max_symmetric_about_mean' c_range.""",
        "percentile": """Invalid c_range_dict['values'] encountered.
        For c_range type 'percentile', c_range_dict['values'] must be a list of length 2,
         with elements (preferably ints) in [0, 100].
         Changing to 'min_max_symmetric_about_mean' c_range.""",
        "percentile_symmetric_about_zero": """Invalid c_range_dict['values'] encountered.
        For c_range type 'percentile', c_range_dict['values'] must be a list of length 2,
         with elements (preferably ints) in [0, 100].
         Changing to 'min_max_symmetric_about_mean' c_range.""",
    }
    try:
        auto_sym_zero = c_range_dict["auto_sym_zero"]
        if auto_sym_zero is None:
            auto_sym_zero = True
    except KeyError:
        auto_sym_zero = True

    c_range_type = c_range_dict["type"]

    if c_range_type not in ["percentile", "min_max"]:
        auto_sym_zero = False

    if auto_sym_zero and np.any(image < 0) and np.any(image > 0):
        if c_range_type == "min_max":
            c_range_type = "min_max_symmetric_about_zero"
        elif c_range_type == "percentile":
            c_range_type = "percentile_symmetric_about_zero"
    try:
        c_range_values = c_range_dict["values"]
    except KeyError:
        c_range_values = None

    range_calculator_dict = {
        "min_max": _min_max,
        "deviation_from_mean": _deviation_from_mean,
        "min_max_symmetric_about_mean": _min_max_sym_mean,
        "min_max_symmetric_about_zero": _min_max_sym_zero,
        "percentile": _percentile,
        "percentile_symmetric_about_zero": _percentile_sym_zero,
        "strict_range": _strict_range,
        "mean_plus_minus": _mean_plus_minus,
    }

    if c_range_type == "strict_range":
        if (
            not isinstance(c_range_values, (list, tuple))
            or len(c_range_values) != 2  # noqa: W503
            or (not isinstance(c_range_values[0], (float, int)))  # noqa: W503
            or (not isinstance(c_range_values[1], (float, int)))  # noqa: W503
            or c_range_values[0] > c_range_values[1]  # noqa: W503
        ):
            warn(warning_messages[c_range_type])
            return _min_max_sym_mean(image, c_range_values)
    elif c_range_type == "mean_plus_minus":
        if not isinstance(c_range_values, (float, int)):
            warn(warning_messages[c_range_type])
            return _min_max_sym_mean(image, c_range_values)
    elif c_range_type == "deviation_from_mean":
        if (
            not isinstance(c_range_values, (float, int))
            or c_range_values < 0  # noqa: W503
            or c_range_values > 1  # noqa: W503
        ):
            warn(warning_messages[c_range_type])
            return _min_max_sym_mean(image, c_range_values)

    elif c_range_type.startswith("percentile"):
        if (
            not isinstance(c_range_values, (list, tuple))
            or len(c_range_values) != 2  # noqa: W503
            or not isinstance(c_range_values[0], (float, int))
            or not isinstance(c_range_values[1], (float, int))
            or not 100 >= c_range_values[0] >= 0
            or not 100 >= c_range_values[1] >= 0
        ):
            warn(warning_messages[c_range_type])
            return _min_max_sym_mean(image, c_range_values)
    return range_calculator_dict[c_range_type](image, c_range_values)


# ============================


def _min_max(image, c_range_values):
    """
    Map between minimum and maximum values in image

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : unknown (depends on user settings)
        See `qdmpy.plot.common.get_colormap_range`
    """
    return [np.nanmin(image), np.nanmax(image)]


def _strict_range(image, c_range_values):
    """
    Map between c_range_values

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : unknown (depends on user settings)
        See `qdmpy.plot.common.get_colormap_range`
    """
    return list(c_range_values)


def _min_max_sym_mean(image, c_range_values):
    """
    Map symmetrically about mean, capturing all values in image.

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : unknown (depends on user settings)
        See `qdmpy.plot.common.get_colormap_range`
    """
    minimum = np.nanmin(image)
    maximum = np.nanmax(image)
    mean = np.mean(image)
    max_distance_from_mean = np.max([abs(maximum - mean), abs(minimum - mean)])
    return [mean - max_distance_from_mean, mean + max_distance_from_mean]


def _min_max_sym_zero(image, c_range_values):
    """
    Map symmetrically about zero, capturing all values in image.

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : unknown (depends on user settings)
        See `qdmpy.plot.common.get_colormap_range`
    """
    min_abs = np.abs(np.nanmin(image))
    max_abs = np.abs(np.nanmax(image))
    larger = np.nanmax([min_abs, max_abs])
    return [-larger, larger]


def _deviation_from_mean(image, c_range_values):
    """
    Map a (decimal) deviation from mean, i.e. between (1 - dev) * mean and (1 + dev) * mean

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : unknown (depends on user settings)
        See `qdmpy.plot.common.get_colormap_range`
    """
    return [
        (1 - c_range_values) * np.mean(image),
        (1 + c_range_values) * np.mean(image),
    ]


def _percentile(image, c_range_values):
    """
    Maps the range between two percentiles of the data.

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : unknown (depends on user settings)
        See `qdmpy.plot.common.get_colormap_range`
    """
    return np.nanpercentile(image, c_range_values)


def _percentile_sym_zero(image, c_range_values):
    """
    Maps the range between two percentiles of the data, but ensuring symmetry about zero

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : unknown (depends on user settings)
        See `qdmpy.plot.common.get_colormap_range`
    """
    plow, phigh = np.nanpercentile(image, c_range_values)  # e.g. [10, 90]
    val = max(abs(plow), abs(phigh))
    return [-val, val]


def _mean_plus_minus(image, c_range_values):
    """
    Maps the range to mean +- value given in c_range_values

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : unknown (depends on user settings)
        See `qdmpy.plot.common.get_colormap_range`
    """
    mean = np.mean(image)
    return [mean - c_range_values, mean + c_range_values]


# ============================================================================

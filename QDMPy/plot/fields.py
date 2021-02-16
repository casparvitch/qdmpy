# - coding: utf-8 -*-
"""
This module holds functions for plotting fields.

Functions
---------
 - `QDMPy.plot.fields.plot_bnvs_and_dshifts`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {"QDMPy.plot.fields.plot_bnvs_and_dshifts": True}

# ============================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================================================================

import QDMPy.plot.common as plot_common

# ============================================================================


def plot_bnvs_and_dshifts(options, name, bnvs, dshifts):
    """
    Plots bnv arrays above dshift arrays.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    name : str
        Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames

    bnvs : list
        List of np arrays (2D) giving B_NV for each NV family/orientation.
        If num_peaks is odd, the bnv is given as the shift of that peak,
        and the dshifts is left as np.nans.

    dshifts : list
        List of np arrays (2D) giving the D (~DFS) of each NV family/orientation.
        If num_peaks is odd, the bnv is given as the shift of that peak,
        and the dshifts is left as np.nans.

    Returns
    -------
    fig : matplotlib Figure object
    """

    if not bnvs and not dshifts:
        return None

    figsize = mpl.rcParams["figure.figsize"].copy()
    width = len(bnvs)
    height = 2 if dshifts else 1
    figsize[0] *= width  # number of columns
    figsize[1] *= height  # number of rows

    figsize[0] *= 3 / 4
    figsize[1] *= 3 / 4

    fig, axs = plt.subplots(height, width, figsize=figsize, constrained_layout=True)

    c_map = options["colormaps"]["bnv_images"]
    # axs index: axs[row, col]
    for i, bnv in enumerate(bnvs):
        c_range = plot_common._get_colormap_range(
            options["colormap_range_dicts"]["bnv_images"], bnv
        )
        title = f"{name} B NV_{i}"
        ax = axs[i] if not dshifts else axs[0, i]
        plot_common.plot_image_on_ax(fig, ax, options, bnv, title, c_map, c_range, "B (G)")

    c_map = options["colormaps"]["dshift_images"]
    for i, dshift in enumerate(dshifts):
        c_range = plot_common._get_colormap_range(
            options["colormap_range_dicts"]["dshift_images"], dshift
        )
        title = f"{name} D_{i}"
        plot_common.plot_image_on_ax(
            fig, axs[1, i], options, dshift, title, c_map, c_range, "D (MHz)"
        )

    if options["save_plots"]:
        fig.savefig(options["sub_ref_dir"] / (f"Bnv_{name}." + options["save_fig_type"]))

    return fig


# ============================================================================

# - coding: utf-8 -*-
"""
This module holds functions for plotting fields.

Functions
---------
 - `QDMPy.plot.field.plot_bnvs_and_dshifts`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {"QDMPy.plot.fields.plot_bnvs_and_dshifts": True}

# ============================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

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

    figsize[0] *= 3 / 4  # (usually 4 images wide not 3...)
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


def plot_bfield(options, name, field_params):
    """docstring"""
    if field_params is None:
        return None

    components = ["x", "y", "z"]

    for p in ["B" + comp for comp in components]:
        if p not in field_params:
            raise ValueError(f"bfield param '{p} missing from field_params, skipping bfield plot.")
            return None

    bfields = [field_params["B" + comp] for comp in components]

    figsize = mpl.rcParams["figure.figsize"].copy()
    width = len(bfields)
    height = 1  # number of rows
    figsize[0] *= width  # number of columns

    fig, ax = plt.subplots(height, width, figsize=figsize, constrained_layout=True)

    c_map = options["colormaps"]["bfield_images"]

    for i, bcomponent in enumerate(bfields):
        c_range = plot_common._get_colormap_range(
            options["colormap_range_dicts"]["bfield_images"], bcomponent
        )
        title = f"{name} B" + components[i]
        plot_common.plot_image_on_ax(
            fig, ax[i], options, bcomponent, title, c_map, c_range, "B (G)"
        )

    if options["save_plots"]:
        fig.savefig(options["sub_ref_dir"] / (f"Bfield_{name}." + options["save_fig_type"]))

    return fig


# ============================================================================


def plot_dshift_fit(options, name, field_params):
    if field_params is None:
        return None

    if "D" not in field_params:
        raise ValueError("'D' param missing from field_params, skipping Dshift_fit plot.")
        return None

    fig, ax = plt.subplots(constrained_layout=True)

    c_map = options["colormaps"]["dshift_images"]
    c_range = plot_common._get_colormap_range(
        options["colormap_range_dicts"]["dshift_images"], field_params["D"]
    )
    title = f"{name} Dshift fit"
    plot_common.plot_image_on_ax(
        fig, ax, options, field_params["D"], title, c_map, c_range, "D (MHz)"
    )

    if options["save_plots"]:
        fig.savefig(options["sub_ref_dir"] / (f"Dshift_fit_{name}." + options["save_fig_type"]))

    return fig


# ============================================================================


def plot_ham_residual(options, name, field_params):

    if field_params is None:
        return None

    if "residual_ham" not in field_params:
        raise ValueError(
            "'residual_ham' param missing from field_params, skipping residual ham plot."
        )
        return None

    fig, ax = plt.subplots(constrained_layout=True)

    c_range = plot_common._get_colormap_range(
        options["colormap_range_dicts"]["residual_images"], field_params["residual_ham"]
    )
    c_map = options["colormaps"]["residual_images"]

    title = f"{name} residual (hamiltonian fit)"

    plot_common.plot_image_on_ax(
        fig,
        ax,
        options,
        field_params["residual_ham"],
        title,
        c_map,
        c_range,
        "Error: sum( || residual(params) || ) over bnvs/freqs (a.u.)",
    )

    if options["save_plots"]:
        fig.savefig(options["sub_ref_dir"] / (f"residual_ham_{name}." + options["save_fig_type"]))

    return fig


# ============================================================================


def plot_strain(options, name, field_params):
    raise NotImplementedError()


# ============================================================================


def plot_efield(options, name, field_params):
    raise NotImplementedError()


# ============================================================================

# - coding: utf-8 -*-
"""
This module holds functions for plotting fields.

Functions
---------
 - `qdmpy.plot.field.bnvs_and_dshifts`
 - `qdmpy.plot.field.bfield`
 - `qdmpy.plot.field.dshift_fit`
 - `qdmpy.plot.field.field_residual`
 - `qdmpy.plot.field.field_param`
 - `qdmpy.plot.field.field_param_flattened`
 - `qdmpy.plot.field.bfield_consistency`
 - `qdmpy.plot.field.bfield_theta_phi`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.plot.fields.bnvs_and_dshifts": True,
    "qdmpy.plot.fields.bfield": True,
    "qdmpy.plot.fields.dshift_fit": True,
    "qdmpy.plot.fields.field_residual": True,
    "qdmpy.plot.fields.field_param": True,
    "qdmpy.plot.fields.field_param_flattened": True,
    "qdmpy.plot.fields.bfield_consistency": True,
    "qdmpy.plot.fields.bfield_theta_phi": True,
}

# ============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

# ============================================================================

import qdmpy.plot.common
import qdmpy.field

# ============================================================================


def bnvs_and_dshifts(options, name, bnvs, dshifts):
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

    if not bnvs and (dshifts is None or not dshifts):
        return None

    figsize = mpl.rcParams["figure.figsize"].copy()
    width = len(bnvs)
    height = 2 if dshifts is not None and len(dshifts) > 0 else 1
    figsize[0] *= width  # number of columns
    figsize[1] *= height  # number of rows

    # standardise figsize of output
    # figsize[0] *= 3 / 4  # (usually 4 images wide not 3...)
    # figsize[1] *= 3 / 4

    fig, axs = plt.subplots(height, width, figsize=figsize)

    c_map = options["colormaps"]["bnv_images"]
    # axs index: axs[row, col]
    for i, bnv in enumerate(bnvs):
        data = (
            -bnv
            if options["plot_bnv_flip_with_bias_mag"]
            and options["bias_field_spherical_deg_gauss"][0] < 0
            else bnv
        )

        c_range = qdmpy.plot.common.get_colormap_range(
            options["colormap_range_dicts"]["bnv_images"], data
        )
        title = f"{name} B NV_{i}"
        if width == 1 and (dshifts is None or not dshifts):
            ax = axs
        elif width == 1:
            ax = axs[0]
        elif dshifts is None or not dshifts:
            ax = axs[i]
        else:
            ax = axs[0, i]
        qdmpy.plot.common.plot_image_on_ax(fig, ax, options, data, title, c_map, c_range, "B (G)")
    c_map = options["colormaps"]["dshift_images"]
    for i, dshift in enumerate(dshifts):
        data = (
            -dshift
            if options["plot_bnv_flip_with_bias_mag"]
            and options["bias_field_spherical_deg_gauss"][0] < 0
            else dshift
        )
        c_range = qdmpy.plot.common.get_colormap_range(
            options["colormap_range_dicts"]["dshift_images"], data
        )
        title = f"{name} D_{i}"
        ax = axs[1] if width == 1 else axs[1, i]
        qdmpy.plot.common.plot_image_on_ax(
            fig, ax, options, data, title, c_map, c_range, "D (MHz)"
        )

    if options["save_plots"]:
        fig.savefig(options["field_dir"] / (f"Bnv_{name}." + options["save_fig_type"]))

    return fig


# ============================================================================


def bfield(options, name, field_params):
    """Plot Bxyz if available as keys in 'field_params'.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    name : str
        Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames
    field_params : dict
        Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV.

    Returns
    -------
    fig : matplotlib Figure object
    """
    if field_params is None:
        return None

    components = ["x", "y", "z"]

    other_selectors = ["bground", "full", "mask"]

    if name in other_selectors:
        components = [i + "_" + name for i in components]

    for p in ["B" + comp for comp in components]:
        if p not in field_params:
            warnings.warn(f"bfield param '{p} missing from field_params, skipping bfield plot.")
            return None
        elif field_params[p] is None:
            return None

    bfields = [field_params["B" + comp] for comp in components]

    figsize = mpl.rcParams["figure.figsize"].copy()
    width = len(bfields)
    height = 1  # number of rows
    figsize[0] *= width  # number of columns

    fig, ax = plt.subplots(height, width, figsize=figsize)

    c_map = "Greys" if name == "mask" else options["colormaps"]["bfield_images"]

    for i, bcomponent in enumerate(bfields):
        c_range = (
            [0, 1]
            if name == "mask"
            else qdmpy.plot.common.get_colormap_range(
                options["colormap_range_dicts"]["bfield_images"], bcomponent
            )
        )
        if name not in other_selectors:
            title = f"{name} B" + components[i]
        else:
            title = "B" + components[i]
        qdmpy.plot.common.plot_image_on_ax(
            fig, ax[i], options, bcomponent, title, c_map, c_range, "B (G)"
        )

    if options["save_plots"]:
        fig.savefig(options["field_dir"] / (f"Bfield_{name}." + options["save_fig_type"]))

    return fig


# ============================================================================


def dshift_fit(options, name, field_params):
    """Plot dshift (fit) if available as keys in 'field_params'.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    name : str
        Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames
    field_params : dict
        Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV.

    Returns
    -------
    fig : matplotlib Figure object
    """
    if field_params is None:
        return None

    if "D" not in field_params:
        warnings.warn("'D' param missing from field_params, skipping Dshift_fit plot.")
        return None
    elif field_params["D"] is None:
        return None

    fig, ax = plt.subplots()

    c_map = options["colormaps"]["dshift_images"]
    c_range = qdmpy.plot.common.get_colormap_range(
        options["colormap_range_dicts"]["dshift_images"], field_params["D"]
    )
    title = f"{name} Dshift fit"
    qdmpy.plot.common.plot_image_on_ax(
        fig, ax, options, field_params["D"], title, c_map, c_range, "D (MHz)"
    )

    if options["save_plots"]:
        fig.savefig(options["field_dir"] / (f"Dshift_fit_{name}." + options["save_fig_type"]))

    return fig


# ============================================================================


def field_residual(options, name, field_params):
    """Plot residual if available as keys in 'field_params' (as 'residual_field').

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    name : str
        Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames
    field_params : dict
        Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV.

    Returns
    -------
    fig : matplotlib Figure object
    """

    if field_params is None:
        return None

    if "residual_field" not in field_params:
        warnings.warn(
            "'residual_field' param missing from field_params, skipping field residual plot."
        )
        return None
    elif field_params["residual_field"] is None:
        return None

    fig, ax = plt.subplots()

    c_range = qdmpy.plot.common.get_colormap_range(
        options["colormap_range_dicts"]["residual_images"], field_params["residual_field"]
    )
    c_map = options["colormaps"]["residual_images"]

    title = f"{name} residual (hamiltonian fit)"

    qdmpy.plot.common.plot_image_on_ax(
        fig,
        ax,
        options,
        field_params["residual_field"],
        title,
        c_map,
        c_range,
        "Error: sum( || residual(params) || ) over bnvs/freqs (a.u.)",
    )

    if options["save_plots"]:
        fig.savefig(options["field_dir"] / (f"residual_field_{name}." + options["save_fig_type"]))

    return fig


# ============================================================================


def field_param(
    options,
    name,  # i.e. "sig"/"ref" etc.
    param_name,  # "Bx" etc.
    field_params,
    c_map=None,
    c_map_type="param_images",
    c_range_type="percentile",
    c_range_vals=(5, 95),
    cbar_label="",
):
    """plot a given field param.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    name : str
        Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames.
    param_name : str
        Name of specific param to plot (e.g. "Bx" etc.).
    field_params : dict
        Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV.
    c_map : str, default: None
        colormap to use overrides c_map_type and c_range_type.
    c_map_type : str, default: "param_images"
        colormap type to search options (options["colormaps"][c_map_type]) for.
    c_map_range : str, default: "percentile"
        colormap range option (see `qdmpy.plot.common.get_colormap_range`) to use.
    c_range_vals : number or list, default: (5, 95)
        passed with c_map_range to get_colormap_range
    cbar_label : str, default:""
        label to chuck on ye olde colorbar (z-axis label).

    Returns
    -------
    fig : matplotlib Figure object
    """
    if field_params is None:
        return None
    if param_name not in field_params:
        warnings.warn(f"No param (key) called {param_name} in field_params.")
        return None

    fig, ax = plt.subplots()

    if c_map is None:
        c_range = qdmpy.plot.common.get_colormap_range(
            {"type": c_range_type, "values": c_range_vals}, field_params[param_name]
        )
        c_map = options["colormaps"][c_map_type]

    title = f"{name}: {param_name}"

    qdmpy.plot.common.plot_image_on_ax(
        fig, ax, options, field_params[param_name], title, c_map, c_range, cbar_label
    )

    if options["save_plots"]:
        fig.savefig(options["field_dir"] / (f"{param_name}_{name}." + options["save_fig_type"]))

    return fig


# ============================================================================


def field_param_flattened(
    options,
    name,
    param_name,
    field_params,
    sigmas=None,
    plot_sigmas=True,
    plot_bounds=True,
    plot_guess=True,
    y_label="",
    errorevery=1,
):
    """plot a field param flattened vs pixel number.

    Sigmas are optionally utilised as errobars.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    name : str
        Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames.
    param_name : str
        Name of key in field_params (and sigmas) to plot.
    field_params : dict
        Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV.
    sigmas : dict, default: None
        Dictionary, key: param_keys, val: image (2D) of (field) sigma values across FOV.
    plot_sigmas : bool, default: True
        If true & sigmas is not None, sigmas are used as errorbars.
    plot_bounds : bool, default: True
        If True, (field) fit bound is annotated as horizontal dashed lines.
    plot_guess: bool, default: True
        If True, (field) fit guess is annotates as horizontal dashed line.
    y_label : str, default: ""
        Label to chuck on y axis.
    errorevery : int, default: 1
        Plot errorbar every 'errorevery' data point (so it doesn't get too messy).

    Returns
    -------
    fig : matplotlib Figure object
    """
    if field_params is None:
        return None
    if param_name not in field_params:
        warnings.warn(f"Couldn't find {param_name} in field_params.")
        return None
    if not sigmas or param_name not in sigmas:
        plot_sigmas = False

    figsize = mpl.rcParams["figure.figsize"].copy()
    figsize[0] *= 2  # make some extra space in width...

    fig, ax = plt.subplots(figsize=figsize)

    # get guesses and bounds
    if param_name.startswith("residual"):
        guess = None
        bounds = None
        plot_sigmas = False
    elif options["field_method_used"] == "hamiltonian_fitting":
        guess_dict, bound_dict = qdmpy.field.get_ham_guess_and_bounds(options)
        guess = guess_dict[param_name]
        bounds = bound_dict[param_name]
    else:
        bounds = None  # no bounds if not a fit
        comps = ["Bx", "By", "Bz"]
        if param_name in comps:
            bguesses = options["bias_field_cartesian_gauss"]
            guess = bguesses[comps.index(param_name)]

    if not plot_guess:
        guess = None

    ax.set_xlabel("Pixel # (flattened)")
    ax.set_ylabel(y_label)
    ax.grid()

    # if is a masked array, converts all masked vals to np.nan
    yvals = np.ma.filled(field_params[param_name], fill_value=np.nan).flatten()

    color = "blue"
    if plot_sigmas:
        xvals = list(range(len(yvals)))
        yerr = np.ma.filled(sigmas[param_name], fill_value=np.nan).flatten()
        if np.all(np.isnan(yerr)):
            yerr = None
            warnings.warn("All sigmas were nans.")
        ax.errorbar(
            xvals,
            yvals,
            xerr=None,
            yerr=yerr,
            marker="o",
            mfc="w",
            ms=mpl.rcParams["lines.markersize"],
            mec=color,
            ecolor=color,
            ls="",
            zorder=20,
            errorevery=errorevery,
        )
    else:
        ax.plot(
            yvals,
            marker="o",
            mfc="w",
            ms=mpl.rcParams["lines.markersize"],
            mec=color,
            ls="",
            zorder=20,
        )
    legend_names = [name + "_" + param_name]
    custom_lines = [
        Line2D(
            [0],
            [0],
            marker="o",
            mfc="w",
            ms=mpl.rcParams["lines.markersize"] * 2,
            mec=color,
            ls="",
        )
    ]
    if guess is not None:
        ax.axhline(
            guess,
            ls=(0, (1, 1)),
            c="k",
            zorder=10,
            lw=mpl.rcParams["lines.linewidth"] * 2,
        )
        legend_names.append("Guess")
        custom_lines.append(
            Line2D([0], [0], color="k", ls=(0, (1, 1)), lw=mpl.rcParams["lines.linewidth"] * 2)
        )
    if bounds is not None and isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        for b in bounds:
            ax.axhline(b, ls=(0, (2, 1)), c="grey", zorder=9)
        legend_names.append("Bounds")
        custom_lines.append(
            Line2D([0], [0], color="k", ls=(0, (2, 1)), lw=mpl.rcParams["lines.linewidth"])
        )

    ax.legend(
        custom_lines,
        legend_names,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0,
        frameon=False,
        ncol=len(legend_names),
        fontsize="medium",
    )

    if options["save_plots"]:
        fig.savefig(
            options["field_dir"]
            / (f"{name}_{param_name}_fit_flattened." + options["save_fig_type"])
        )

    return fig


# ============================================================================


def strain(options, name, field_params):
    raise NotImplementedError()


# ============================================================================


def efield(options, name, field_params):
    raise NotImplementedError()


# ============================================================================


def bfield_consistency(options, name, field_params):
    """plot bfield vs bfield_recon

    Parameters
    ----------
    options : dict
        Generic options dict holding all the user options.
    name : str
        Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames.
    field_params : dict
        Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV.

    Returns
    -------
    fig : matplotlib Figure object
    """

    if field_params is None:
        return None

    components = ["x", "y", "z"]

    for p in ["B" + comp for comp in components]:
        if p not in field_params:
            warnings.warn(f"bfield param '{p} missing from field_params, skipping bfield plot.")
            return None
        elif field_params[p] is None:
            return None
    for p in ["B" + comp + "_recon" for comp in components]:
        if p not in field_params:
            warnings.warn(f"bfield param '{p} missing from field_params, skipping bfield plot.")
            return None
        elif field_params[p] is None:
            return None

    bfields = [field_params["B" + comp] for comp in components]
    bfield_recons = [field_params["B" + comp + "_recon"] for comp in components]

    figsize = mpl.rcParams["figure.figsize"].copy()
    width = 3
    height = 2  # number of rows
    figsize[0] *= width  # number of columns
    figsize[1] *= height

    fig, ax = plt.subplots(height, width, figsize=figsize)

    c_map = options["colormaps"]["bfield_images"]

    bfield_ranges = []  # store bfield map cmap ranges to use for recon as well
    for i, bcomponent in enumerate(bfields):
        c_range = qdmpy.plot.common.get_colormap_range(
            options["colormap_range_dicts"]["bfield_images"], bcomponent
        )
        bfield_ranges.append(c_range)
        title = "B" + components[i]
        qdmpy.plot.common.plot_image_on_ax(
            fig, ax[0, i], options, bcomponent, title, c_map, c_range, "B (G)"
        )

    for i, bcomp_recon in enumerate(bfield_recons):
        title = "B" + components[i] + "_recon"
        qdmpy.plot.common.plot_image_on_ax(
            fig, ax[1, i], options, bcomp_recon, title, c_map, bfield_ranges[i], "B (G)"
        )

    if options["save_plots"]:
        fig.savefig(options["field_dir"] / (f"Bfield_{name}_recon." + options["save_fig_type"]))

    return fig


# ============================================================================


def bfield_theta_phi(
    options,
    name,
    field_params,
    c_map=None,
    c_map_type="bfield_images",
    c_range_type="percentile",
    c_range_vals=(5, 95),
    cbar_label="",
):
    """Plots B_theta_phi if found in field_params (the vector field projected onto
    some direction).

    Parameters
    ----------
    options : dict
        Generic options dict holding all the user options.
    name : str
        Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames.
    field_params : dict
        Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV.
    c_map : str, default: None
        colormap to use overrides c_map_type and c_range_type.
    c_map_type : str, default: "bfield_images"
        colormap type to search options (options["colormaps"][c_map_type]) for.
    c_map_range : str, default: "percentile"
        colormap range option (see `qdmpy.plot.common.get_colormap_range`) to use.
    c_range_vals : number or list, default: (5, 95)
        passed with c_map_range to get_colormap_range.
    c_bar_label : str, default:""
        label to chuck on ye olde colorbar (z-axis label).
    Returns
    -------
    fig : matplotlib Figure object
    """
    if field_params is None:
        return None

    if "B_theta_phi" not in field_params:
        warnings.warn("Couldn't find 'B_theta_phi' in field_params.")
        return None

    b = field_params["B_theta_phi"]
    theta, phi = options["bfield_proj_angles_(deg)"]

    fig, ax = plt.subplots()

    if c_map is None:
        c_range = qdmpy.plot.common.get_colormap_range(
            {"type": c_range_type, "values": c_range_vals}, b
        )
        c_map = options["colormaps"][c_map_type]

    title = f"{name}: B_theta_{np.round(theta,1)}_phi_{np.round(phi,1)}"

    qdmpy.plot.common.plot_image_on_ax(fig, ax, options, b, title, c_map, c_range, cbar_label)

    if options["save_plots"]:
        fig.savefig(
            options["field_dir"]
            / (
                f"B_theta_{int(np.round(theta))}_phi_{int(np.round(phi))}_{name}."
                + options["save_fig_type"]
            )
        )
    return fig

# -*- coding: utf-8 -*-
"""
Interface for source sub-module.

Functions
---------
 - `qdmpy.source.interface.odmr_source_retrieval`
 - `qdmpy.source.interface.get_current_density`
 - `qdmpy.source.interface.get_magnetisation`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.source.interface.odmr_source_retrieval": True,
    "qdmpy.source.interface.get_current_density": True,
    "qdmpy.source.interface.get_magnetisation": True,
}

# ============================================================================

import warnings
import numpy as np

# ============================================================================

import qdmpy.fourier
import qdmpy.field
import qdmpy.io
import qdmpy.itool

# ============================================================================


def odmr_source_retrieval(options, bnvs, field_params):
    """
    Calculates source field that created field measured by bnvs and field params.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    bnvs : list
        List of bnv results (each a 2D image).
    sig_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of field values across FOV.

    Returns
    -------
    source_params : dict
        Dictionary, key: param_keys, val: image (2D) of source field values across FOV.


    For methodology see
        D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky,
        J.-P. Tetienne, and L. C. L. Hollenberg,
        Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field
        Measurements, Phys. Rev. Applied 14, 024076 (2020).
        https://doi.org/10.1103/PhysRevApplied.14.024076
        https://arxiv.org/abs/2005.06788
    """
    if options["source_type"] is None:
        return None

    # generate output directories
    qdmpy.io.source.prep_output_directories(options)

    # do what was asked
    source_fns = {"current_density": get_current_density, "magnetisation": get_magnetisation}
    source_params = source_fns[options["source_type"]](options, bnvs, field_params)

    if source_params is None:
        return None

    # zero-point normalisation
    norm_region = options["zero_point_normalisation_region"]
    if norm_region is not None:
        x_min = norm_region[0][0]
        x_max = norm_region[1][0]
        y_min = norm_region[0][1]
        y_max = norm_region[1][1]
        for key in source_params.keys():
            source_params[key] -= np.nanmean(source_params[key][y_min:y_max, x_min:x_max])
            source_params[key] -= np.nanmean(source_params[key][y_min:y_max, x_min:x_max])

    options["source_params"] = [i for i in source_params.keys()]
    return source_params


# ============================================================================


def get_current_density(
    options,
    bnvs,
    field_params,
):
    """
    Gets current density from bnvs and field_params according to options in options.
    Returned as a dict similar to fit_params/field_params etc.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    bnvs : list
        List of bnv results (each a 2D image).
    sig_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of field values across FOV.

    Returns
    -------
    source_params : dict
        Dictionary, key: param_keys, val: image (2D) of source field values across FOV.

    See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky,
        J.-P. Tetienne, and L. C. L. Hollenberg,
        Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field
        Measurements, Phys. Rev. Applied 14, 024076 (2020).
        https://doi.org/10.1103/PhysRevApplied.14.024076
        https://arxiv.org/abs/2005.06788
    """
    useful_opts = [
        options["fourier_pad_mode"],
        options["fourier_pad_factor"],
        options["system"].get_raw_pixel_size(options) * options["total_bin"],
        options["fourier_k_vector_epsilon"],
        options["fourier_do_hanning_filter"],
        options["fourier_low_cutoff"],
        options["fourier_high_cutoff"],
        options["standoff"],
        options["nv_layer_thickness"],
    ]

    if any([i in ["from_bxy", "from_bz"] for i in options["recon_methods"]]):

        # first check if Bx, By, Bz in fit_params
        # extract them
        if field_params is None:
            return None

        components = ["x", "y", "z"]

        for p in ["B" + comp for comp in components]:
            if p not in field_params:
                warnings.warn(
                    f"bfield param '{p} missing from field_params, skipping current calculation."
                )
                return None
            elif field_params[p] is None:
                return None

        bx, by, bz = [field_params["B" + comp] for comp in components]

    if any([i in ["from_bnv"] for i in options["recon_methods"]]):

        unvs = qdmpy.field.get_unvs(options)
        unv_opt = options["recon_unv_index"]
        if unv_opt is not None:
            unv = unvs[unv_opt]
            bnv = bnvs[unv_opt]
        else:
            unv = unvs[0]
            bnv = bnvs[0]

    source_params = {}
    for method in options["recon_methods"]:
        if method == "from_bxy":
            jx, jy = qdmpy.fourier.get_j_from_bxy([bx, by, bz], *useful_opts)
        elif method == "from_bz":
            jx, jy = qdmpy.fourier.get_j_from_bz([bx, by, bz], *useful_opts)
        elif method == "from_bnv":
            jx, jy = qdmpy.fourier.get_j_from_bnv(bnv, unv, *useful_opts)
        else:
            warnings.warn(f"recon_method option {method} not recognised.")
            return None

        jnorm = np.sqrt(jx ** 2 + jy ** 2)

        if options["source_bground_method"]:
            if "polygons" in options and (
                options["mask_polygons_bground"]
                or options["source_bground_method"] == "interpolate"
            ):
                polygons = options["polygons"]
            else:
                polygons = None
            jx_bground = qdmpy.itool.get_background(
                jx,
                options["source_bground_method"],
                polygons=polygons,
                **options["source_bground_params"],
            )
            jy_bground = qdmpy.itool.get_background(
                jy,
                options["source_bground_method"],
                polygons=polygons,
                **options["source_bground_params"],
            )
            jnorm_bground = qdmpy.itool.get_background(
                jnorm,
                options["source_bground_method"],
                polygons=polygons,
                **options["source_bground_params"],
            )
            source_params = {
                **source_params,
                **{
                    "Jx_" + method + "_full": jx,
                    "Jy_" + method + "_full": jy,
                    "Jnorm_" + method + "_full": jnorm,
                    "Jx_" + method + "_bground": jx_bground,
                    "Jy_" + method + "_bground": jy_bground,
                    "Jnorm_" + method + "_bground": jnorm_bground,
                    "Jx_" + method: jx - jx_bground,
                    "Jy_" + method: jy - jy_bground,
                    "Jnorm_" + method: jnorm - jnorm_bground,
                },
            }
        else:
            source_params = {
                **source_params,
                **{"Jx_" + method: jx, "Jy_" + method: jy, "Jnorm_" + method: jnorm},
            }

    return source_params


# ============================================================================


def get_magnetisation(
    options,
    bnvs,
    field_params,
):
    """
    Gets magnetisation from bnvs and field_params according to options in options.
    Returned as a dict similar to fit_params/field_params etc.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    bnvs : list
        List of bnv results (each a 2D image).
    sig_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of field values across FOV.

    Returns
    -------
    source_params : dict
        Dictionary, key: param_keys, val: image (2D) of source field values across FOV

    See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky,
        J.-P. Tetienne, and L. C. L. Hollenberg,
        Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field
        Measurements, Phys. Rev. Applied 14, 024076 (2020).
        https://doi.org/10.1103/PhysRevApplied.14.024076
        https://arxiv.org/abs/2005.06788
    """
    useful_opts = [
        options["magnetisation_angle"],
        options["fourier_pad_mode"],
        options["fourier_pad_factor"],
        options["system"].get_raw_pixel_size(options) * options["total_bin"],
        options["fourier_k_vector_epsilon"],
        options["fourier_do_hanning_filter"],
        options["fourier_low_cutoff"],
        options["fourier_high_cutoff"],
        options["standoff"],
        options["nv_layer_thickness"],
    ]

    if any([i in ["from_bxy", "from_bz"] for i in options["recon_methods"]]):

        # first check if Bx, By, Bz in fit_params
        # extract them
        if field_params is None:
            return None

        components = ["x", "y", "z"]

        for p in ["B" + comp for comp in components]:
            if p not in field_params:
                warnings.warn(
                    f"bfield param '{p} missing from field_params, skipping mag calculation."
                )
                return None
            elif field_params[p] is None:
                return None

        bx, by, bz = [field_params["B" + comp] for comp in components]

    if any([i in ["from_bnv"] for i in options["recon_methods"]]):

        unvs = qdmpy.field.get_unvs(options)
        unv_opt = options["recon_unv_index"]
        if unv_opt is not None:
            unv = unvs[unv_opt]
            bnv = bnvs[unv_opt]
        else:
            unv = unvs[0]
            bnv = bnvs[0]

    source_params = {}
    for method in options["recon_methods"]:
        if method == "from_bxy":
            m = qdmpy.fourier.get_m_from_bxy([bx, by, bz], *useful_opts)
        elif method == "from_bz":
            m = qdmpy.fourier.get_m_from_bz([bx, by, bz], *useful_opts)
        elif method == "from_bnv":
            m = qdmpy.fourier.get_m_from_bnv(bnv, unv, *useful_opts)
        else:
            warnings.warn("recon_method option not recognised.")
            return None

        if options["source_bground_method"]:
            if "polygons" in options and (
                options["mask_polygons_bground"]
                or options["source_bground_method"] == "interpolate"
            ):
                polygons = options["polygons"]
            else:
                polygons = None
            m_bground = qdmpy.itool.get_background(
                m,
                options["source_bground_method"],
                polygons=polygons,
                **options["source_bground_params"],
            )

        if not options["magnetisation_angle"]:
            if options["source_bground_method"]:
                source_params = {
                    **source_params,
                    **{
                        "Mz_" + method + "_full": m,
                        "Mz_" + method + "_bground": m_bground,
                        "Mz_" + method: m - m_bground,
                    },
                }
            else:
                source_params = {**source_params, **{"Mz_" + method: m}}
        else:
            if options["source_bground_method"]:
                source_params = {
                    **source_params,
                    **{
                        "Mpsi_" + method + "_full": m,
                        "Mpsi_" + method + "_bground": m_bground,
                        "Mpsi_" + method: m - m_bground,
                    },
                }
            else:
                source_params = {**source_params, **{"Mpsi_" + method: m}}
    return source_params

# -*- coding: utf-8 -*-
"""
Interface for source sub-module.

Functions
---------
 - `qdmpy.source.interface.odmr_source_retrieval`
 - `qdmpy.source.interface.get_current_density`
 - `qdmpy.source.interface.get_magnetization`
 - `qdmpy.source.interface.add_divperp_j`
 - `qdmpy.source.interface.in_plane_mag_normalise`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.source.interface.odmr_source_retrieval": True,
    "qdmpy.source.interface.get_current_density": True,
    "qdmpy.source.interface.get_magnetization": True,
    "qdmpy.source.interface.add_divperp_j": True,
    "qdmpy.source.interface.in_plane_mag_normalise": True,
}

# ============================================================================

import numpy as np

# ============================================================================

import qdmpy.shared.geom
import qdmpy.source.io
import qdmpy.source.current
import qdmpy.source.magnetization
from qdmpy.shared.misc import warn

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
    qdmpy.source.io.prep_output_directories(options)

    # do what was asked
    source_fns = {
        "current_density": get_current_density,
        "magnetization": get_magnetization,
    }
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
            source_params[key] -= np.nanmean(
                source_params[key][y_min:y_max, x_min:x_max]
            )
            source_params[key] -= np.nanmean(
                source_params[key][y_min:y_max, x_min:x_max]
            )

    options["source_params"] = list(source_params.keys())
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

    if any(
        [i in ["from_bxy", "from_bz", "without_ft"] for i in options["recon_methods"]]
    ):

        # first check if Bx, By, Bz in fit_params
        # extract them
        if field_params is None:
            return None

        components = ["x", "y", "z"]

        for p in ["B" + comp for comp in components]:
            if p not in field_params:
                warn(
                    f"bfield param '{p}' missing from field_params, skipping current"
                    " calculation."
                )
                return None
            elif field_params[p] is None:
                return None

        bx, by, bz = [field_params["B" + comp] for comp in components]

    if any([i in ["from_bnv"] for i in options["recon_methods"]]):
        unvs = qdmpy.shared.geom.get_unvs(options)
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
            jx, jy = qdmpy.source.current.get_j_from_bxy(
                [bx, by, bz],
                *useful_opts,
                nvs_above_sample=options["NVs_above_sample"],
            )
        elif method == "from_bz":
            jx, jy = qdmpy.source.current.get_j_from_bz([bx, by, bz], *useful_opts)
        elif method == "from_bnv":
            jx, jy = qdmpy.source.current.get_j_from_bnv(
                bnv,
                unv,
                *useful_opts,
                nvs_above_sample=options["NVs_above_sample"],
            )
        # elif method == "from_bxyz_w_src":
        #     jx, jy = qdmpy.source.current.get_j_from_bxyz_w_src(
        #         [bx, by, bz], *useful_opts, sigma=options["src_sigma"]
        # )
        elif method == "without_ft":
            jx, jy = qdmpy.source.current.get_j_without_ft([bx, by, bz])
        else:
            warn(f"recon_method '{method}' not recognised for j recon, skipping.")
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
            jx_bground, jx_mask = qdmpy.shared.itool.get_background(
                jx,
                options["source_bground_method"],
                polygons=polygons,
                **options["source_bground_params"],
            )
            jy_bground, jy_mask = qdmpy.shared.itool.get_background(
                jy,
                options["source_bground_method"],
                polygons=polygons,
                **options["source_bground_params"],
            )
            jnorm_bground, jnorm_mask = qdmpy.shared.itool.get_background(
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
                    "Jx_" + method + "_mask": jx_mask,
                    "Jy_" + method + "_mask": jy_mask,
                    "Jnorm_" + method + "_mask": jnorm_mask,
                },
            }
        else:
            source_params = {
                **source_params,
                **{"Jx_" + method: jx, "Jy_" + method: jy, "Jnorm_" + method: jnorm},
            }

    add_divperp_j(options, source_params)

    return source_params


# ============================================================================


def get_magnetization(
    options,
    bnvs,
    field_params,
):
    """
    Gets magnetization from bnvs and field_params according to options in options.
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
        options["magnetization_angle"],
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
                warn(
                    f"bfield param '{p}' missing from field_params, skipping mag"
                    " calculation."
                )
                return None
            elif field_params[p] is None:
                return None

        bx, by, bz = [field_params["B" + comp] for comp in components]

    if any([i in ["from_bnv"] for i in options["recon_methods"]]):

        unvs = qdmpy.shared.geom.get_unvs(options)
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
            m = qdmpy.source.magnetization.get_m_from_bxy([bx, by, bz], *useful_opts)
        elif method == "from_bz":
            m = qdmpy.source.magnetization.get_m_from_bz([bx, by, bz], *useful_opts)
        elif method == "from_bnv":
            m = qdmpy.source.magnetization.get_m_from_bnv(
                bnv,
                unv,
                *useful_opts,
                nvs_above_sample=options["NVs_above_sample"],
            )
        else:
            warn(
                f"recon_method '{method}' option not recognised for mag. recon,"
                " skipping."
            )
            return None

        if (
            options["magnetization_angle"] is not None
            and options["in_plane_mag_norm_number_pixels"]
        ):
            m = in_plane_mag_normalise(
                m,
                options["magnetization_angle"],
                options["in_plane_mag_norm_number_pixels"],
            )

        if options["source_bground_method"]:
            if "polygons" in options and (
                options["mask_polygons_bground"]
                or options["source_bground_method"] == "interpolate"
            ):
                polygons = options["polygons"]
            else:
                polygons = None
            m_bground, m_mask = qdmpy.shared.itool.get_background(
                m,
                options["source_bground_method"],
                polygons=polygons,
                **options["source_bground_params"],
            )

        if options["magnetization_angle"] is None:
            if options["source_bground_method"]:
                source_params = {
                    **source_params,
                    **{
                        "Mz_" + method + "_full": m,
                        "Mz_" + method + "_bground": m_bground,
                        "Mz_" + method: m - m_bground,
                        "Mz_" + method + "_mask": m_mask,
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
                        "Mpsi_" + method + "_mask": m_mask,
                    },
                }
            else:
                source_params = {**source_params, **{"Mpsi_" + method: m}}
    return source_params


# ============================================================================


def add_divperp_j(options, source_params):
    r"""jxy -> Divperp J

    Divperp = divergence in x and y only (perpendicular to surface normal)

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options (for the main/signal experiment).
    source_params : dict
        Dictionary, key: param_keys, val: image (2D) of source field values across FOV.

    Returns
    -------
    nothing (operates in place on field_params)


    $$ \nabla \times {\bf J} = \frac{\partial {\bf J} }{\partial x} + \frac{\partial {\bf J}}{\partial y} + \frac{\partial {\bf J}}{\partial z} $$

    $$ \nabla_{\perp} \times {\bf J} = \frac{\partial {\bf J} }{\partial x} + \frac{\partial {\bf J}}{\partial y} $$


    """

    if source_params is None:
        return None

    methods = options["recon_methods"]

    components = ["x", "y"]

    for method in methods:
        for p in ["J" + comp + "_" + method for comp in components]:
            if p not in source_params:
                warn(
                    f"source param '{p}' missing from source_params, skipping j recon."
                )
                return None
            elif source_params[p] is None:
                return None

    for method in methods:
        jx, jy = [source_params["J" + comp + "_" + method] for comp in components]
        conserv_j = qdmpy.source.current.get_divperp_j(
            [jx, jy],
            options["fourier_pad_mode"],
            options["fourier_pad_factor"],
            options["system"].get_raw_pixel_size(options) * options["total_bin"],
            options["fourier_k_vector_epsilon"],
        )

        source_params[f"divperp_J_{method}"] = conserv_j

    return None


# ============================================================================


def in_plane_mag_normalise(mag_image, psi, edge_pixels_used):
    """Normalise in-plane magnetization by taking average of mag near edge of image per line @ psi.
    The jist of this function was copied from D. Broadway's previous version of the code.

    Parameters
    ----------
    mag_image : np array
        2D magnetization array as directly calculated.
    psi : float
        Assumed in-plane magnetization angle (deg)
    edge_pixels_used : int
        Number of pixels to use at edge of image to calculate average to subtract.

    Returns
    -------
    mag_image : np array
        in-plane magnetization image with line artifacts substracted.
    """
    psi = np.deg2rad(psi)
    new_im = mag_image.copy()
    if psi < 0:
        mag_image = np.flip(mag_image, 1)
        new_im = np.flip(new_im, 1)
        psi = np.abs(psi)
        flip_back = True
    else:
        flip_back = False
    height, width = mag_image.shape
    max_y_idx = height - 1

    # first get indices of a line from origin at bottom left at psi (from +x to +y)
    origin_line_y = max_y_idx - np.tan(psi) * range(
        width
    )  # y = y_max - x*tan(psi) (y downwards)
    origin_line_y_ints = [
        round(y) for y in origin_line_y.tolist()
    ]  # y-idxs of line from bot. left @ psi

    offset = 0  # now look for parallel lines, at +- 'offset' in y
    for _ in range(height):
        above_y_idxs = []  # above origin line, i.e. lower y
        above_x_idxs = []  # corresponding x indices to above_y_idxs
        below_y_idxs = []
        below_x_idxs = []
        for x_idx, y_idx in enumerate(origin_line_y_ints):
            if y_idx - offset >= 0 and y_idx - offset <= max_y_idx:
                above_y_idxs.append(y_idx - offset)
                above_x_idxs.append(x_idx)

            if y_idx + offset >= 0 and y_idx + offset <= max_y_idx:
                below_y_idxs.append(y_idx + offset)
                below_x_idxs.append(x_idx)

        for coords in [(above_y_idxs, above_x_idxs), (below_y_idxs, below_x_idxs)]:
            im_cut = mag_image[coords]
            new_im[coords] = im_cut - (
                np.mean(
                    im_cut[0:edge_pixels_used] + np.mean(im_cut[-edge_pixels_used:])
                )
                / 2
            )

        offset += 1

    if flip_back:
        new_im = np.flip(new_im, 1)
    return new_im

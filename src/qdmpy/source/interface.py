# -*- coding: utf-8 -*-
"""
TODO.
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {"qdmpy.source.interface.": True}

# ============================================================================

import warnings
import numpy as np

# ============================================================================

import qdmpy.fourier
import qdmpy.field
import qdmpy.io

# ============================================================================


def odmr_source_retrieval(options, bnvs, field_params):
    if options["source_type"] is None:
        return None

    # generate output directories
    qdmpy.io.source.prep_output_directories(options)

    # do what was asked
    source_fns = {"current_density": get_current_density}
    source_params = source_fns[options["source_type"]](options, bnvs, field_params)

    # zero-point normalisation
    norm_region = options["current_zero_point_normalisation_region"]
    if norm_region is not None:
        x_min = norm_region[0][0]
        x_max = norm_region[1][0]
        y_min = norm_region[0][1]
        y_max = norm_region[1][1]
        source_params["Jx"] -= np.nanmean(source_params["Jx"][y_min:y_max, x_min:x_max])
        source_params["Jy"] -= np.nanmean(source_params["Jx"][y_min:y_max, x_min:x_max])
        source_params["Jnorm"] = np.sqrt(source_params["Jx"] ** 2 + source_params["Jy"] ** 2)

    options["source_params"] = [i for i in source_params.keys()]
    return source_params


# ============================================================================


def get_current_density(
    options,
    bnvs,
    field_params,
):
    """[summary]

    TODO docs!
    """

    if options["current_recon_method"] in ["from_bxy", "from_bz"]:
        # first check if Bx, By, Bz in fit_params
        # extract them
        if field_params is None:
            return None

        components = ["x", "y", "z"]

        for p in ["B" + comp for comp in components]:
            if p not in field_params:
                warnings.warn(
                    f"bfield param '{p} missing from field_params, skipping bfield plot."
                )
                return None
            elif field_params[p] is None:
                return None

        bx, by, bz = [field_params["B" + comp] for comp in components]
    else:
        unvs = qdmpy.field.get_unvs(options)
        unv_opt = options["current_recon_unv_index"]
        if unv_opt is not None:
            unv = unvs[unv_opt]
            bnv = bnvs[unv_opt]
        else:
            unv = unvs[0]
            bnv = bnvs[0]

    if options["current_recon_method"] == "from_bxy":
        jx, jy = qdmpy.fourier.get_j_from_bxy(
            [bx, by, bz],
            options["fourier_pad_mode"],
            options["fourier_pad_factor"],
            options["system"].get_raw_pixel_size(options) * options["total_bin"],
            options["fourier_k_vector_epsilon"],
            options["fourier_do_hanning_filter"],
            options["fourier_hanning_low_cutoff"],
            options["fourier_hanning_high_cutoff"],
            options["standoff"],
        )
    elif options["current_recon_method"] == "from_bz":
        jx, jy = qdmpy.fourier.get_j_from_bxy(
            [bx, by, bz],
            options["fourier_pad_mode"],
            options["fourier_pad_factor"],
            options["system"].get_raw_pixel_size(options) * options["total_bin"],
            options["fourier_k_vector_epsilon"],
            options["fourier_do_hanning_filter"],
            options["fourier_hanning_low_cutoff"],
            options["fourier_hanning_high_cutoff"],
            options["standoff"],
        )
    elif options["current_recon_method"] == "from_bnv":
        jx, jy = qdmpy.fourier.get_j_from_bxy(
            bnv,
            unv,
            options["fourier_pad_mode"],
            options["fourier_pad_factor"],
            options["system"].get_raw_pixel_size(options) * options["total_bin"],
            options["fourier_k_vector_epsilon"],
            options["fourier_do_hanning_filter"],
            options["fourier_hanning_low_cutoff"],
            options["fourier_hanning_high_cutoff"],
            options["standoff"],
        )
    else:
        warnings.warn("current_recon_method option not recognised.")
        return None

    source_params = {"Jx": jx, "Jy": jy, "Jnorm": np.sqrt(jx ** 2 + jy ** 2)}

    return source_params

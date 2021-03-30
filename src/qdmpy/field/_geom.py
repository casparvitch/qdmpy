# -*- coding: utf-8 -*-
"""
This module holds tools for determining the geometry of the NV-bias field
system etc., required for retrieving/reconstructing vector fields.

Functions
---------
 - `qdmpy.field._geom.get_unvs`
 - `qdmpy.field._geom.get_B_bias`
 - `qdmpy.field._geom.get_unv_frames`
 - `qdmpy.field._geom.add_bfield_reconstructed`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.field._geom.get_unvs": True,
    "qdmpy.field._geom.get_B_bias": True,
    "qdmpy.field._geom.get_unv_frames": True,
    "qdmpy.field._geom.add_bfield_reconstructed": True,
}

# ============================================================================

import numpy as np
import numpy.linalg as LA
from math import radians

# ============================================================================


def get_unvs(options):
    """
    Returns orientation (relative to lab frame) of NVs. Shape: (4,3) regardless of sample.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    unvs : np array
        Shape: (4,3). Equivalent to uNV_Z for each NV.
    """

    bias_x, bias_y, bias_z = get_B_bias(options)

    # nv_ori = np.zeros((4, 3)) # not required anywhere?
    # nv_signs = np.zeros(4)
    unvs = np.zeros((4, 3))  # z unit vectors of unv frame (in lab frame)

    # Get the NV orientations
    from qdmpy.constants import (
        NV_AXES_100_110,
        NV_AXES_100_100,
        NV_AXES_111,
    )  # avoid cyclic dependencies

    if options["use_unvs"]:
        unvs = np.array(options["unvs"])
        if unvs.shape != (4, 3):
            raise ValueError("Incorrect unvs format passed to Hamiltonian. Expected shape: (4,3).")
    else:
        if options["diamond_ori"] == "<100>_<100>":
            nv_axes = NV_AXES_100_100
        elif options["diamond_ori"] == "<100>_<110>":
            nv_axes = NV_AXES_100_110
        elif options["diamond_ori"] == "<111>":
            nv_axes = NV_AXES_111
        else:
            raise RuntimeError("diamond_ori not recognised.")

        for nv_num in range(len(nv_axes)):
            projection = np.dot(nv_axes[nv_num]["ori"], [bias_x, bias_y, bias_z])
            nv_axes[nv_num]["mag"] = np.abs(projection)
            nv_axes[nv_num]["sign"] = np.sign(projection)

        sorted_dict = sorted(nv_axes, key=lambda x: x["mag"], reverse=True)

        for idx in range(len(sorted_dict)):
            # nv_ori[idx, :] = sorted_dict[idx]["ori"] # not required anywhere?
            # nv_signs[idx] = sorted_dict[idx]["sign"]
            unvs[idx, :] = np.array(sorted_dict[idx]["ori"]) * sorted_dict[idx]["sign"]

    return unvs


# ============================================================================


def get_B_bias(options):
    """
    Returns (bx, by, bz) for the bias field (supplied in options dict) in Gauss

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    bxyz : tuple
        (bx, by, bz) for the bias field, in Gauss.
    """
    bias_field = None
    if options["auto_read_bias"]:
        bias_on, bias_field = options["system"].get_bias_field(options)
        if not bias_on:
            bias_field = None
    if bias_field is not None:
        Bmag, Btheta_rad, Bphi_rad = bias_field
    else:
        Bmag = options["bias_mag"]
        Btheta_rad = radians(options["bias_theta"])
        Bphi_rad = radians(options["bias_phi"])

    bx = Bmag * np.sin(Btheta_rad) * np.cos(Bphi_rad)
    by = Bmag * np.sin(Btheta_rad) * np.sin(Bphi_rad)
    bz = Bmag * np.cos(Btheta_rad)
    return bx, by, bz


# ============================================================================


def get_unv_frames(options):
    """
    Returns array representing each NV reference frame.
    I.e. each index is a 2D array: [uNV_X, uNV_Y, uNV_Z] representing the unit vectors
    for that NV reference frame, in the lab frame.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    unv_frames : np array
        [ [uNV1_X, uNV1_Y, uNV1_Z], [uNV2_X, uNV2_Y, uNV2_Z], ...]
    """
    nv_signed_ori = get_unvs(options)
    unv_frames = np.zeros((4, 3, 3))
    for i in range(4):
        # calculate uNV frame in the lab frame
        uNV_Z = nv_signed_ori[i]
        # We have full freedom to pick x/y as long as xyz are all orthogonal
        # we can ensure this by picking Y to be orthog. to both the NV axis
        # and another NV axis, then get X to be the cross between those two.
        uNV_Y = np.cross(uNV_Z, nv_signed_ori[-i - 1])
        uNV_Y = uNV_Y / LA.norm(uNV_Y)
        uNV_X = np.cross(uNV_Y, uNV_Z)
        unv_frames[i, ::] = [uNV_X, uNV_Y, uNV_Z]

    return unv_frames


# ============================================================================

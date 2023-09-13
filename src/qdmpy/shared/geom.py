# -*- coding: utf-8 -*-
"""
This module holds tools for determining the geometry of the NV-bias field
system etc., required for retrieving/reconstructing vector fields.


Functions
---------
 - `qdmpy.shared.geom.get_unvs`
 - `qdmpy.shared.geom.get_unv_frames`

Constants
---------
 - `qdmpy.shared.geom.NV_AXES_100_110`
 - `qdmpy.shared.geom.NV_AXES_100_100`
 - `qdmpy.shared.geom.NV_AXES_111`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.shared.geom.get_unvs": True,
    "qdmpy.shared.geom.get_unv_frames": True,
    "qdmpy.shared.geom.NV_AXES_100_110": True,
    "qdmpy.shared.geom.NV_AXES_100_100": True,
    "qdmpy.shared.geom.NV_AXES_111": True,
}

# ============================================================================

import numpy as np
import numpy.linalg as LA  # noqa: N812

# ============================================================================

# ============================================================================


# NOTE for other NV orientations, pass in unvs -> not possible to determine in full
#   generality the orientations for <111> etc.

# nv orientations (unit vectors) wrt lab frame [x, y, z]
NV_AXES_100_110 = [
    {"nv_number": 0, "ori": [np.sqrt(2 / 3), 0, np.sqrt(1 / 3)]},
    {"nv_number": 1, "ori": [-np.sqrt(2 / 3), 0, np.sqrt(1 / 3)]},
    {"nv_number": 2, "ori": [0, np.sqrt(2 / 3), -np.sqrt(1 / 3)]},
    {"nv_number": 3, "ori": [0, -np.sqrt(2 / 3), -np.sqrt(1 / 3)]},
]
"""
<100> top face oriented, <110> edge face oriented diamond (CVD).

NV orientations (unit vectors) relative to lab frame.

Assuming diamond is square to lab frame:

first 3 numbers: orientation of top face of diamond, e.g. <100>

second 3 numbers: orientation of edges of diamond, e.g. <110>

CVD Diamonds are usually <100>, <110>. HPHT usually <100>, <100>.

![](https://i.imgur.com/Rudnzyo.png)

Purple plane corresponds to top (or bottom) face of diamond, orange planes correspond to edge faces.
"""

NV_AXES_100_100 = [
    {"nv_number": 0, "ori": [np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]},
    {
        "nv_number": 1,
        "ori": [-np.sqrt(1 / 3), -np.sqrt(1 / 3), np.sqrt(1 / 3)],
    },
    {
        "nv_number": 2,
        "ori": [np.sqrt(1 / 3), -np.sqrt(1 / 3), -np.sqrt(1 / 3)],
    },
    {
        "nv_number": 3,
        "ori": [-np.sqrt(1 / 3), np.sqrt(1 / 3), -np.sqrt(1 / 3)],
    },
]
"""
<100> top face oriented, <100> edge face oriented diamond (HPHT).

NV orientations (unit vectors) relative to lab frame.

Assuming diamond is square to lab frame:

first 3 numbers: orientation of top face of diamond, e.g. <100>

second 3 numbers: orientation of edges of diamond, e.g. <110>

CVD Diamonds are usually <100>, <110>. HPHT usually <100>, <100>.

![](https://i.imgur.com/cpErjAH.png)

Purple plane: top face of diamond, orange plane: edge faces.
"""

NV_AXES_111 = [
    {"nv_number": 0, "ori": [0, 0, 1]},
    {"nv_number": 1, "ori": [np.nan, np.nan, np.nan]},
    {"nv_number": 2, "ori": [np.nan, np.nan, np.nan]},
    {"nv_number": 3, "ori": [np.nan, np.nan, np.nan]},
]
"""
<111> top face oriented.

NV orientations (unit vectors) relative to lab frame.

Only the first nv can be oriented in general. This constant
is defined as a convenience for single-bnv <111> measurements.

<111> diamonds have an NV family oriented in z, i.e. perpindicular
to the diamond surface.
"""


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
        Shape: (4,3). Equivalent to uNV_Z for each NV. (Sorted largest to smallest Bnv)
    """

    bias_x, bias_y, bias_z = options["bias_field_cartesian_gauss"]

    unvs = np.zeros((4, 3))  # z unit vectors of unv frame (in lab frame)

    if options["use_unvs"]:
        unvs = np.array(options["unvs"])
        if unvs.shape != (4, 3):
            raise ValueError(
                "Incorrect unvs format passed to Hamiltonian. Expected shape: (4,3)."
            )
        if options["auto_order_unvs"]:
            nv_axes = [
                {"nv_number": i, "ori": ori.copy()} for i, ori in enumerate(unvs)
            ]
            for nv_num in range(len(nv_axes)):
                projection = np.dot(nv_axes[nv_num]["ori"], [bias_x, bias_y, bias_z])
                nv_axes[nv_num]["mag"] = np.abs(projection)
                nv_axes[nv_num]["sign"] = np.sign(projection)
            sorted_dict = sorted(nv_axes, key=lambda x: x["mag"], reverse=True)

            for idx in range(len(sorted_dict)):
                unvs[idx, :] = (
                    np.array(sorted_dict[idx]["ori"]) * sorted_dict[idx]["sign"]
                )
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
            unvs[idx, :] = np.array(sorted_dict[idx]["ori"]) * sorted_dict[idx]["sign"]

    options["unvs_used"] = unvs

    return unvs


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
        unv_z = nv_signed_ori[i]
        # We have full freedom to pick x/y as long as xyz are all orthogonal
        # we can ensure this by picking Y to be orthog. to both the NV axis
        # and another NV axis, then get X to be the cross between those two.
        unv_y = np.cross(unv_z, nv_signed_ori[-i - 1])
        unv_y = unv_y / LA.norm(unv_y)
        unv_x = np.cross(unv_y, unv_z)
        unv_frames[i, ::] = [unv_x, unv_y, unv_z]

    return unv_frames


# ============================================================================

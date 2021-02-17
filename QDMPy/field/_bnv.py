# -*- coding: utf-8 -*-
"""
This module holds tools for calculating the bnv from
ODMR datasets (after they've been fit with the `QDMPy.fit.interface` tooling).

Functions
---------
 - `QDMPy.field._bnv.get_bnvs_and_dshifts`
 - `QDMPy.field._bnv.check_exp_bnv_compatibility`
 - `QDMPy.field._bnv.bnv_refsub`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.field._bnv.get_bnvs_and_dshifts": True,
    "QDMPy.field._bnv.check_exp_bnv_compatibility": True,
    "QDMPy.field._bnv.bnv_refsub": True,
}

# ============================================================================

import numpy as np

# ============================================================================

import QDMPy.constants

# ============================================================================


def get_bnvs_and_dshifts(pixel_fit_params):
    """
    pixel_fit_params -> bnvs, dshifts (both lists of np arrays, 2D)

    Arguments
    ---------
    fit_result_dict : OrderedDict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Ordered by the order of functions in options["fit_functions"].
        If None, returns ([], [])

    Returns
    -------
    bnvs : list
        List of np arrays (2D) giving B_NV for each NV family/orientation.
        If num_peaks is odd, the bnv is given as the shift of that peak,
        and the dshifts is left as np.nans.

    dshifts : list
        List of np arrays (2D) giving the D (~DFS) of each NV family/orientation.
        If num_peaks is odd, the bnv is given as the shift of that peak,
        and the dshifts is left as np.nans.
    """
    if pixel_fit_params is None:
        return [], []

    # find params for peak position (all must start with 'pos')
    peak_posns = []
    for param_name, param_map in pixel_fit_params.items():
        if param_name.startswith("pos"):
            peak_posns.append(param_map)

    # ensure peaks are in correct order by sorting their average position
    peak_posns.sort(key=np.nanmean)
    num_peaks = len(peak_posns)

    if num_peaks == 1:
        bnvs = [np.abs(peak_posns[0] / (2 * QDMPy.constants.GAMMA))]
        dshifts = np.empty(bnvs[0].shape)
        dshifts.fill(np.nan)
    elif num_peaks == 2:
        bnvs = [np.abs(peak_posns[1] - peak_posns[0]) / (2 * QDMPy.constants.GAMMA)]
        dshifts = [(peak_posns[1] + peak_posns[0])]
    else:
        bnvs = []
        dshifts = []
        for i in range(num_peaks // 2):
            bnvs.append(np.abs(peak_posns[-i - 1] - peak_posns[i]) / (2 * QDMPy.constants.GAMMA))
            dshifts.append(peak_posns[-i - 1] + peak_posns[i])
        if ((num_peaks // 2) * 2) + 1 == num_peaks:
            middle_bnv = np.abs(peak_posns[num_peaks // 2 + 1]) / (2 * QDMPy.constants.GAMMA)
            bnvs.append(middle_bnv)
            middle_dshift = np.empty(middle_bnv.shape)
            middle_dshift.fill(np.nan)
            dshifts.append(middle_dshift)
    return bnvs, dshifts  # not the most elegant data structure ?


# ============================================================================


def check_exp_bnv_compatibility(sig_bnvs, ref_bnvs):
    """
    Checks size (and keys) of fit results match between sig experiment and reference.

    Arguments
    ---------
    sig_bnvs : list
        List of np arrays (2D) giving B_NV for each NV family/orientation.
        If num_peaks is odd, the bnv is given as the shift of that peak,
        and the d_shift is left as np.nans.

    ref_bnvs : list
        Same as bnvs, but for reference measurement (or None if no reference used).

    """
    # no reference supplied, so of course they're compatible! sig_bnvs=[] if no pixel fitting run
    if not sig_bnvs or not ref_bnvs:
        return

    if len(sig_bnvs) != len(ref_bnvs):
        raise RuntimeError(
            "Number of bnvs/dshifts in sig experiment different from number in reference."
        )
    if sig_bnvs[0].shape != ref_bnvs[0].shape:
        raise RuntimeError("Different image shape in main experiment to reference.")


# ============================================================================


def bnv_refsub(options, sig_bnvs, ref_bnvs):
    """docstring here"""
    # TODO
    # documentation +
    # "bnv_bsub_method" option -> allow other bsubs e.g.
    # subtract_blurred, subtract_outside_polygons etc. should be defined on any 2D array,
    # do that later
    if ref_bnvs:
        check_exp_bnv_compatibility(sig_bnvs, ref_bnvs)
        return [sig - ref for sig, ref in zip(sig_bnvs, ref_bnvs)]
    else:
        return sig_bnvs

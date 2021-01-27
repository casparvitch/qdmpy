# -*- coding: utf-8 -*-
"""
This module holds tools for calculating the magnetic field (\\vec{B}) from
ODMR datasets (after they've been fit with the `QDMPy.fit.interface` tooling).

Functions
---------
"""

# ============================================================================

__author__ = "Sam Scholten"

# ============================================================================

import numpy as np

# ============================================================================


# TODO
#  - going to need to have some BNV reference capability...
#       - pass directory of previous fit result
#       - allow to be calculated in same notebook? Give options a different name...
#  - plot BNVs -> do now(ish) so we can test
#  - linecuts of param fits vs pixel (as in matlab) -> plot.fits
#       - fit (line) vs data (pts)
#       - initial guess (line) vs data (pts)
#       - similarly: linecuts Bxyz and D vs pixels
# - plotting of B_NVs:
#       - plane subtraction
#       - mean subtraction (offer other types?)
#       - reference subtraction
#       - offer a bunch of these things?
# - so yes still need some reference capability...
#   - create new output dirs for different subtractions...


def get_bnvs_and_dshifts(options, pixel_fit_params):
    """
    TODO

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    fit_result_dict : OrderedDict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Ordered by the order of functions in options["fit_functions"].

    Returns
    -------
    bnvs : list
        List of np arrays (2D) giving B_NV for each NV family/orientation.
        If num_peaks is odd, the bnv is given as the shift of that peak,
        and the d_shift is left as np.nans.

    d_shifts : list
        List of np arrays (2D) giving the D (~DFS) of each NV family/orientation.
        If num_peaks is odd, the bnv is given as the shift of that peak,
        and the d_shift is left as np.nans.
    """

    # find params for peak position (all must start with 'pos')
    peak_posns = []
    for param_name, param_map in pixel_fit_params.items():
        if param_name.startswith("pos"):
            peak_posns.append(param_map)

    # ensure peaks are in correct order by sorting their average position
    peak_posns.sort(key=np.nanmean)
    num_peaks = len(peak_posns)

    gamma = 2.8  # MHz/G

    if num_peaks == 1:
        bnvs = [np.abs(peak_posns[0] / (2 * gamma))]
        d_shifts = np.empty(bnvs[0].shape)
        d_shifts.fill(np.nan)
    elif num_peaks == 2:
        bnvs = [np.abs(peak_posns[1] - peak_posns[0]) / (2 * gamma)]
        d_shifts = [(peak_posns[1] + peak_posns[0]) / (2 * gamma)]
    else:
        bnvs = []
        d_shifts = []
        for i in range(num_peaks // 2):
            bnvs.append(np.abs(peak_posns[-i - 1] - peak_posns[i]) / (2 * gamma))
            d_shifts.append((peak_posns[-i - 1] + peak_posns[i]) / (2 * gamma))
        if ((num_peaks // 2) * 2) + 1 == num_peaks:
            middle_bnv = np.abs(peak_posns[num_peaks // 2 + 1]) / (2 * gamma)
            bnvs.append(middle_bnv)
            middle_d_shift = np.empty(middle_bnv.shape)
            middle_d_shift.fill(np.nan)
            d_shifts.append(middle_d_shift)
    return bnvs, d_shifts  # Note not the most elegant data structure

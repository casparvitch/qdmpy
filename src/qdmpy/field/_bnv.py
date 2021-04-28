# -*- coding: utf-8 -*-
"""
This module holds tools for calculating the bnv from
ODMR datasets (after they've been fit with the `qdmpy.fit.interface` tooling).

Functions
---------
 - `qdmpy.field._bnv.get_bnvs_and_dshifts`
 - `qdmpy.field._bnv.get_bnv_sd`
 - `qdmpy.field._bnv.check_exp_bnv_compatibility`
 - `qdmpy.field._bnv.bnv_refsub`
 - `qdmpy.field._bnv.sub_bground_bnvs`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.field._bnv.get_bnvs_and_dshifts": True,
    "qdmpy.field._bnv.get_bnv_sd": True,
    "qdmpy.field._bnv.check_exp_bnv_compatibility": True,
    "qdmpy.field._bnv.bnv_refsub": True,
    "qdmpy.field._bnv.sub_bground_bnvs": True,
}

# ============================================================================

import numpy as np

# ============================================================================

import qdmpy.itool as Qitool

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

    from qdmpy.constants import GAMMA

    if num_peaks == 1:
        bnvs = [np.abs(peak_posns[0] / (2 * GAMMA))]
        dshifts = [np.empty(bnvs[0].shape)]
        dshifts[0].fill(np.nan)
    elif num_peaks == 2:
        bnvs = [np.abs(peak_posns[1] - peak_posns[0]) / (2 * GAMMA)]
        dshifts = [(peak_posns[1] + peak_posns[0]) / 2]
    else:
        bnvs = []
        dshifts = []
        for i in range(num_peaks // 2):
            bnvs.append(np.abs(peak_posns[-i - 1] - peak_posns[i]) / (2 * GAMMA))
            dshifts.append((peak_posns[-i - 1] + peak_posns[i]) / 2)
        if ((num_peaks // 2) * 2) + 1 == num_peaks:
            middle_bnv = np.abs(peak_posns[num_peaks // 2 + 1]) / (2 * GAMMA)
            bnvs.append(middle_bnv)
            middle_dshift = np.empty(middle_bnv.shape)
            middle_dshift.fill(np.nan)
            dshifts.append(middle_dshift)
    return bnvs, dshifts


# ============================================================================


def get_bnv_sd(sigmas):
    """ get standard deviation of bnvs given SD of peaks. """
    if sigmas is None:
        return None
    # find params for peak position (all must start with 'pos')
    peak_sd = []
    for param_name, sigma_map in sigmas.items():
        if param_name.startswith("pos"):
            peak_sd.append([int(param_name[-1]), sigma_map])  # [peak num, map]

    # ensure peaks are in correct order by sorting their keys
    peak_sd.sort(key=lambda x: x[0])
    peak_sd = [x[1] for x in peak_sd]
    num_peaks = len(peak_sd)

    from qdmpy.constants import GAMMA

    if num_peaks == 1:
        return peak_sd / (2 * GAMMA)
    elif num_peaks == 2:
        return (peak_sd[0] + peak_sd[1]) / (2 * GAMMA)
    else:
        sd = []
        for i in range(num_peaks // 2):
            sd.append((peak_sd[-i - 1] + peak_sd[i]) / (2 * GAMMA))
        if ((num_peaks // 2) * 2) + 1 == num_peaks:
            sd.append(peak_sd[num_peaks // 2 + 1] / (2 * GAMMA))
        return sd


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
    """Calculate sig - ref bnv list.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options (for the main/signal experiment).
    sig_bnvs : list
        List of np arrays (2D) giving B_NV for each NV family/orientation in sig experiment.
        If num_peaks is odd, the bnv is given as the shift of that peak,
        and the dshifts is left as np.nans.
    ref_bnvs : dict
        Same as sig_bnvs but for ref experiment.

    Returns
    -------
    sig_sub_ref_bnvs : list
        sig - ref images
    """
    if ref_bnvs:
        check_exp_bnv_compatibility(sig_bnvs, ref_bnvs)
        return [sig - ref for sig, ref in zip(sig_bnvs, ref_bnvs)]
    else:
        return sig_bnvs.copy()


# ============================================================================


def sub_bground_bnvs(options, bnvs, method, **method_settings):
    """Subtract a background from the bnvs.

    Methods available:
        - "fix_zero"
            - Fix background to be a constant offset (z value)
            - params required in method_params_dict:
                "zero" an int/float, defining the constant offset of the background
        - "three_point"
            - Calculate plane background with linear algebra from three [x,y] lateral positions
              given
            - params required in method_params_dict:
                - "points" a len-3 iterable containing [x, y] points
        - "mean"
            - background calculated from mean of image
            - no params required
        - "poly"
            - background calculated from polynomial fit to image.
            - params required in method_params_dict:
                - "order": an int, the 'order' polynomial to fit. (e.g. 1 = plane).
        - "gaussian"
            - background calculated from gaussian fit to image.
            - no params required
        - "interpolate"
            - Background defined by the dataset smoothed via a sigma-gaussian filtering,
                and method-interpolation over masked (polygon) regions.
            - params required in method_params_dict:
                - "interp_method": nearest, linear, cubic.
                - "sigma": sigma passed to gaussian filter (see scipy.ndimage.gaussian_filter)
                    which is utilized on the background before interpolating
        - "gaussian_filter"
            - background calculated from image filtered with a gaussian filter.
            - params required in method_params_dict:
                - "sigma": sigma passed to gaussian filter (see scipy.ndimage.gaussian_filter)

    polygon utilization:
        - if method is not interpolate, the image is masked where the polygons are
          and the background is calculated without these regions
        - if the method is interpolate, these regions are interpolated over (and the rest
          of the image, gaussian smoothed, is 'background').


    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options (for the main/signal experiment).
    bnvs : list
        List of bnvs images (2D ndarrays)
    method : str
        Method to use for background subtraction. See above for details.
    **method_settings : dict
        (i.e. keyword arguments).
        Parameters passed to background subtraction algorithm. See above for details

    Returns
    -------
    output_bnvs
        bnvs with background subtracted
    """
    if options["mask_polygons_bground"] and "polygons" in options:
        polygons = options["polygons"]
    else:
        polygons = None
    output_bnvs = []
    for bnv in bnvs:
        bground = Qitool.get_background(bnv, method, polygons=polygons, **method_settings)
        output_bnvs.append(bnv - bground)

    return output_bnvs

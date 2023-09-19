# -*- coding: utf-8 -*-
"""
This module holds tools for calculating the bnv from
ODMR datasets (after they've been fit with the `qdmpy.pl.interface` tooling).

Functions
---------
 - `qdmpy.field.bnv.get_bnvs_and_dshifts`
 - `qdmpy.field.bnv.get_bnv_sd`
 - `qdmpy.field.bnv.check_exp_bnv_compatibility`
 - `qdmpy.field.bnv.bnv_refsub`
 - `qdmpy.field.bnv.sub_bground_bnvs`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.field.bnv.get_bnvs_and_dshifts": True,
    "qdmpy.field.bnv.get_bnv_sd": True,
    "qdmpy.field.bnv.check_exp_bnv_compatibility": True,
    "qdmpy.field.bnv.bnv_refsub": True,
    "qdmpy.field.bnv.sub_bground_bnvs": True,
}

# ============================================================================

import numpy as np
from pyfftw.interfaces import numpy_fft
from copy import copy

# ============================================================================

import qdmpy.shared.fourier
import qdmpy.shared.itool

# ============================================================================


GSLAC = 1024
"""
Ground state level anticrossing (in Gauss).
Used to determine if a single-peak ODMR resonance is shifting to larger or smaller field.
(if bias field magnitude is larger than the GSLAC value bnv is reversed).
Currently NOT used for ODMR with >1 resonance.
"""


GAMMA = 2.80  # MHz/G
r"""
The Bohr magneton times the Landé g-factor. See [Doherty2013](https://doi.org/10.1016/j.physrep.2013.02.001)
for details of the g-factor anisotropy.

|                                                                  |                                                               |
|------------------------------------------------------------------|---------------------------------------------------------------|
| \( \gamma_{\rm NV} = \mu_{\rm B} g_e  \)                         |                                                               |
| \( \mu_B = 1.39962449361 \times 10^{10}\ {\rm Hz} \rm{T}^{-1} \) |  [NIST](https://physics.nist.gov/cgi-bin/cuu/Value?mubshhz)   |
| \( \mu_B = 1.399...\ {\rm MHz/G} \)                              |                                                               |
| \( g_e \approx 2.0023 \)                                         |  [Doherty2013](https://doi.org/10.1016/j.physrep.2013.02.001) |
| \( \Rightarrow  \gamma_{\rm NV} \approx 2.80 {\rm MHz/G} \)      |                                                               |

"""

# ============================================================================


def get_bnvs_and_dshifts(pixel_fit_params, bias_field_spherical_deg, chosen_freqs):
    """
        pixel_fit_params -> bnvs, dshifts (both lists of np arrays, 2D)

        Arguments
        ---------
        fit_result_dict : OrderedDict
            Dictionary, key: param_keys, val: image (2D) of param values across FOV.
            Ordered by the order of functions in options["fit_functions"].
            If None, returns ([], [])
        bias_field_spherical_deg : tuple
            Bias field in spherical polar degrees (and gauss).
        freqs_to_use : array-like, length 8, each evaluating as True/False
            Which resonant frequencies are being used?
    `
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

    bias_mag = np.abs(bias_field_spherical_deg[0])

    # ensure peaks are in correct order by sorting their average position
    peak_posns.sort(key=np.nanmean)
    num_peaks = len(peak_posns)

    if num_peaks == 1:
        if bias_mag > GSLAC and chosen_freqs[0]:
            bnvs = [peak_posns[0] / GAMMA + 1024.0]
        else:
            if np.mean(peak_posns[0]) < 2870:
                bnvs = [(2870 - peak_posns[0]) / GAMMA]
            else:
                bnvs = [(peak_posns[0] - 2870) / GAMMA]

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
            peak = peak_posns[num_peaks // 2 + 1]
            sign = (
                -1 if np.mean(peak) < 2870 else +1
            )  # det. if L/R resonance (rel to bias)
            middle_bnv = sign * peak / GAMMA
            bnvs.append(middle_bnv)
            middle_dshift = np.empty(middle_bnv.shape)
            middle_dshift.fill(np.nan)
            dshifts.append(middle_dshift)
    return bnvs, dshifts


# ============================================================================


def get_bnv_sd(sigmas):
    """get standard deviation of bnvs given SD of peaks."""
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
            "Number of bnvs/dshifts in sig experiment different from number in"
            " reference."
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
        Parameters passed to background subtraction algorithm. See above for details.

    Returns
    -------
    output_bnvs
        bnvs with background subtracted
    """
    if "polygons" in options and (
        options["mask_polygons_bground"] or method == "interpolate"
    ):
        polygons = options["polygons"]
    else:
        polygons = None
    output_bnvs = []
    for bnv in bnvs:
        bground, _ = qdmpy.shared.itool.get_background(
            bnv, method, polygons=polygons, **method_settings
        )
        output_bnvs.append(bnv - bground)

    return output_bnvs


# ============================================================================


def prop_single_bnv(
    single_bnv,
    unv,
    pad_mode,
    pad_factor,
    pixel_size,
    k_vector_epsilon,
    nvs_above_sample,
):
    r"""
    Propagate single bnv to full vector magnetic field.

    Arguments
    ---------
    single_bnv : np array
        Single bnv map (np 2D array).
    unv : array-like, 1D
        Shape: 3, the uNV_Z corresponding to the above bnv map.
    pad_mode : str
        Mode to use for fourier padding. See np.pad for options.
    pad_factor : int
        Factor to pad image on all sides. E.g. pad_factor = 2 => 2 * image width on left and right
        and 2 * image height above and below.
    pixel_size : float
        Size of pixel in bnv, the rebinned pixel size.
        E.g. options["system"].get_raw_pixel_size(options) * options["total_bin"].
    k_vector_epsilon : float
        Add an epsilon value to the k-vectors to avoid some issues with 1/0.
    nvs_above_sample : bool
        True if NVs exist at higher z (in lab frame) than sample.

    Returns
    -------
    bx_reg, by_reg, bz_reg : np arrays (2D)

    \hat{\bf B} = {\bf v} \hat{B}_z({\bf k})

    (hat denotes 2D fourier transform, \vec{k} is 2D wavevector) and where

    {\bf v} = (-ik_x / k, -ik_y / k, 1)

    \hat{B}_z({\bf k}) = \frac{\hat{B}_{\rm NV}}{{\bf u}_{\rm NV} \cdot {\bf v}}

    See 'Box 1' in F. Casola, T. van der Sar, and A. Yacoby,
        Probing Condensed Matter Physics with Magnetometry Based on Nitrogen-Vacancy Centres in
        Diamond, Nature Reviews Materials 3, 17088 (2018).
        http://dx.doi.org/10.1038/natrevmats.2017.88
        https://arxiv.org/abs/1804.08742
    """

    bnv = copy(single_bnv)

    # first pad bnv
    padded_bnv, padder = qdmpy.shared.fourier.pad_image(bnv, pad_mode, pad_factor)

    fft_bnv = numpy_fft.fftshift(numpy_fft.fft2(padded_bnv))
    ky, kx, k = qdmpy.shared.fourier.define_k_vectors(
        fft_bnv.shape, pixel_size, k_vector_epsilon
    )

    unv_cpy = copy(unv) if nvs_above_sample else [-unv[0], -unv[1], unv[2]]

    # define transformation matrices -> e.g. see Casola 2018 given above
    u = [-1j * kx / k, -1j * ky / k, 1]
    unv_dot_u = unv_cpy[0] * u[0] + unv_cpy[1] * u[1] + unv_cpy[2] * u[2]

    bnv2bx = u[0] / unv_dot_u
    bnv2by = u[1] / unv_dot_u
    bnv2bz = u[2] / unv_dot_u
    # Expanded algebra below:
    # bnv2bx = 1 / (unv_cpy[0] + unv_cpy[1] * ky / kx + 1j * unv_cpy[2] * k / kx)
    # bnv2by = 1 / (unv_cpy[0] * kx / ky + unv_cpy[1] + 1j * unv_cpy[2] * k / ky)
    # bnv2bz = 1 / (-1j * unv_cpy[0] * kx / k - 1j * unv_cpy[1] * ky / k + unv_cpy[2])

    bnv2bx = qdmpy.shared.fourier.set_naninf_to_zero(bnv2bx)
    bnv2by = qdmpy.shared.fourier.set_naninf_to_zero(bnv2by)
    bnv2bz = qdmpy.shared.fourier.set_naninf_to_zero(bnv2bz)

    # transform to xyz
    fft_bx = fft_bnv * bnv2bx
    fft_by = fft_bnv * bnv2by
    fft_bz = fft_bnv * bnv2bz

    # fourier transform back into real space
    bx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_bx)).real
    by = numpy_fft.ifft2(numpy_fft.ifftshift(fft_by)).real
    bz = numpy_fft.ifft2(numpy_fft.ifftshift(fft_bz)).real

    # only return non-padded region
    bx_reg = qdmpy.shared.fourier.unpad_image(bx, padder)
    by_reg = qdmpy.shared.fourier.unpad_image(by, padder)
    bz_reg = qdmpy.shared.fourier.unpad_image(bz, padder)

    return bx_reg, by_reg, bz_reg


# ============================================================================

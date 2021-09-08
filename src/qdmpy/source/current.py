# -*- coding: utf-8 -*-
"""
Implement inversion of magnetic field to current density.

Functions
---------
 - `qdmpy.source.current.get_divperp_j`
 - `qdmpy.source.current.get_j_from_bxy`
 - `qdmpy.source.current.get_j_from_bz`
 - `qdmpy.source.current.get_j_from_bnv`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.source.current.get_divperp_j": True,
    "qdmpy.source.current.get_j_from_bxy": True,
    "qdmpy.source.current.get_j_from_bz": True,
    "qdmpy.source.current.get_j_from_bnv": True,
}

# ============================================================================

from pyfftw.interfaces import numpy_fft
from copy import copy
import numpy as np

# ============================================================================

import qdmpy.shared.fourier
from qdmpy.shared.fourier import define_current_transform

# ============================================================================


def get_divperp_j(
    jvec,
    pad_mode,
    pad_factor,
    pixel_size,
    k_vector_epsilon,
):
    r"""
    jxy calculated -> perpindicular (in-plane) divergence of j

    Arguments
    ---------
    jvec : list
        List of magnetic field components, e.g [jx_image, jy_image]
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


    Returns
    -------
    jx_recon, jy_recon : np arrays (2D)
        The reconstructed j (source) field maps.

    $$ \nabla \times {\bf J} = \frac{\partial {\bf J} }{\partial x} + \frac{\partial {\bf J}}{\partial y} + \frac{\partial {\bf J}}{\partial z} $$

    $$ \nabla_{\perp} \times {\bf J} = \frac{\partial {\bf J} }{\partial x} + \frac{\partial {\bf J}}{\partial y} $$

    """

    jx, jy = copy(jvec)
    # first pad each comp
    jx_pad, padder = qdmpy.shared.fourier.pad_image(jx, pad_mode, pad_factor)
    jy_pad, _ = qdmpy.shared.fourier.pad_image(jy, pad_mode, pad_factor)

    fft_jx = numpy_fft.fftshift(numpy_fft.fft2(jx_pad))
    fft_jy = numpy_fft.fftshift(numpy_fft.fft2(jy_pad))
    fft_jx = qdmpy.shared.fourier.set_naninf_to_zero(fft_jx)
    fft_jy = qdmpy.shared.fourier.set_naninf_to_zero(fft_jy)

    ky, kx, k = qdmpy.shared.fourier.define_k_vectors(
        fft_jx.shape, pixel_size, k_vector_epsilon
    )

    fft_divperp_j = -1j * kx * fft_jx + -1j * ky + fft_jy

    divperp_j = numpy_fft.ifft2(numpy_fft.ifftshift(fft_divperp_j)).real

    # only return non-padded region
    divperp_j_reg = qdmpy.shared.fourier.unpad_image(divperp_j, padder)
    mx = max(np.abs([np.nanmax(divperp_j_reg), np.nanmin(divperp_j_reg)]))
    return divperp_j_reg / mx


# ============================================================================


def get_j_from_bxy(
    bfield,
    pad_mode,
    pad_factor,
    pixel_size,
    k_vector_epsilon,
    do_hanning_filter,
    hanning_low_cutoff,
    hanning_high_cutoff,
    standoff,
    nv_layer_thickness,
    nvs_above_sample,
):

    r"""Bxy measured -> Jxy via fourier methods.

    Arguments
    ---------
    bfield : list
        List of magnetic field components, e.g [bx_image, by_image, bz_image]
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
    do_hanning_filter : bool
        Do a hanning filter?
    hanning_high_cutoff : float
        Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set
        via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff.
    hanning_low_cutoff : float
        Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set
        via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff.
    standoff : float
        Distance NV layer <-> Sample (in metres)
    nv_layer_thickness : float
        Thickness of NV layer (in metres)
    nvs_above_sample : bool
        True if NVs exist at higher z (in lab frame) than sample.

    Returns
    -------
    jx, jy : np arrays (2D)
        The calculated current density images, in A/m.

    See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky,
        J.-P. Tetienne, and L. C. L. Hollenberg,
        Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field
        Measurements, Phys. Rev. Applied 14, 024076 (2020).
        https://doi.org/10.1103/PhysRevApplied.14.024076
        https://arxiv.org/abs/2005.06788
    """
    # copy and convert Gauss -> Tesla
    bx = copy(bfield[0]) * 1e-4
    by = copy(bfield[1]) * 1e-4
    # first pad each comp
    bx_pad, padder = qdmpy.shared.fourier.pad_image(bx, pad_mode, pad_factor)
    by_pad, _ = qdmpy.shared.fourier.pad_image(by, pad_mode, pad_factor)

    fft_bx = numpy_fft.fftshift(numpy_fft.fft2(bx_pad))
    fft_by = numpy_fft.fftshift(numpy_fft.fft2(by_pad))
    fft_bx = qdmpy.shared.fourier.set_naninf_to_zero(fft_bx)
    fft_by = qdmpy.shared.fourier.set_naninf_to_zero(fft_by)

    ky, kx, k = qdmpy.shared.fourier.define_k_vectors(
        fft_bx.shape, pixel_size, k_vector_epsilon
    )

    sign = 1 if nvs_above_sample else -1

    # define transform
    _, bx_to_jy = define_current_transform(
        [sign, 0, 0], ky, kx, k, standoff, nv_layer_thickness
    )
    by_to_jx, _ = define_current_transform(
        [0, sign, 0], ky, kx, k, standoff, nv_layer_thickness
    )

    hanning_filt = qdmpy.shared.fourier.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
    )

    bx_to_jy = qdmpy.shared.fourier.set_naninf_to_zero(hanning_filt * bx_to_jy)
    by_to_jx = qdmpy.shared.fourier.set_naninf_to_zero(hanning_filt * by_to_jx)

    fft_jx = fft_by * by_to_jx
    fft_jy = fft_bx * bx_to_jy

    # fourier transform back into real space
    jx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jx)).real
    jy = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jy)).real

    # only return non-padded region
    jx_reg = qdmpy.shared.fourier.unpad_image(jx, padder)
    jy_reg = qdmpy.shared.fourier.unpad_image(jy, padder)
    return jx_reg, jy_reg


# ============================================================================


def get_j_from_bz(
    bfield,
    pad_mode,
    pad_factor,
    pixel_size,
    k_vector_epsilon,
    do_hanning_filter,
    hanning_low_cutoff,
    hanning_high_cutoff,
    standoff,
    nv_layer_thickness,
):
    r"""Bz measured -> Jxy via fourier methods.

    Arguments
    ---------
    bfield : list
        List of magnetic field components, e.g [bx_image, by_image, bz_image]
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
    do_hanning_filter : bool
        Do a hanning filter?
    hanning_high_cutoff : float
        Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set
        via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff.
    hanning_low_cutoff : float
        Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set
        via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff.
    standoff : float
        Distance NV layer <-> Sample (in metres)
    nv_layer_thickness : float
        Thickness of NV layer (in metres)

    Returns
    -------
    jx, jy : np arrays (2D)
        The calculated current density images, in A/m.

    See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky,
        J.-P. Tetienne, and L. C. L. Hollenberg,
        Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field
        Measurements, Phys. Rev. Applied 14, 024076 (2020).
        https://doi.org/10.1103/PhysRevApplied.14.024076
        https://arxiv.org/abs/2005.06788
    """
    bz = copy(bfield[2]) * 1e-4  # copy and convert Gauss to Tesla
    bz_pad, padder = qdmpy.shared.fourier.pad_image(bz, pad_mode, pad_factor)

    fft_bz = numpy_fft.fftshift(numpy_fft.fft2(bz_pad))
    fft_bz = qdmpy.shared.fourier.set_naninf_to_zero(fft_bz)

    ky, kx, k = qdmpy.shared.fourier.define_k_vectors(
        fft_bz.shape, pixel_size, k_vector_epsilon
    )

    # define transformation
    bz_to_jx, bz_to_jy = define_current_transform(
        [0, 0, 1], ky, kx, k, standoff, nv_layer_thickness
    )

    hanning_filt = qdmpy.shared.fourier.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
    )

    bz_to_jx = qdmpy.shared.fourier.set_naninf_to_zero(hanning_filt * bz_to_jx)
    bz_to_jy = qdmpy.shared.fourier.set_naninf_to_zero(hanning_filt * bz_to_jy)

    fft_jx = bz_to_jx * fft_bz
    fft_jy = bz_to_jy * fft_bz

    # fourier transform back into real space
    jx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jx)).real
    jy = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jy)).real

    # only return non-padded region
    jx_reg = qdmpy.shared.fourier.unpad_image(jx, padder)
    jy_reg = qdmpy.shared.fourier.unpad_image(jy, padder)
    return jx_reg, jy_reg


# ============================================================================


def get_j_without_ft(bfield):
    r"""Bxy measured -> (pseudo-)Jxy *without* fourier methods.

    Arguments
    ---------
    bfield : list
        List of magnetic field components, e.g [bx_image, by_image, bz_image]

    Returns
    -------
    jx, jy : np arrays (2D)
        The calculated current density images, in A/m.

    Get Jx, Jy approximation, without any fourier propogation. Simply rescale
    to currect units (Teslas *2 / mu_0)
    """
    scale = (1e-4 * 2) / (1.25663706212 * 1e-6)
    return -bfield[1] * scale, bfield[0] * scale


# didn't really work: if include add 'src_sigma' option
# def get_j_from_bxyz_w_src(
#     bfield,
#     pad_mode,
#     pad_factor,
#     pixel_size,
#     k_vector_epsilon,
#     do_hanning_filter,
#     hanning_low_cutoff,
#     hanning_high_cutoff,
#     standoff,
#     nv_layer_thickness,
#     sigma,
# ):
#     r"""Bz measured -> Jxy via fourier methods.

#     Arguments
#     ---------
#     bfield : list
#         List of magnetic field components, e.g [bx_image, by_image, bz_image]
#     pad_mode : str
#         Mode to use for fourier padding. See np.pad for options.
#     pad_factor : int
#         Factor to pad image on all sides. E.g. pad_factor = 2 => 2 * image width on left and right
#         and 2 * image height above and below.
#     pixel_size : float
#         Size of pixel in bnv, the rebinned pixel size.
#         E.g. options["system"].get_raw_pixel_size(options) * options["total_bin"].
#     k_vector_epsilon : float
#         Add an epsilon value to the k-vectors to avoid some issues with 1/0.
#     do_hanning_filter : bool
#         Do a hanning filter?
#     hanning_high_cutoff : float
#         Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set
#         via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff.
#     hanning_low_cutoff : float
#         Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set
#         via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff.
#     standoff : float
#         Distance NV layer <-> Sample (in metres)
#     nv_layer_thickness : float
#         Thickness of NV layer (in metres)

#     Returns
#     -------
#     jx, jy : np arrays (2D)
#         The calculated current density images, in A/m.

#     <explain>

#     See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky,
#         J.-P. Tetienne, and L. C. L. Hollenberg,
#         Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field
#         Measurements, Phys. Rev. Applied 14, 024076 (2020).
#         https://doi.org/10.1103/PhysRevApplied.14.024076
#         https://arxiv.org/abs/2005.06788
#     """
#     bx = copy(bfield[0]) * 1e-4  # copy and convert Gauss to Tesla
#     by = copy(bfield[1]) * 1e-4  # copy and convert Gauss to Tesla
#     bz = copy(bfield[2]) * 1e-4  # copy and convert Gauss to Tesla
#     bx_pad, padder = qdmpy.shared.fourier.pad_image(bx, pad_mode, pad_factor)
#     by_pad, padder = qdmpy.shared.fourier.pad_image(by, pad_mode, pad_factor)
#     bz_pad, padder = qdmpy.shared.fourier.pad_image(bz, pad_mode, pad_factor)

#     fft_bx = numpy_fft.fftshift(numpy_fft.fft2(bx_pad))
#     fft_bx = qdmpy.shared.fourier.set_naninf_to_zero(fft_bx)
#     fft_by = numpy_fft.fftshift(numpy_fft.fft2(by_pad))
#     fft_by = qdmpy.shared.fourier.set_naninf_to_zero(fft_by)
#     fft_bz = numpy_fft.fftshift(numpy_fft.fft2(bz_pad))
#     fft_bz = qdmpy.shared.fourier.set_naninf_to_zero(fft_bz)

#     ky, kx, k = qdmpy.shared.fourier.define_k_vectors(fft_bz.shape, pixel_size, k_vector_epsilon)

#     from qdmpy.constants import MU_0

#     if standoff:
#         exp_factor = np.exp(1 * k * standoff)
#     else:
#         exp_factor = 1

#     alpha = 2 * exp_factor / MU_0

#     fft_s = -1j * alpha * kx * fft_bx + 1j * alpha * ky * fft_by

#     if sigma is not None and sigma:
#         fft_s_real = Qitool.get_im_filtered(fft_s.real, "gaussian", sigma=sigma)
#         fft_s_imag = Qitool.get_im_filtered(fft_s.imag, "gaussian", sigma=sigma)
#         fft_s = 1j * fft_s_imag + fft_s_real

#     # forward eq: (I = complex unit) {1j added in debugging, woops}
#     #   -I ky  I kx
#     # {{-----, ----}, {kx, ky}} @ {jx, jy} = {bz, I FT{source term}}
#     #    a k   a k
#     # invert that LHS matrix:
#     #           kx             ky
#     # {{I a ky, --}, {-I a kx, --}}
#     #           k              k

#     fft_jx = +1j * alpha * ky * fft_bz + (kx / k) * 1j * fft_s
#     fft_jy = -1j * alpha * kx * fft_bz + (ky / k) * 1j * fft_s

#     if nv_layer_thickness and standoff:
#         arg = k * nv_layer_thickness / 2
#         nv_thickness_correction = np.sinh(arg) / arg
#     else:
#         nv_thickness_correction = 1

#     corrections = nv_thickness_correction * qdmpy.shared.fourier.hanning_filter_kspace(
#         k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
#     )

#     fft_jx = qdmpy.shared.fourier.set_naninf_to_zero(corrections * fft_jx)
#     fft_jy = qdmpy.shared.fourier.set_naninf_to_zero(corrections * fft_jy)

#     # fourier transform back into real space
#     jx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jx)).real
#     jy = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jy)).real

#     # only return non-padded region
#     jx_reg = qdmpy.shared.fourier.unpad_image(jx, padder)
#     jy_reg = qdmpy.shared.fourier.unpad_image(jy, padder)

#     return jx_reg, jy_reg


# ============================================================================


def get_j_from_bnv(
    bnv,
    unv,
    pad_mode,
    pad_factor,
    pixel_size,
    k_vector_epsilon,
    do_hanning_filter,
    hanning_low_cutoff,
    hanning_high_cutoff,
    standoff,
    nv_layer_thickness,
    nvs_above_sample,
):
    r"""(Single) Bnv measured -> Jxy via fourier methods.

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
    do_hanning_filter : bool
        Do a hanning filter?
    hanning_high_cutoff : float
        Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set
        via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff.
    hanning_low_cutoff : float
        Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set
        via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff.
    standoff : float
        Distance NV layer <-> Sample (in metres)
    nv_layer_thickness : float
        Thickness of NV layer (in metres)
    nvs_above_sample : bool
        True if NVs exist at higher z (in lab frame) than sample.

    Returns
    -------
    jx, jy : np arrays (2D)
        The calculated current density images, in A/m.

    See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky,
        J.-P. Tetienne, and L. C. L. Hollenberg,
        Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field
        Measurements, Phys. Rev. Applied 14, 024076 (2020).
        https://doi.org/10.1103/PhysRevApplied.14.024076
        https://arxiv.org/abs/2005.06788
    """
    b = copy(bnv) * 1e-4  # copy and convert Gauss to Tesla
    bnv_pad, padder = qdmpy.shared.fourier.pad_image(b, pad_mode, pad_factor)

    fft_bnv = numpy_fft.fftshift(numpy_fft.fft2(bnv_pad))

    ky, kx, k = qdmpy.shared.fourier.define_k_vectors(
        fft_bnv.shape, pixel_size, k_vector_epsilon
    )

    if nvs_above_sample:
        unv_cpy = copy(unv)
    else:
        unv_cpy = [-unv[0], -unv[1], unv[2]]

    # define transform
    bnv_to_jx, bnv_to_jy = define_current_transform(
        unv_cpy, ky, kx, k, standoff, nv_layer_thickness
    )

    hanning_filt = qdmpy.shared.fourier.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
    )

    bnv_to_jx = qdmpy.shared.fourier.set_naninf_to_zero(hanning_filt * bnv_to_jx)
    bnv_to_jy = qdmpy.shared.fourier.set_naninf_to_zero(hanning_filt * bnv_to_jy)

    fft_jx = bnv_to_jx * fft_bnv
    fft_jy = bnv_to_jy * fft_bnv

    # fourier transform back into real space
    jx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jx)).real
    jy = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jy)).real

    # only return non-padded region
    jx_reg = qdmpy.shared.fourier.unpad_image(jx, padder)
    jy_reg = qdmpy.shared.fourier.unpad_image(jy, padder)
    return jx_reg, jy_reg


# ============================================================================

# -*- coding: utf-8 -*-
"""
Implement inversion of magnetic field to magnetization source

Functions
---------
 - `qdmpy.source.magnetization.define_magnetization_transformation`
 - `qdmpy.source.magnetization.get_m_from_bxy`
 - `qdmpy.source.magnetization.get_m_from_bz`
 - `qdmpy.source.magnetization.get_m_from_bnv`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.shared.fourier.get_reconstructed_bfield": True,
    "qdmpy.source.magnetization.define_magnetization_transformation": True,
    "qdmpy.source.magnetization.get_m_from_bxy": True,
    "qdmpy.source.magnetization.get_m_from_bz": True,
    "qdmpy.source.magnetization.get_m_from_bnv": True,
}

# ============================================================================

import numpy as np
from pyfftw.interfaces import numpy_fft
from copy import copy

# ============================================================================

import qdmpy.shared.fourier
from qdmpy.shared.fourier import MU_0, MAG_UNIT_CONV

# ============================================================================
# ============================================================================


def define_magnetization_transformation(ky, kx, k, standoff):
    """M => b fourier-space transformation.


    Parameters
    ----------
    ky, kx, k : np array
        Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 )

    standoff : float
        Distance NV layer <-> Sample

    Returns
    -------
    d_matrix : np array
        Transformation such that B = d_matrix * m. E.g. for z magnetized sample:
        m_to_bnv = (
            unv[0] * d_matrix[2, 0, ::] + unv[1] * d_matrix[2, 1, ::] + unv[2] * d_matrix[2, 2, ::]
        )
        -> First index '2' is for z magnitisation (see m_from_bxy for in-plane mag process), the
        second index is for the measurement axis (0:x, 1:y, 2:z), and the last index iterates
        through the k values/vectors.


    See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky,
        J.-P. Tetienne, and L. C. L. Hollenberg,
        Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field
        Measurements, Phys. Rev. Applied 14, 024076 (2020).
        https://doi.org/10.1103/PhysRevApplied.14.024076
        https://arxiv.org/abs/2005.06788
    """

    if standoff:
        exp_factor = np.exp(1 * k * standoff)
    else:
        exp_factor = 1

    alpha = 2 * exp_factor / MU_0

    return (-1 / alpha) * np.array(
        [
            [kx ** 2 / k, (kx * ky) / k, 1j * kx],
            [(kx * ky) / k, ky ** 2 / k, 1j * ky],
            [1j * kx, 1j * ky, -k],
        ]
    )
    # return (1 / alpha) * np.array(
    #     [
    #         [-(kx ** 2 + 2 * ky ** 2) / k, kx * ky / k, 1j * kx],
    #         [kx * ky / k, -(2 * kx ** 2 + ky ** 2) / k, 1j * ky],
    #         [-1j * kx, -1j * ky, -k],
    #     ]
    # )


# ============================================================================


def get_m_from_bxy(
    bfield,
    mag_angle,
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
    r"""Bxy measured -> M via fourier methods.

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
    m : np array (2D)
        The calculated magnetization, in mu_B / nm^2

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

    ky, kx, k = qdmpy.shared.fourier.define_k_vectors(fft_bx.shape, pixel_size, k_vector_epsilon)

    # define transform
    d_matrix = define_magnetization_transformation(ky, kx, k, standoff)

    if mag_angle is None:
        m_to_bx = d_matrix[2, 0, ::]  # z magnetized
        m_to_by = d_matrix[2, 1, ::]
    else:
        psi = np.deg2rad(mag_angle)
        m_to_bx = np.cos(psi) * d_matrix[0, 0, ::] + np.sin(psi) * d_matrix[1, 0, ::]
        m_to_by = np.cos(psi) * d_matrix[0, 1, ::] + np.sin(psi) * d_matrix[1, 1, ::]

    hanning_filt = qdmpy.shared.fourier.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
    )

    if nv_layer_thickness and standoff:
        arg = k * nv_layer_thickness / 2
        nv_thickness_correction = np.sinh(arg) / arg
    else:
        nv_thickness_correction = 1

    with np.errstate(all="ignore"):
        fft_m_bx = fft_bx * hanning_filt * nv_thickness_correction / m_to_bx
        fft_m_by = fft_by * hanning_filt * nv_thickness_correction / m_to_by
        fft_m = (fft_m_bx + fft_m_by) / 2

    fft_m = qdmpy.shared.fourier.set_naninf_to_zero(fft_m)

    # fourier transform back into real space
    m = np.fft.ifft2(np.fft.ifftshift(fft_m)).real

    # only return non-padded region
    m_reg = qdmpy.shared.fourier.unpad_image(m, padder)

    return m_reg * MAG_UNIT_CONV


# ============================================================================


def get_m_from_bz(
    bfield,
    mag_angle,
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
    r"""Bz measured -> M via fourier methods.

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
    m : np array (2D)
        The calculated magnetization, in mu_B / nm^2

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

    ky, kx, k = qdmpy.shared.fourier.define_k_vectors(fft_bz.shape, pixel_size, k_vector_epsilon)

    # define transform
    d_matrix = define_magnetization_transformation(ky, kx, k, standoff)

    if mag_angle is None:
        m_to_bz = d_matrix[2, 2, ::]  # z magnetized
    else:
        psi = np.deg2rad(mag_angle)
        m_to_bz = np.cos(psi) * d_matrix[0, 2, ::] + np.sin(psi) * d_matrix[1, 2, ::]

    hanning_filt = qdmpy.shared.fourier.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
    )

    if nv_layer_thickness and standoff:
        arg = k * nv_layer_thickness / 2
        nv_thickness_correction = np.sinh(arg) / arg
    else:
        nv_thickness_correction = 1

    with np.errstate(all="ignore"):
        fft_m = fft_bz * hanning_filt * nv_thickness_correction / m_to_bz

    # Replace troublesome pixels in fourier space
    fft_m = qdmpy.shared.fourier.set_naninf_to_zero(fft_m)

    # fourier transform back into real space
    m = np.fft.ifft2(np.fft.ifftshift(fft_m)).real

    # only return non-padded region
    m_reg = qdmpy.shared.fourier.unpad_image(m, padder)

    return m_reg * MAG_UNIT_CONV


# ============================================================================


def get_m_from_bnv(
    bnv,
    unv,
    mag_angle,
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
    r"""(Single) Bnv measured -> M via fourier methods.

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
    m : np array (2D)
        The calculated magnetization, in mu_B / nm^2

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

    ky, kx, k = qdmpy.shared.fourier.define_k_vectors(fft_bnv.shape, pixel_size, k_vector_epsilon)

    # define transform
    d_matrix = define_magnetization_transformation(ky, kx, k, standoff)

    if nvs_above_sample:
        unv_cpy = copy(unv)
    else:
        unv_cpy = [-unv[0], -unv[1], unv[2]]

    m_to_bnv = None

    if mag_angle is None:
        m_to_bnv = (
            unv_cpy[0] * d_matrix[2, 0, ::]
            + unv_cpy[1] * d_matrix[2, 1, ::]
            + unv_cpy[2] * d_matrix[2, 2, ::]
        )  # z magnetized
    else:
        # if the flake is magnetized in plane than use this transformation instead
        b_axis = np.nonzero(unv_cpy)[0]
        psi = np.deg2rad(mag_angle)
        for idx in b_axis:
            new = unv_cpy[int(idx)] * (
                np.cos(psi) * d_matrix[0, int(idx), ::] + np.sin(psi) * d_matrix[1, int(idx), ::]
            )
            m_to_bnv = new if m_to_bnv is None else m_to_bnv + new

    hanning_filt = qdmpy.shared.fourier.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
    )

    if nv_layer_thickness and standoff:
        arg = k * nv_layer_thickness / 2
        nv_thickness_correction = np.sinh(arg) / arg
    else:
        nv_thickness_correction = 1

    with np.errstate(all="ignore"):
        # Get m_z from b_xyz
        fft_m = fft_bnv * hanning_filt * nv_thickness_correction / m_to_bnv

    # Replace troublesome pixels in fourier space
    fft_m = qdmpy.shared.fourier.set_naninf_to_zero(fft_m)

    # fourier transform back into real space
    m = np.fft.ifft2(np.fft.ifftshift(fft_m)).real

    # only return non-padded region
    m_reg = qdmpy.shared.fourier.unpad_image(m, padder)

    return m_reg * MAG_UNIT_CONV

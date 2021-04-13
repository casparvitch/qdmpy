# -*- coding: utf-8 -*-
"""
Interface to fourier module.

Functions
---------
 - `qdmpy.fourier.interface.prop_single_bnv`
 - `qdmpy.fourier.interface.get_reconstructed_bfield`
 - `qdmpy.fourier.interface.get_j_from_bxy`
 - `qdmpy.fourier.interface.get_j_from_bz`
 - `qdmpy.fourier.interface.get_j_from_bnv`
 - `qdmpy.fourier.interface.get_m_from_bxy`
 - `qdmpy.fourier.interface.get_m_from_bz`
 - `qdmpy.fourier.interface.get_m_from_bnv`
 - `qdmpy.fourier.interface.define_current_transform`
 - `qdmpy.fourier.interface.define_magnetisation_transformation`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.fourier.interface.prop_single_bnv": True,
    "qdmpy.fourier.interface.get_reconstructed_bfield": True,
    "qdmpy.fourier.interface.get_j_from_bxy": True,
    "qdmpy.fourier.interface.get_j_from_bz": True,
    "qdmpy.fourier.interface.get_j_from_bnv": True,
    "qdmpy.fourier.interface.get_m_from_bxy": True,
    "qdmpy.fourier.interface.get_m_from_bz": True,
    "qdmpy.fourier.interface.get_m_from_bnv": True,
    "qdmpy.fourier.interface.define_current_transform": True,
    "qdmpy.fourier.interface.define_magnetisation_transformation": True,
}

# ============================================================================

from pyfftw.interfaces import numpy_fft
from copy import copy
import numpy as np

# ============================================================================

import qdmpy.fourier._shared
from qdmpy.constants import MU_0, MAG_UNIT_CONV

# ============================================================================


def prop_single_bnv(
    single_bnv,
    unv,
    pad_mode,
    pad_factor,
    pixel_size,
    k_vector_epsilon,
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
    padded_bnv, padder = qdmpy.fourier._shared.pad_image(bnv, pad_mode, pad_factor)

    fft_bnv = numpy_fft.fftshift(numpy_fft.fft2(padded_bnv))
    ky, kx, k = qdmpy.fourier._shared.define_k_vectors(fft_bnv.shape, pixel_size, k_vector_epsilon)

    # define transformation matrices -> e.g. see Casola 2018 given above
    bnv2bx = 1 / (unv[0] + unv[1] * ky / kx + 1j * unv[2] * k / kx)
    bnv2by = 1 / (unv[0] * kx / ky + unv[1] + 1j * unv[2] * k / ky)
    bnv2bz = 1 / (-1j * unv[0] * kx / k - 1j * unv[1] * ky / k + unv[2])

    bnv2bx = qdmpy.fourier._shared.set_naninf_to_zero(bnv2bx)
    bnv2by = qdmpy.fourier._shared.set_naninf_to_zero(bnv2by)
    bnv2bz = qdmpy.fourier._shared.set_naninf_to_zero(bnv2bz)

    # transform to xyz
    fft_bx = fft_bnv * bnv2bx
    fft_by = fft_bnv * bnv2by
    fft_bz = fft_bnv * bnv2bz

    # fourier transform back into real space
    bx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_bx)).real
    by = numpy_fft.ifft2(numpy_fft.ifftshift(fft_by)).real
    bz = numpy_fft.ifft2(numpy_fft.ifftshift(fft_bz)).real

    # only return non-padded region
    bx_reg = qdmpy.fourier._shared.unpad_image(bx, padder)
    by_reg = qdmpy.fourier._shared.unpad_image(by, padder)
    bz_reg = qdmpy.fourier._shared.unpad_image(bz, padder)

    return bx_reg, by_reg, bz_reg


# ============================================================================


def get_reconstructed_bfield(
    bfield,
    pad_mode,
    pad_factor,
    pixel_size,
    k_vector_epsilon,
):
    r"""
    Bxyz measured -> Bxyz_recon via fourier methods.

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
    bx_recon, by_recon, bz_recon : np arrays (2D)
        The reconstructed bfield maps.

    For a proper explanation of methodology, see [CURR_RECON]_.

    $$  \nabla \times {\bf B} = 0 $$

    to get Bx_recon and By_recon from Bz (in a source-free region), and

    $$ \nabla \cdot {\bf B} = 0 $$

    to get Bz_recon from Bx and By

    Start with e.g.:

    $$ \frac{\partial B_x^{\rm recon}}{\partial z} = \frac{\partial B_z}{\partial x} $$

    with the definitions

    $$ \hat{B}:=  \hat{\mathcal{F}}_{xy} \big\{ B \big\} $$

    and

    $$ k = \sqrt{k_x^2 + k_y^2} $$

    we have:

    $$ \frac{\partial }{\partial z} \hat{B}_x^{\rm recon}(x,y,z=z_{\rm NV}) = i k_x \hat{B}_z(x,y, z=z_{\rm NV}) $$.

    Now using upward continuation [CURR_RECON]_ to evaluate the z partial:

    $$ -k \hat{B}_x^{\rm recon}(x,y,z=z_{\rm NV}) = i k_x \hat{B}_z(k_x, k_y, z_{\rm NV}) $$

    such that for

    $$ k \neq 0 $$

    we have

    $$ (\hat{B}_x^{\rm recon}(x,y,z=z_{\rm NV}), \hat{B}_y^{\rm recon}(x,y,z=z_{\rm NV})) = \frac{-i}{k} (k_x, k_y) \hat{B}_z(x,y,,z=z_{\rm NV}) $$


    Utilising the zero-divergence property of the magnetic field, it can also be shown:

    $$ \hat{B}_z^{\rm recon}(x,y,z=z_{\rm NV}) = \frac{i}{k} \left( k_x \hat{B}_x(x,y,z=z_{\rm NV}) + k_y \hat{B}_y(x,y,z=z_{\rm NV}) \right) $$

    References
    ----------
    .. [CURR_RECON] E. A. Lima and B. P. Weiss,
                    Obtaining Vector Magnetic Field Maps from Single-Component Measurements of
                    Geological Samples, Journal of Geophysical Research: Solid Earth 114, (2009).
                    https://doi.org/10.1029/2008JB006006

    """

    bx, by, bz = copy(bfield)
    # first pad each comp
    bx_pad, padder = qdmpy.fourier._shared.pad_image(bx, pad_mode, pad_factor)
    by_pad, _ = qdmpy.fourier._shared.pad_image(by, pad_mode, pad_factor)
    bz_pad, _ = qdmpy.fourier._shared.pad_image(bz, pad_mode, pad_factor)

    fft_bx = numpy_fft.fftshift(numpy_fft.fft2(bx_pad))
    fft_by = numpy_fft.fftshift(numpy_fft.fft2(by_pad))
    fft_bz = numpy_fft.fftshift(numpy_fft.fft2(bz_pad))
    fft_bx = qdmpy.fourier._shared.set_naninf_to_zero(fft_bx)
    fft_by = qdmpy.fourier._shared.set_naninf_to_zero(fft_by)
    fft_bz = qdmpy.fourier._shared.set_naninf_to_zero(fft_bz)

    ky, kx, k = qdmpy.fourier._shared.define_k_vectors(fft_bx.shape, pixel_size, k_vector_epsilon)

    # could chuck in an 'NV_above_or_below' sign here. 2020-04-13 swapped minus sign
    # -> assumes NV above source
    bz2bx = (1j / k) * kx
    bz2by = (1j / k) * ky
    bx2bz = -(1j / k) * kx
    by2bz = -(1j / k) * ky

    bz2bx = qdmpy.fourier._shared.set_naninf_to_zero(bz2bx)
    bz2by = qdmpy.fourier._shared.set_naninf_to_zero(bz2by)
    bx2bz = qdmpy.fourier._shared.set_naninf_to_zero(bx2bz)
    by2bz = qdmpy.fourier._shared.set_naninf_to_zero(by2bz)

    fft_bx_recon = fft_bz * bz2bx
    fft_by_recon = fft_bz * bz2by

    fft_bz_recon = fft_bx * bx2bz + fft_by * by2bz

    # fourier transform back into real space
    bx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_bx_recon)).real
    by = numpy_fft.ifft2(numpy_fft.ifftshift(fft_by_recon)).real
    bz = numpy_fft.ifft2(numpy_fft.ifftshift(fft_bz_recon)).real

    # only return non-padded region
    bx_reg = qdmpy.fourier._shared.unpad_image(bx, padder)
    by_reg = qdmpy.fourier._shared.unpad_image(by, padder)
    bz_reg = qdmpy.fourier._shared.unpad_image(bz, padder)
    return bx_reg, by_reg, bz_reg


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
    bx_pad, padder = qdmpy.fourier._shared.pad_image(bx, pad_mode, pad_factor)
    by_pad, _ = qdmpy.fourier._shared.pad_image(by, pad_mode, pad_factor)

    fft_bx = numpy_fft.fftshift(numpy_fft.fft2(bx_pad))
    fft_by = numpy_fft.fftshift(numpy_fft.fft2(by_pad))
    fft_bx = qdmpy.fourier._shared.set_naninf_to_zero(fft_bx)
    fft_by = qdmpy.fourier._shared.set_naninf_to_zero(fft_by)

    ky, kx, k = qdmpy.fourier._shared.define_k_vectors(fft_bx.shape, pixel_size, k_vector_epsilon)

    # define transform
    _, bx_to_jy = define_current_transform([1, 0, 0], ky, kx, k, standoff)
    by_to_jx, _ = define_current_transform([0, 1, 0], ky, kx, k, standoff)

    hanning_filt = qdmpy.fourier._shared.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
    )

    if nv_layer_thickness and standoff:
        arg = k * nv_layer_thickness / 2
        nv_thickness_correction = np.sinh(arg) / arg
    else:
        nv_thickness_correction = 1

    bx_to_jy = qdmpy.fourier._shared.set_naninf_to_zero(
        hanning_filt * nv_thickness_correction * bx_to_jy
    )
    by_to_jx = qdmpy.fourier._shared.set_naninf_to_zero(
        hanning_filt * nv_thickness_correction * by_to_jx
    )

    fft_jx = by_to_jx * fft_by
    fft_jy = bx_to_jy * fft_bx

    # fourier transform back into real space
    jx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jx)).real
    jy = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jy)).real

    # only return non-padded region
    jx_reg = qdmpy.fourier._shared.unpad_image(jx, padder)
    jy_reg = qdmpy.fourier._shared.unpad_image(jy, padder)
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
    bz_pad, padder = qdmpy.fourier._shared.pad_image(bz, pad_mode, pad_factor)

    fft_bz = numpy_fft.fftshift(numpy_fft.fft2(bz_pad))
    fft_bz = qdmpy.fourier._shared.set_naninf_to_zero(fft_bz)

    ky, kx, k = qdmpy.fourier._shared.define_k_vectors(fft_bz.shape, pixel_size, k_vector_epsilon)

    # define transformation
    bz_to_jx, bz_to_jy = define_current_transform([0, 0, 1], ky, kx, k, standoff)

    hanning_filt = qdmpy.fourier._shared.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
    )

    if nv_layer_thickness and standoff:
        arg = k * nv_layer_thickness / 2
        nv_thickness_correction = np.sinh(arg) / arg
    else:
        nv_thickness_correction = 1

    bz_to_jx = qdmpy.fourier._shared.set_naninf_to_zero(
        hanning_filt * nv_thickness_correction * bz_to_jx
    )
    bz_to_jy = qdmpy.fourier._shared.set_naninf_to_zero(
        hanning_filt * nv_thickness_correction * bz_to_jy
    )

    fft_jx = bz_to_jx * fft_bz
    fft_jy = bz_to_jy * fft_bz

    # fourier transform back into real space
    jx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jx)).real
    jy = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jy)).real

    # only return non-padded region
    jx_reg = qdmpy.fourier._shared.unpad_image(jx, padder)
    jy_reg = qdmpy.fourier._shared.unpad_image(jy, padder)
    return jx_reg, jy_reg


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
    bnv_pad, padder = qdmpy.fourier._shared.pad_image(b, pad_mode, pad_factor)

    fft_bnv = numpy_fft.fftshift(numpy_fft.fft2(bnv_pad))

    ky, kx, k = qdmpy.fourier._shared.define_k_vectors(fft_bnv.shape, pixel_size, k_vector_epsilon)

    # define transform
    bnv_to_jx, bnv_to_jy = define_current_transform(unv, ky, kx, k, standoff)

    hanning_filt = qdmpy.fourier._shared.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
    )

    if nv_layer_thickness and standoff:
        arg = k * nv_layer_thickness / 2
        nv_thickness_correction = np.sinh(arg) / arg
    else:
        nv_thickness_correction = 1

    bnv_to_jx = qdmpy.fourier._shared.set_naninf_to_zero(
        hanning_filt * nv_thickness_correction * bnv_to_jx
    )
    bnv_to_jy = qdmpy.fourier._shared.set_naninf_to_zero(
        hanning_filt * nv_thickness_correction * bnv_to_jy
    )

    fft_jx = bnv_to_jx * fft_bnv
    fft_jy = bnv_to_jy * fft_bnv

    # fourier transform back into real space
    jx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jx)).real
    jy = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jy)).real

    # only return non-padded region
    jx_reg = qdmpy.fourier._shared.unpad_image(jx, padder)
    jy_reg = qdmpy.fourier._shared.unpad_image(jy, padder)
    return jx_reg, jy_reg


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
        The calculated magnetisation, in mu_B / nm^2

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
    bx_pad, padder = qdmpy.fourier._shared.pad_image(bx, pad_mode, pad_factor)
    by_pad, _ = qdmpy.fourier._shared.pad_image(by, pad_mode, pad_factor)

    fft_bx = numpy_fft.fftshift(numpy_fft.fft2(bx_pad))
    fft_by = numpy_fft.fftshift(numpy_fft.fft2(by_pad))
    fft_bx = qdmpy.fourier._shared.set_naninf_to_zero(fft_bx)
    fft_by = qdmpy.fourier._shared.set_naninf_to_zero(fft_by)

    ky, kx, k = qdmpy.fourier._shared.define_k_vectors(fft_bx.shape, pixel_size, k_vector_epsilon)

    # define transform
    d_matrix = define_magnetisation_transformation(ky, kx, k, standoff)

    if not mag_angle or mag_angle is None:
        m_to_bx = d_matrix[2, 0, ::]  # z magnetised
        m_to_by = d_matrix[2, 1, ::]
    else:
        psi = np.deg2rad(mag_angle)
        m_to_bx = np.cos(psi) * d_matrix[0, 0, ::] + np.sin(psi) * d_matrix[1, 0, ::]
        m_to_by = np.cos(psi) * d_matrix[0, 1, ::] + np.sin(psi) * d_matrix[1, 1, ::]

    hanning_filt = qdmpy.fourier._shared.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
    )

    if nv_layer_thickness and standoff:
        arg = k * nv_layer_thickness / 2
        nv_thickness_correction = np.sinh(arg) / arg
    else:
        nv_thickness_correction = 1

    fft_m_bx = fft_bx * hanning_filt * nv_thickness_correction / m_to_bx
    fft_m_by = fft_by * hanning_filt * nv_thickness_correction / m_to_by
    fft_m = (fft_m_bx + fft_m_by) / 2

    fft_m = qdmpy.fourier._shared.set_naninf_to_zero(fft_m)

    # fourier transform back into real space
    m = np.fft.ifft2(np.fft.ifftshift(fft_m)).real

    # only return non-padded region
    m_reg = qdmpy.fourier._shared.unpad_image(m, padder)

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
        The calculated magnetisation, in mu_B / nm^2

    See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky,
        J.-P. Tetienne, and L. C. L. Hollenberg,
        Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field
        Measurements, Phys. Rev. Applied 14, 024076 (2020).
        https://doi.org/10.1103/PhysRevApplied.14.024076
        https://arxiv.org/abs/2005.06788
    """
    bz = copy(bfield[2]) * 1e-4  # copy and convert Gauss to Tesla
    bz_pad, padder = qdmpy.fourier._shared.pad_image(bz, pad_mode, pad_factor)

    fft_bz = numpy_fft.fftshift(numpy_fft.fft2(bz_pad))
    fft_bz = qdmpy.fourier._shared.set_naninf_to_zero(fft_bz)

    ky, kx, k = qdmpy.fourier._shared.define_k_vectors(fft_bz.shape, pixel_size, k_vector_epsilon)

    # define transform
    d_matrix = define_magnetisation_transformation(ky, kx, k, standoff)

    if not mag_angle or mag_angle is None:
        m_to_bz = d_matrix[2, 2, ::]  # z magnetised
    else:
        psi = np.deg2rad(mag_angle)
        m_to_bz = np.cos(psi) * d_matrix[0, 2, ::] + np.sin(psi) * d_matrix[1, 2, ::]

    hanning_filt = qdmpy.fourier._shared.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
    )

    if nv_layer_thickness and standoff:
        arg = k * nv_layer_thickness / 2
        nv_thickness_correction = np.sinh(arg) / arg
    else:
        nv_thickness_correction = 1

    fft_m = fft_bz * hanning_filt * nv_thickness_correction / m_to_bz

    # Replace troublesome pixels in fourier space
    fft_m = qdmpy.fourier._shared.set_naninf_to_zero(fft_m)

    # fourier transform back into real space
    m = np.fft.ifft2(np.fft.ifftshift(fft_m)).real

    # only return non-padded region
    m_reg = qdmpy.fourier._shared.unpad_image(m, padder)

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

    Returns
    -------
    m : np array (2D)
        The calculated magnetisation, in mu_B / nm^2

    See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky,
        J.-P. Tetienne, and L. C. L. Hollenberg,
        Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field
        Measurements, Phys. Rev. Applied 14, 024076 (2020).
        https://doi.org/10.1103/PhysRevApplied.14.024076
        https://arxiv.org/abs/2005.06788
    """

    b = copy(bnv) * 1e-4  # copy and convert Gauss to Tesla
    bnv_pad, padder = qdmpy.fourier._shared.pad_image(b, pad_mode, pad_factor)

    fft_bnv = numpy_fft.fftshift(numpy_fft.fft2(bnv_pad))

    ky, kx, k = qdmpy.fourier._shared.define_k_vectors(fft_bnv.shape, pixel_size, k_vector_epsilon)

    # define transform
    d_matrix = define_magnetisation_transformation(ky, kx, k, standoff)

    m_to_bnv = None

    if not mag_angle or mag_angle is None:
        m_to_bnv = (
            unv[0] * d_matrix[2, 0, ::] + unv[1] * d_matrix[2, 1, ::] + unv[2] * d_matrix[2, 2, ::]
        )  # z magnetised
    else:
        # if the flake is magnetised in plane than use this transformation instead
        b_axis = np.nonzero(unv)[0]
        psi = np.deg2rad(mag_angle)
        for idx in b_axis:
            new = unv[int(idx)] * (
                np.cos(psi) * d_matrix[0, int(idx), ::] + np.sin(psi) * d_matrix[1, int(idx), ::]
            )
            m_to_bnv = new if m_to_bnv is None else m_to_bnv + new

    hanning_filt = qdmpy.fourier._shared.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
    )

    if nv_layer_thickness and standoff:
        arg = k * nv_layer_thickness / 2
        nv_thickness_correction = np.sinh(arg) / arg
    else:
        nv_thickness_correction = 1

    # Get m_z from b_xyz
    fft_m = fft_bnv * hanning_filt * nv_thickness_correction / m_to_bnv

    # Replace troublesome pixels in fourier space
    fft_m = qdmpy.fourier._shared.set_naninf_to_zero(fft_m)

    # fourier transform back into real space
    m = np.fft.ifft2(np.fft.ifftshift(fft_m)).real

    # only return non-padded region
    m_reg = qdmpy.fourier._shared.unpad_image(m, padder)

    return m_reg * MAG_UNIT_CONV


# ============================================================================


def define_current_transform(u_proj, ky, kx, k, standoff=None):
    """b => J fourier-space transformation.

    Arguments
    ---------
    u_proj : array-like
        Shape: 3, the direction the magnetic field was measured on (projected onto).
    ky, kx, k : np arrays
        Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 )

    standoff : float or None, default : None
        Distance NV layer <-> sample

    Returns
    -------
    b_to_jx, b_to_jy : np arrays (2D)

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

    b_to_jx = (alpha * ky) / (-u_proj[1] * ky - u_proj[0] * kx + 1j * u_proj[2] * k)
    b_to_jy = (alpha * kx) / (u_proj[0] * kx + u_proj[1] * ky - 1j * u_proj[2] * k)

    return b_to_jx, b_to_jy


# ============================================================================


def define_magnetisation_transformation(ky, kx, k, standoff):
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
        Transformation such that B = d_matrix * m. E.g. for z magnetised sample:
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

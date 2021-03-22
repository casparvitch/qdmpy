# -*- coding: utf-8 -*-
"""
TODO module for all general fourier methods.
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {"qdmpy.fourier._shared.": True}

# ============================================================================

from pyfftw.interfaces import numpy_fft
from copy import copy
import numpy as np

# ============================================================================

import qdmpy.fourier._shared
from qdmpy.constants import MU_0

# ============================================================================


def prop_single_bnv(
    single_bnv,
    unv,
    pad_mode,
    pad_factor,
    pixel_size,
    k_vector_epsilon,
    do_hanning_filter,
    hanning_high_cutoff,
    hanning_low_cutoff,
):
    r"""[summary]

    [description]

    \hat{\bf B} = {\bf v} \hat{B}_z({\bf k})

    (hat denotes 2D fourier transform, \vec{k} is 2D wavevector) and where

    {\bf v} = (-ik_x / k, -ik_y / k, 1)

    \hat{B}_z({\bf k}) = \frac{\hat{B}_{\rm NV}}{{\bf u}_{\rm NV} \cdot {\bf v}}


    See Casola 2018 Nature Review Materials, http://dx.doi.org/10.1038/natrevmats.2017.88
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

    # get hanning filter, remove invalid elements
    hanning_filt = qdmpy.fourier._shared.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff
    )
    bnv2bx = qdmpy.fourier._shared.set_naninf_to_zero(hanning_filt * bnv2bx)
    bnv2by = qdmpy.fourier._shared.set_naninf_to_zero(hanning_filt * bnv2by)
    bnv2bz = qdmpy.fourier._shared.set_naninf_to_zero(hanning_filt * bnv2bz)

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
    do_hanning_filter,
    hanning_low_cutoff,
    hanning_high_cutoff,
):
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

    bz2bx = -(1j / k) * kx
    bz2by = -(1j / k) * ky
    bx2bz = (1j / k) * kx
    by2bz = (1j / k) * ky

    hanning_filt = qdmpy.fourier._shared.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff
    )
    bz2bx = qdmpy.fourier._shared.set_naninf_to_zero(hanning_filt * bz2bx)
    bz2by = qdmpy.fourier._shared.set_naninf_to_zero(hanning_filt * bz2by)
    bx2bz = qdmpy.fourier._shared.set_naninf_to_zero(hanning_filt * bx2bz)
    by2bz = qdmpy.fourier._shared.set_naninf_to_zero(hanning_filt * by2bz)

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
):
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
    _, bx_to_jy = define_current_transform([1, 0, 0], kx, ky, k, standoff)
    by_to_jx, _ = define_current_transform([0, 1, 0], kx, ky, k, standoff)

    hanning_filt = qdmpy.fourier._shared.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff
    )
    bx_to_jy = qdmpy.fourier._shared.set_naninf_to_zero(hanning_filt * bx_to_jy)
    by_to_jx = qdmpy.fourier._shared.set_naninf_to_zero(hanning_filt * by_to_jx)

    fft_jx = by_to_jx * fft_by
    fft_jy = bx_to_jy * fft_bx

    # fourier transform back into real space
    jx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jx)).real
    jy = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jy)).real

    # only return non-padded region
    jx_reg = qdmpy.fourier._shared.unpad_image(fft_jx, padder)
    jy_reg = qdmpy.fourier._shared.unpad_image(fft_jy, padder)
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
):
    bz = copy(bfield[2]) * 1e-4  # copy and convert Gauss to Tesla
    bz_pad, padder = qdmpy.fourier._shared.pad_image(bz, pad_mode, pad_factor)

    fft_bz = numpy_fft.fftshift(numpy_fft.fft2(bz_pad))
    fft_bz = qdmpy.fourier._shared.set_naninf_to_zero(fft_bz)

    ky, kx, k = qdmpy.fourier._shared.define_k_vectors(fft_bz.shape, pixel_size, k_vector_epsilon)

    # define transform
    bz_to_jx, bz_to_jy = define_current_transform([0, 0, 1], kx, ky, k, standoff)

    hanning_filt = qdmpy.fourier._shared.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff
    )
    bz_to_jx = qdmpy.fourier._shared.set_naninf_to_zero(hanning_filt * bz_to_jx)
    bz_to_jy = qdmpy.fourier._shared.set_naninf_to_zero(hanning_filt * bz_to_jy)

    fft_jx = bz_to_jx * fft_bz
    fft_jy = bz_to_jy * fft_bz

    # fourier transform back into real space
    jx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jx)).real
    jy = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jy)).real

    # only return non-padded region
    jx_reg = qdmpy.fourier._shared.unpad_image(fft_jx, padder)
    jy_reg = qdmpy.fourier._shared.unpad_image(fft_jy, padder)
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
):
    b = copy(bnv) * 1e-4  # copy and convert Gauss to Tesla
    bnv_pad, padder = qdmpy.fourier._shared.pad_image(b, pad_mode, pad_factor)

    fft_bnv = numpy_fft.fftshift(numpy_fft.fft2(bnv_pad))

    ky, kx, k = qdmpy.fourier._shared.define_k_vectors(fft_bnv.shape, pixel_size, k_vector_epsilon)

    # define transform
    bnv_to_jx, bnv_to_jy = define_current_transform(unv, kx, ky, k, standoff)

    hanning_filt = qdmpy.fourier._shared.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff
    )
    bnv_to_jx = qdmpy.fourier._shared.set_naninf_to_zero(hanning_filt * bnv_to_jx)
    bnv_to_jy = qdmpy.fourier._shared.set_naninf_to_zero(hanning_filt * bnv_to_jy)

    fft_jx = bnv_to_jx * fft_bnv
    fft_jy = bnv_to_jy * fft_bnv

    # fourier transform back into real space
    jx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jx)).real
    jy = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jy)).real

    # only return non-padded region
    jx_reg = qdmpy.fourier._shared.unpad_image(fft_jx, padder)
    jy_reg = qdmpy.fourier._shared.unpad_image(fft_jy, padder)
    return jx_reg, jy_reg


# ============================================================================


def define_current_transform(u_proj, kx, ky, k, standoff=None):
    """[summary]
    b => J

    See Broadway 2020 http://dx.doi.org/10.1103/PhysRevApplied.14.024076
    """
    if standoff:
        exp_factor = np.exp(-1 * k * standoff)
    else:
        exp_factor = 1

    alpha = 2 * exp_factor / MU_0

    # NOTE sign on ky terms was negative on pwpy? due to -ky ine define transform?
    b_to_jx = (alpha * ky) / (-u_proj[1] * ky - u_proj[0] * kx + 1j * u_proj[2] * k)
    b_to_jy = (alpha * kx) / (u_proj[0] * kx + u_proj[1] * ky - 1j * u_proj[2] * k)

    return b_to_jx, b_to_jy

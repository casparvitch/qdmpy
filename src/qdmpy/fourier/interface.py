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

# ============================================================================

import qdmpy.fourier._shared

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
    bnv = copy(single_bnv)

    # first pad bnv
    padded_bnv, padder = qdmpy.fourier._shared.pad_image(bnv, pad_mode, pad_factor)

    fft_bnv = numpy_fft.fftshift(numpy_fft.fft2(padded_bnv))
    ky, kx, k = qdmpy.fourier._shared.define_k_vectors(fft_bnv.shape, pixel_size, k_vector_epsilon)

    # define transformation matrices TODO add a source for this...
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
    plot_im(bnv)
    plot_im(padded_bnv)
    plot_im(bx)
    plot_im(bx_reg)
    print(unv, padder)
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

    # write function in fourier (fspace_transform?)
    # pad/extrap., FFT, multiply, IFFT (+ relevant shifts), unpad_image(?), Real.

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

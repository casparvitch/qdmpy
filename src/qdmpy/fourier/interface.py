# -*- coding: utf-8 -*-
"""
TODO module for all general fourier methods.
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {"qdmpy.fourier._shared.": True}

# ============================================================================

import numpy as numpy
from pyfftw.interfaces import numpy_fft

# ============================================================================

import qdmpy.fourier._shared

# ============================================================================


def prop_single_bnv(single_bnv, unv, pad_mode, pad_factor):

    # first pad bnv
    padded_bnv, padder = qdmpy.fourier._shared.pad_image(single_bnv, pad_mode, pad_factor)

    fft_bnv = numpy_fft.fftshift(numpy_fft.fft2(padded_bnv))
    kx, ky, k = qdmpy.fourier._shared.define_k_vectors(
        fft_bnv.shape
    )  # TODO check output of this guy

    # define transformation matrices
    bnv2bx = 1 / (unv[0] + unv[1] * ky / kx + 1j * unv[2] * k / kx)
    bnv2by = 1 / (unv[0] * kx / ky + unv[1] + 1j * unv[2] * k / ky)
    bnv2bz = 1 / (-1j * unv[0] * kx / k - 1j * unv[1] * ky / k + unv[2])

    # get hanning filter

    # remove invalid elements from transformation matrices

    # transform to xyz
    fft_bx = fft_bnv * bnv2bx
    fft_by = fft_bnv * bnv2by
    fft_bz = fft_bnv * bnv2bz

    # fourier transform back into real space
    bx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_bx)).real
    by = numpy_fft.ifft2(numpy_fft.ifftshift(fft_by)).real
    bz = numpy_fft.ifft2(numpy_fft.ifftshift(fft_bz)).real

    # only return non-padded region
    bx_reg = qdmpy.fourier._shared.unpad(bx, padder)
    by_reg = qdmpy.fourier._shared.unpad(by, padder)
    bz_reg = qdmpy.fourier._shared.unpad(bz, padder)
    return bx_reg, by_reg, bz_reg

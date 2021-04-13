# -*- coding: utf-8 -*-
"""
This module holds shared tooling for fourier functions/methods.

Functions
---------
 - `qdmpy.fourier._shared.unpad_image`
 - `qdmpy.fourier._shared.pad_image`
 - `qdmpy.fourier._shared.define_k_vectors`
 - `qdmpy.fourier._shared.set_naninf_to_zero`
 - `qdmpy.fourier._shared.hanning_filter_kspace`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.fourier._shared.unpad_image": True,
    "qdmpy.fourier._shared.pad_image": True,
    "qdmpy.fourier._shared.define_k_vectors": True,
    "qdmpy.fourier._shared.set_naninf_to_zero": True,
    "qdmpy.fourier._shared.hanning_filter_kspace": True,
}

# ============================================================================

import numpy as np
from pyfftw.interfaces import numpy_fft

# ============================================================================


# ============================================================================


def unpad_image(x, padder):
    """undo a padding defined by `QDMPy.fourier._shared.pad_image` (it returns
    the padder list)"""
    slices = []
    for c in padder:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]


# ============================================================================


def pad_image(image, pad_mode, pad_factor):
    """
    pad_mode -> see np.pad
    pad_factor -> either side of image
    """

    if len(np.shape(image)) != 2:
        raise ValueError("image passed to pad_image was not 2D.")

    image = np.array(image)

    if pad_mode is None:
        return image, ((0, 0), (0, 0))

    size_y, size_x = image.shape

    y_pad = pad_factor * size_y
    x_pad = pad_factor * size_x
    padder = ((y_pad, y_pad), (x_pad, x_pad))
    padded_image = np.pad(image, mode=pad_mode, pad_width=padder)

    return padded_image, padder


# ============================================================================


def define_k_vectors(shape, pixel_size, k_vector_epsilon):
    """Get scaled k vectors (as meshgrid) for fft.

    Arguments
    ----------
    shape : list
        Shape of fft array to get k vectors for.
    pixel_size : float
        Pixel size, e.g. options["system"].get_raw_pixel_size(options) * options["total_bin"].
    k_vector_epsilon : float
        Add an epsilon value to the k-vectors to avoid some issues with 1/0.

    Returns
    -------
    ky, kx, k : np array
        Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 )
    """

    # scaling for the k vectors so they are in the right units
    scaling = np.float64(2 * np.pi / pixel_size)

    # get the fft frequencies and shift the ordering and forces type to be float64
    ky_vec = scaling * numpy_fft.fftshift(numpy_fft.fftfreq(shape[0]))
    kx_vec = scaling * numpy_fft.fftshift(numpy_fft.fftfreq(shape[1]))

    # Include a small factor in the k vectors to remove division by zero issues (min_k)
    # Make a meshgrid to pass back
    if k_vector_epsilon:
        ky, kx = np.meshgrid(ky_vec + k_vector_epsilon, kx_vec - k_vector_epsilon, indexing="ij")
    else:
        ky, kx = np.meshgrid(ky_vec, kx_vec, indexing="ij")

    k = np.sqrt(ky ** 2 + kx ** 2)
    return ky, -kx, k  # negative here to maintain correct image orientation (ass. NV above source)


# ============================================================================


def set_naninf_to_zero(array):
    """ replaces NaNs and infs with zero"""
    idxs = np.logical_or(np.isnan(array), np.isinf(array))
    array[idxs] = 0
    return array


# ============================================================================


def hanning_filter_kspace(k, do_filt, hanning_low_cutoff, hanning_high_cutoff, standoff):
    """Computes a hanning image filter with both low and high pass filters.

    Arguments
    ---------
    k : np array
        Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 )
    do_filt : bool
        Do a hanning filter?
    hanning_high_cutoff : float
        Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set
        via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff.
    hanning_low_cutoff : float
        Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set
        via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff.
    standoff : float
        Distance NV layer <-> Sample.

    Returns
    -------
    img_filter : (2d array, float)
        bandpass filter to remove artifacts in the FFT process.
    """
    # Define Hanning filter to prevent noise amplification at frequencies higher than the
    # spatial resolution

    if do_filt and standoff:
        hy = np.hanning(k.shape[0])
        hx = np.hanning(k.shape[1])
        img_filt = np.sqrt(np.outer(hy, hx))
        # apply cutoffs
        if hanning_high_cutoff is not None:
            k_cut_high = (2 * np.pi) / hanning_high_cutoff
        else:
            k_cut_high = (2 * np.pi) / standoff
            img_filt[k > k_cut_high] = 0
        if hanning_low_cutoff is not None:
            k_cut_low = (2 * np.pi) / hanning_low_cutoff
            img_filt[k < k_cut_low] = 0
    else:
        img_filt = 1
    return img_filt


# ============================================================================

# -*- coding: utf-8 -*-
"""
This module holds ...

Functions
---------
 - `qdmpy.fourier._shared.`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {"qdmpy.fourier._shared.": True}

# ============================================================================

import numpy as np
from pyfftw.interfaces import numpy_fft

# ============================================================================


# ============================================================================


def unpad_image(x, padder):
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

    if np.shape(image) != 2:
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
    # scaling for the k vectors so they are in the right units
    scaling = np.float64(2 * np.pi / pixel_size)

    # get the fft frequencies and shift the ordering and forces type to be float64
    ky_vec = scaling * numpy_fft.fftshift(numpy_fft.fftfreq(shape[0]))
    kx_vec = scaling * numpy_fft.fftshift(numpy_fft.fftfreq(shape[1]))

    # Include a small factor in the k vectors to remove division by zero issues (min_k)
    # Make a meshgrid to pass back
    # NOTE careful here do you need 'ij' indexing
    if k_vector_epsilon:
        ky, kx = np.meshgrid(
            ky_vec + k_vector_epsilon,
            kx_vec - k_vector_epsilon,
        )
    else:
        ky, kx = np.meshgrid(ky_vec, kx_vec)

    k = np.sqrt(ky ** 2 + kx ** 2)
    return ky, kx, k

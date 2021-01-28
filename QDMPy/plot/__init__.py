"""
Sub-package for plotting results of QDMPy.

This (sub-) package exposes specific functions to the user:
---------
- `QDMPy.plot.common`:
    - `QDMPy.plot.common.set_mpl_rcparams`
    - `QDMPy.plot.common.plot_image`
    - `QDMPy.plot.common.plot_image_on_ax`
- `QDMPy.plot.fits`:
    - `QDMPy.plot.fits.plot_ROI_PL_image`
    - `QDMPy.plot.fits.plot_AOI_PL_images`
    - `QDMPy.plot.fits.plot_ROI_avg_fits`
    - `QDMPy.plot.fits.plot_AOI_spectra`
    - `QDMPy.plot.fits.plot_AOI_spectra_fit`
    - `QDMPy.plot.fits.plot_param_image`
    - `QDMPy.plot.fits.plot_param_images`
"""

from QDMPy.plot.common import set_mpl_rcparams, plot_image, plot_image_on_ax  # noqa: F401

from QDMPy.plot.fits import (
    plot_ROI_PL_image,
    plot_AOI_PL_images,
    plot_ROI_avg_fits,
    plot_AOI_spectra,
    plot_AOI_spectra_fit,
    plot_param_image,
    plot_param_images,
)

from QDMPy.plot.bnvs import *

"""
Sub-package for plotting results of qdmpy.

This (sub-) package exposes specific functions to the user:
---------
- `qdmpy.plot.common`:
    - `qdmpy.plot.common.set_mpl_rcparams`
    - `qdmpy.plot.common.plot_image`
    - `qdmpy.plot.common.plot_image_on_ax`
- `qdmpy.plot.fit`:
    - `qdmpy.plot.fit.plot_ROI_PL_image`
    - `qdmpy.plot.fit.plot_AOI_PL_images`
    - `qdmpy.plot.fit.plot_ROI_avg_fits`
    - `qdmpy.plot.fit.plot_AOI_spectra`
    - `qdmpy.plot.fit.plot_AOI_spectra_fit`
    - `qdmpy.plot.fit.plot_param_image`
    - `qdmpy.plot.fit.plot_param_images`
- `qdmpy.plot.field`, currently all contents
"""
__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.plot.common": True,
    "qdmpy.plot.field": True,
    "qdmpy.plot.fit": True,
}

from qdmpy.plot.common import *

from qdmpy.plot.fit import *

from qdmpy.plot.field import *

"""
Sub-package for plotting results of qdmpy.

This (sub-) package exposes specific functions to the user:
---------
- `qdmpy.plot.common`:
    - `qdmpy.plot.common.set_mpl_rcparams`
    - `qdmpy.plot.common.plot_image`
    - `qdmpy.plot.common.plot_image_on_ax`
- `qdmpy.plot.fit`:
    - `qdmpy.plot.fit.plot_ROI_pl_image`
    - `qdmpy.plot.fit.plot_AOI_pl_images`
    - `qdmpy.plot.fit.plot_ROI_avg_fits`
    - `qdmpy.plot.fit.plot_AOI_spectra`
    - `qdmpy.plot.fit.plot_AOI_spectra_fit`
    - `qdmpy.plot.fit.plot_param_image`
    - `qdmpy.plot.fit.plot_param_images`
- `qdmpy.plot.field`, currently all contents
- `qdmpy.plot.source`, currently all contents
"""
__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.plot.common": True,
    "qdmpy.plot.field": True,
    "qdmpy.plot.fit": True,
    "qdmpy.plot.source": True,
}

from qdmpy.plot.common import *

from qdmpy.plot.pl import *

from qdmpy.plot.field import *

from qdmpy.plot.source import *

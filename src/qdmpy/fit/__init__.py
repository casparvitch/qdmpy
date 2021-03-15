# -*- coding: utf-8 -*-
"""
Sub-package for fitting widefield NV microscopy (photoluminescence) data.

This (sub-) package exposes all of the concents of `qdmpy.fit.interface`, as well
as qdmpy.fit.model (FitModel class etc.)
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.fit._gpufit": False,  # by default don't document as pygpufit not always available.
    "qdmpy.fit._scipyfit": True,
    "qdmpy.fit._shared": True,
    "qdmpy.fit._funcs": True,
    "qdmpy.fit.model": True,
    "qdmpy.fit.interface": True,
}

from qdmpy.fit.interface import *
from qdmpy.fit.model import *

# -*- coding: utf-8 -*-
"""
Sub-package for fitting widefield NV microscopy (photoluminescence) data.

This (sub-) package exposes all of the concents of `QDMPy.fit.interface`, as well
as QDMPy.fit.model (FitModel class etc.)
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.fit._gpufit": False,  # by default don't document as pygpufit not always available.
    "QDMPy.fit._scipyfit": True,
    "QDMPy.fit._shared": True,
    "QDMPy.fit._funcs": True,
    "QDMPy.fit.model": True,
    "QDMPy.fit.interface": True,
}

from QDMPy.fit.interface import *
from QDMPy.fit.model import *

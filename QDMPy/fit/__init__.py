# -*- coding: utf-8 -*-
"""
Sub-package for fitting widefield NV microscopy (photoluminescence) data.

This (sub-) package exposes all of the concents of `QDMPy.fit.interface`, as well
as the FitModel class
"""

__pdoc__ = {
    "QDMPy.fit._gpufit": False,
    "QDMPy.fit._scipyfit": True,
    "QDMPy.fit._shared": True,
    "QDMPy.fit._models": True,
    "QDMPy.fit._funcs": True,
    "QDMPy.fit.interface": True,
}

from QDMPy.fit.interface import *
from QDMPy.fit._models import FitModel

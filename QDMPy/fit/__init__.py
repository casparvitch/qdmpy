# -*- coding: utf-8 -*-
"""
Sub-package for fitting widefield NV microscopy (photoluminescence) data.

This (sub-) package exposes all of the concents of `QDMPy.fit.interface`, as well
as 1QDMPy.fit.model1 (FitModel class etc.)
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.fit._gpufit": False,
    "QDMPy.fit._scipyfit": True,
    "QDMPy.fit._shared": True,
    "QDMPy.fit._funcs": True,
    "QDMPy.fit.model": True,
    "QDMPy.fit.interface": True,
}

from QDMPy.fit.interface import *
from QDMPy.fit.model import *

# -*- coding: utf-8 -*-
"""
Sub-package for fitting widefield NV microscopy (photoluminescence) data.

This (sub-) package exposes all of the concents of `QDMPy.fit.interface`.
"""

__pdoc__ = {
    "QDMPy.fit._gpufit": False,
    "QDMPy.fit._scipyfit": True,
    "QDMPy.fit._shared": True,
    "QDMPy.fit._models": True,
    "QDMPy.fit.interface": True,
}


from QDMPy.fit.interface import *

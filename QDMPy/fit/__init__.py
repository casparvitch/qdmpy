"""
Sub-package for fitting widefield NV microscopy data.

This (sub-) package exposes all of the concents of `QDMPy.fit.interface`.
"""

__pdoc__ = {
    "QDMPy.fit._gpufit": True,
    "QDMPy.fit._scipyfit": True,
    "QDMPy.fit._shared": True,
    "QDMPy.fit._models": True,
}


from QDMPy.fit.interface import *

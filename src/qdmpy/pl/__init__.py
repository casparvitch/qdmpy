# -*- coding: utf-8 -*-
"""
Sub-package for fitting widefield NV microscopy (photoluminescence) data.

This (sub-) package exposes all of the concents of `qdmpy.pl.interface,
`qdmpy.pl.model` (FitModel class etc.) and `qdmpy.pl.io`
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.pl.interface": True,
    "qdmpy.pl.common": True,
    "qdmpy.pl.funcs": True,
    "qdmpy.pl.gpufit": False,  # by default don't document as pygpufit not always available.
    "qdmpy.pl.io": True,
    "qdmpy.pl.model": True,
    "qdmpy.pl.scipyfit": True,
}

from qdmpy.pl.interface import *  # noqa: F401, F403
from qdmpy.pl.model import *  # noqa: F401, F403
from qdmpy.pl.io import *  # noqa: F401, F403

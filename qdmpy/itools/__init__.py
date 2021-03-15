# -*- coding: utf-8 -*-
"""
Image tools sub-package.

This (sub-) package exposes all of the concents of `qdmpy.itools.interface`,
as well as the `qdmpy.itools._polygon.Polygon` object.
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.itools.interface": True,
    "qdmpy.itools._polygon": True,
    "qdmpy.itools._filter": True,
    "qdmpy.itools._mask": True,
    "qdmpy.itools._bground": True,
}
from qdmpy.itools.interface import *
from qdmpy.itools._polygon import Polygon

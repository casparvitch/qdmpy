# -*- coding: utf-8 -*-
"""
Image tools sub-package.

This (sub-) package exposes all of the concents of `qdmpy.itool.interface`,
as well as the `qdmpy.itool._polygon.Polygon` object.
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.itool.interface": True,
    "qdmpy.itool._polygon": True,
    "qdmpy.itool._filter": True,
    "qdmpy.itool._mask": True,
    "qdmpy.itool._bground": True,
}
from qdmpy.itool.interface import *
from qdmpy.itool._polygon import Polygon

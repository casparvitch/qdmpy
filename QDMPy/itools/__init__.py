# -*- coding: utf-8 -*-
"""
Image tools sub-package.

This (sub-) package exposes all of the concents of `QDMPy.itools.interface`,
as well as the `QDMPy.itools._polygon.Polygon` object.
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.itools.interface": True,
    "QDMPy.itools._polygon": True,
    "QDMPy.itools._filter": True,
    "QDMPy.itools._mask": True,
    "QDMPy.itools._bground": True,
}
from QDMPy.itools.interface import *
from QDMPy.itools._polygon import Polygon

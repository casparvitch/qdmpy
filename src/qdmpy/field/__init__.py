# -*- coding: utf-8 -*-
"""Sub-package for converting pixel fitting data to fields (i.e. magnetic, electric, strain).

Currently this is just DC fields, but AC fields could be added in the future (e.g. from
T1 measurements).

This (sub-) package exposes all of the concents of `qdmpy.field.interface` and `qdmpy.field.io`
"""
__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.field._bnv": True,
    "qdmpy.field._bxyz": True,
    "qdmpy.field._geom": True,
    "qdmpy.field.interface": True,
}

from qdmpy.field.interface import *
from qdmpy.field.io import *

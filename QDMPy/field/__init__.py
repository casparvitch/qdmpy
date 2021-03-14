"""
Sub-package for converting pixel fitting data to fields (i.e. magnetic, electric, strain).

Currently this is just DC fields, but AC fields could be added in the future (e.g. from
T1 measurements).

This (sub-) package exposes all of the concents of `QDMPy.field.interface`
"""
__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.field._bnv": True,
    "QDMPy.field._bxyz": True,
    "QDMPy.field._geom": True,
    "QDMPy.field.interface": True,
}

from QDMPy.field.interface import *

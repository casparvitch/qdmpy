"""Quantum Diamond MicroscoPy: A module/package for analysing widefield NV microscopy images.

'Super-module' that holds all of the others within.

Structure of sub-modules: FIXME

- `qdmpy.field`
    - Field module. Contains functions to convert bnvs/resonances to fields (e.g.
     magnetic, electric, ...) through hamiltonian fits/otherwise.
- `qdmpy.magsim`
    - Not completed. FIXME. 
- `qdmpy.pl`
    - Module for dealing with pl data. Contains procedures for fitting raw photoliminescence,
      outputting results etc.
- `qdmpy.plot`
    - This module contains all of the plotting functions (matplotlib based).
- `qdmpy.source`
    - Contains tools for reconstructing source fields (e.g. current densities or intrinsic 
      magnetization) from the measured magnetic field calculated in qdmpy.field.
- `qdmpy.shared`
    - Contains procedures shared between the other higher level modules. Cannot import from the
      other modules or you'll get circular import errors. Specific tooling here includes those
      to help with fourier transforms, NV geometry, image tooling such as filtering and 
      background subtraction, as well as json io and polygon selection.
- `qdmpy.system`
    - This modules contains the tooling for defining institution specific settings for example
     for loading raw datafiles etc. These settings can be implemented down to the specific
     experimental 'system' to define pixel sizes etc.

qdmpy itself also exposes some functions from qdmpy.interface
"""

from qdmpy.interface import *  # noqa: F401, F403

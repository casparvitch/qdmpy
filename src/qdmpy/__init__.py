"""Quantum Diamond MicroscoPy: A module/package for analysing widefield NV microscopy images.

'Super-module' that holds all of the others within.

Structure of sub-modules:

- `qdmpy.field`
    + Field module. Contains functions to convert bnvs/resonances to fields (e.g.
     magnetic, electric, ...) through the hamiltonian model.
- `qdmpy.fit`
    - Fitting module. This module contains the functions etc. to fit raw photoluminescence
     data.
- `qdmpy.fourier`
    - This module has not been implemented yet, but will contain fourier tooling for
     other modules, in particular the sources module.
- `qdmpy.hamiltonian`
    - Module holding hamiltonian models. Fit bnvs/resonances to a hamiltonian model
     to extract useful field parameters. Shouldn't be used on its own, but through the
     field module.
- `qdmpy.io`
    - input/output module. Here you can find functions for saving/loading data to/from disk.
- `qdmpy.itool`
    - 'Image tools' module. This module contains miscellaneous tools for working with images,
     such as background removal, filtering, masking and polygon annotations.
- `qdmpy.plot`
    - This module contains all of the plotting functions (matplotlib based).
- `qdmpy.source`
    - This module has not been implemented yet, but will contain tools for reconstructing
     source fields (e.g. current densities or intrinsic magnetisation) from the measured
     vector magnetic field calculated from qdmpy.field.
- `qdmpy.constants`
    - This file contains useful global constants, all defined in one place for easy editing.
     E.g. the spin-1 operator matrices are defined here.
- `qdmpy.system`
    - This modules contains the tooling for defining institution specific settings for example
     for loading raw datafiles etc. These settings can be implemented down to the specific
     experimental 'system' to define pixel sizes etc.

qdmpy itself also exposes some functions from qdmpy.interface
"""

from qdmpy.interface import *  # noqa: F401, F403

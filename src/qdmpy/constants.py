# -*- coding: utf-8 -*-
"""Module defining global constants of the qdmpy package.

Of particular interest to the user are the SYSTEMS and AVAILABLE_FNS dictionaries.

You can customise/extend the qdmpy package by defining a subclass of the
`qdmpy.system.systems.System` object for the specifics of your experimental setup.
To communicate this addition to qdmpy, add it to the SYSTEMS dictionary
(e.g. import qdmpy.constants; qdmpy.constants.SYSTEMS["System_Name"] = MySystem).
Where we have defined MySystem: class MySystem(qdmpy.system.systems.System): ... etc.
Follow the templates in `qdmpy.system.systems` to construct your object.

The AVAILABLE_FNS dictionary defines the available `qdmpy.fit.model.FitModel`s.
These fit models are used in all fit backends (scipyfit and gpufit at the time of
writing). However, the numerics (e.g. actually evaluating the model) are only used
by the scipyfit backend, and only the parameter definitions are used by gpufit.
To add a fit model to gpufit, see the gpufit source
(https://gpufit.readthedocs.io/en/latest/). The gpufit ModelIDs are currently
hard-coded into qdmpy - Lorentzians 1-8 peaks and a Stretched Exponential
(for e.g. T_1 experiments) are expected by qdmpy to be defined in pygpufit.

Functions
---------
 - `qdmpy.constants.choose_system`

Module variables
----------------
 - `qdmpy.constants.SYSTEMS`
 - `qdmpy.constants.AVAILABLE_FNS`
 - `qdmpy.constants.GAMMA`
 - `qdmpy.constants.GSLAC`
 - `qdmpy.constants.MU_0`
 - `qdmpy.constants.MAG_UNIT_CONV`
 - `qdmpy.constants.S_MAT_X`
 - `qdmpy.constants.S_MAT_Y`
 - `qdmpy.constants.S_MAT_Z`
 - `qdmpy.constants.NV_AXES_100_110`
 - `qdmpy.constants.NV_AXES_100_100`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.constants.SYSTEMS": True,
    "qdmpy.constants.AVAILABLE_FNS": True,
    "qdmpy.constants.choose_system": True,
    "qdmpy.constants.GAMMA": True,
    "qdmpy.constants.GSLAC": True,
    "qdmpy.constants.MU_0": True,
    "qdmpy.constants.MAG_UNIT_CONV": True,
    "qdmpy.constants.S_MAT_X": True,
    "qdmpy.constants.S_MAT_Y": True,
    "qdmpy.constants.S_MAT_Z": True,
    "qdmpy.constants.NV_AXES_100_110": True,
    "qdmpy.constants.NV_AXES_100_100": True,
}

# ============================================================================

import numpy as np

# ============================================================================

import qdmpy.system
import qdmpy.fit._funcs
import qdmpy.hamiltonian._hamiltonians

# ============================================================================


SYSTEMS = {
    "Zyla": qdmpy.system.Zyla,
    "Liams_Widefield": qdmpy.system.LiamsWidefield,
    "Cryo_Widefield": qdmpy.system.CryoWidefield,
    "cQDM": qdmpy.system.cQDM,
}
"""Dictionary that defines systems available for use.

Add any systems you define here so you can use them.
"""

AVAILABLE_FNS = {
    "lorentzian": qdmpy.fit._funcs.Lorentzian,
    "lorentzian_hyperfine_14": qdmpy.fit._funcs.LorentzianHyperfine14,
    "lorentzian_hyperfine_15": qdmpy.fit._funcs.LorentzianHyperfine15,
    "gaussian": qdmpy.fit._funcs.Gaussian,
    "gaussian_hyperfine_14": qdmpy.fit._funcs.GaussianHyperfine14,
    "gaussian_hyperfine_15": qdmpy.fit._funcs.GaussianHyperfine15,
    "constant": qdmpy.fit._funcs.Constant,
    "linear": qdmpy.fit._funcs.Linear,
    "circular": qdmpy.fit._funcs.Circular,
    "stretched_exponential": qdmpy.fit._funcs.StretchedExponential,
    "damped_rabi": qdmpy.fit._funcs.DampedRabi,
}
"""Dictionary that defines fit functions available for use.

Add any functions you define here so you can use them.

Aviod overlapping function parameter names.
"""

AVAILABLE_HAMILTONIANS = {
    "approx_bxyz": qdmpy.hamiltonian._hamiltonians.ApproxBxyz,
    "bxyz": qdmpy.hamiltonian._hamiltonians.Bxyz,
}
"""Dictionary that defines hamiltonians available for use.

Add any classes you define here so you can use them.

You do not need to avoid overlapping parameter names as hamiltonian
classes can not be used in combination.
"""


def choose_system(name):
    """Returns `qdmpy.system.systems.System` object called 'name'."""
    return SYSTEMS[name]()


S_MAT_X = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
r"""Spin-1 operator: S{\rm X}"""
S_MAT_Y = np.array([[0, -1j, 0], [1j, 0, 1j], [0, 1j, 0]]) / np.sqrt(2)
r"""Spin-1 operator: S{\rm Y}"""
S_MAT_Z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
r"""Spin-1 operator: S{\rm Z}"""


GAMMA = 2.80  # MHz/G
r"""
The Bohr magneton times the Landé g-factor. See [Doherty2013](https://doi.org/10.1016/j.physrep.2013.02.001)
for details of the g-factor anisotropy.

|                                                                  |                                                               |
|------------------------------------------------------------------|---------------------------------------------------------------|
| \( \gamma_{\rm NV} = \mu_{\rm B} g_e  \)                         |                                                               |
| \( \mu_B = 1.39962449361 \times 10^{10}\ {\rm Hz} \rm{T}^{-1} \) |  [NIST](https://physics.nist.gov/cgi-bin/cuu/Value?mubshhz)   |
| \( \mu_B = 1.399...\ {\rm MHz/G} \)                              |                                                               |
| \( g_e \approx 2.0023 \)                                         |  [Doherty2013](https://doi.org/10.1016/j.physrep.2013.02.001) |
| \( \Rightarrow  \gamma_{\rm NV} \approx 2.80 {\rm MHz/G} \)      |                                                               |

"""

GSLAC = 1024
"""
Ground state level anticrossing (in Gauss).
Used to determine if a single-peak ODMR resonance is shifting to larger or smaller field.
(if bias field magnitude is larger than the GSLAC value bnv is reversed).
Currently NOT used for ODMR with >1 resonance.
"""

MU_0 = np.pi * 1e-7
"""
Vacuum permeability
"""

MAG_UNIT_CONV = 1e-18 / 9.274010e-24
"""
Convert unit for magnetisation to something more helpful.

SI unit measured: Amps: A

More useful: Bohr magnetons per nanometre squared: mu_B nm^-2

mu_B -> 9.274 010 e-24 A m^+2 or J/T
m^2 -> 1e+18 nm^2

Measure x amps = x A
 def  mu_B = 9.2_        in units of A m^2
 => x A =  x (1 / 9.2_)  in units of mu_B/m^2
 => x A = (1e-18/9.2_)   in units of mu_B/nm^2
"""

# NOTE for other NV orientations, pass in unvs -> not possible to determine in full
#   generality the orientations for <111> etc.

# nv orientations (unit vectors) wrt lab frame [x, y, z]
NV_AXES_100_110 = [
    {"nv_number": 0, "ori": [np.sqrt(2 / 3), 0, np.sqrt(1 / 3)]},
    {"nv_number": 1, "ori": [-np.sqrt(2 / 3), 0, np.sqrt(1 / 3)]},
    {"nv_number": 2, "ori": [0, np.sqrt(2 / 3), -np.sqrt(1 / 3)]},
    {"nv_number": 3, "ori": [0, -np.sqrt(2 / 3), -np.sqrt(1 / 3)]},
]
"""
<100> top face oriented, <110> edge face oriented diamond (CVD).

NV orientations (unit vectors) relative to lab frame.

Assuming diamond is square to lab frame:

first 3 numbers: orientation of top face of diamond, e.g. <100>

second 3 numbers: orientation of edges of diamond, e.g. <110>

CVD Diamonds are usually <100>, <110>. HPHT usually <100>, <100>.

![](https://i.imgur.com/Rudnzyo.png)

Purple plane corresponds to top (or bottom) face of diamond, orange planes correspond to edge faces.
"""

NV_AXES_100_100 = [
    {"nv_number": 0, "ori": [np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]},
    {"nv_number": 1, "ori": [-np.sqrt(1 / 3), -np.sqrt(1 / 3), np.sqrt(1 / 3)]},
    {"nv_number": 2, "ori": [np.sqrt(1 / 3), -np.sqrt(1 / 3), -np.sqrt(1 / 3)]},
    {"nv_number": 3, "ori": [-np.sqrt(1 / 3), np.sqrt(1 / 3), -np.sqrt(1 / 3)]},
]
"""
<100> top face oriented, <100> edge face oriented diamond (HPHT).

NV orientations (unit vectors) relative to lab frame.

Assuming diamond is square to lab frame:

first 3 numbers: orientation of top face of diamond, e.g. <100>

second 3 numbers: orientation of edges of diamond, e.g. <110>

CVD Diamonds are usually <100>, <110>. HPHT usually <100>, <100>.

![](https://i.imgur.com/cpErjAH.png)

Purple plane: top face of diamond, orange plane: edge faces.
"""

NV_AXES_111 = [
    {"nv_number": 0, "ori": [0, 0, 1]},
    {"nv_number": 1, "ori": [np.nan, np.nan, np.nan]},
    {"nv_number": 2, "ori": [np.nan, np.nan, np.nan]},
    {"nv_number": 3, "ori": [np.nan, np.nan, np.nan]},
]
"""
<111> top face oriented.

NV orientations (unit vectors) relative to lab frame.

Only the first nv can be oriented in general. This constant
is defined as a convenience for single-bnv <111> measurements.

<111> diamonds have an NV family oriented in z, i.e. perpindicular
to the diamond surface.
"""

# -*- coding: utf-8 -*-
"""
This module defines global constants of the QDMPy package.

Of particular interest to the user are the SYSTEMS and AVAILABLE_FNS dictionaries.

You can customise/extend the QDMPy package by defining a subclass of the
`QDMPy.systems.System` object for the specifics of your experimental setup.
To communicate this addition to QDMPy, add it to the SYSTEMS dictionary
(e.g. import QDMPy.constants; QDMPy.constants.SYSTEMS["System_Name"] = MySystem).
Where we have defined MySystem: class MySystem(QDMPy.systems.System): ... etc.
Follow the templates in `QDMPy.systems` to construct your object.

The AVAILABLE_FNS dictionary defines the available `QDMPy.fit._models.FitModel`s.
These fit models are used in all fit backends (scipyfit and gpufit at the time of
writing). However, the numerics (e.g. actually evaluating the model) are only used
by the scipyfit backend, and only the parameter definitions are used by gpufit.
To add a fit model to gpufit, see the gpufit source
(https://gpufit.readthedocs.io/en/latest/). The gpufit ModelIDs are currently
hard-coded into QDMPy - Lorentzians 1-8 peaks and a Stretched Exponential
(for e.g. T_1 experiments) are expected by QDMPy to be defined in pygpufit.

Functions
---------
 - `QDMPy.constants.choose_system`

Module variables
----------------
 - `QDMPy.constants.SYSTEMS`
 - `QDMPy.constants.AVAILABLE_FNS`
 - `QDMPy.constants.GAMMA`
 - `QDMPy.constants.NV_AXES_100_110`
 - `QDMPy.constants.NV_AXES_100_100`
 - `QDMPy.constants.NV_AXES_111_110`
 - `QDMPy.constants.NV_AXES_111_100`
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.constants.SYSTEMS": True,
    "QDMPy.constants.AVAILABLE_FNS": True,
    "QDMPy.constants.choose_system": True,
    "QDMPy.constants.GAMMA": True,
    "QDMPy.constants.NV_AXES_100_110": True,
    "QDMPy.constants.NV_AXES_100_100": True,
    "QDMPy.constants.NV_AXES_111_110": True,
    "QDMPy.constants.NV_AXES_111_100": True,
}

# ============================================================================

import numpy as np

# ============================================================================

import QDMPy.systems
import QDMPy.fit._models

# ============================================================================


SYSTEMS = {"Zyla": QDMPy.systems.Zyla, "Liams_Widefield": QDMPy.systems.LiamsWidefield}
"""
Dictionary that defines systems available for use.

Add any systems you define here so you can use them.
"""

AVAILABLE_FNS = {
    "lorentzian": QDMPy.fit._models.Lorentzian,
    "lorentzian_hyperfine_14": QDMPy.fit._models.Lorentzian_hyperfine_14,
    "lorentzian_hyperfine_15": QDMPy.fit._models.Lorentzian_hyperfine_15,
    "gaussian": QDMPy.fit._models.Gaussian,
    "gaussian_hyperfine_14": QDMPy.fit._models.Gaussian_hyperfine_14,
    "gaussian_hyperfine_15": QDMPy.fit._models.Gaussian_hyperfine_15,
    "constant": QDMPy.fit._models.Constant,
    "linear": QDMPy.fit._models.Linear,
    "circular": QDMPy.fit._models.Circular,
    "stretched_exponential": QDMPy.fit._models.Stretched_exponential,
}
"""
Dictionary that defines fit functions available for use.

Add any functions you define here so you can use them.

Aviod overlapping function parameter names.
"""


def choose_system(name):
    """Returns `QDMPy.systems.System` object called 'name'"""
    return SYSTEMS[name]()


# TODO document these
S_MAT_X = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
S_MAT_Y = np.array([[0, -1j, 0], [1j, 0, 1j], [0, 1j, 0]]) / np.sqrt(2)
S_MAT_Z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
S_MAT_X_AR = np.array([S_MAT_X, S_MAT_X, S_MAT_X, S_MAT_X]).T
S_MAT_Y_AR = np.array([S_MAT_Y, S_MAT_Y, S_MAT_Y, S_MAT_Y]).T
S_MAT_Z_AR = np.array([S_MAT_Z, S_MAT_Z, S_MAT_Z, S_MAT_Z]).T

GAMMA = 2.8  # MHz/G
r"""
The Bohr magneton times the Landé g-factor. See [Doherty2013](https://doi.org/10.1016/j.physrep.2013.02.001)
for details of the g-factor anisotropy.

|                                                                  |                                                               |
|------------------------------------------------------------------|---------------------------------------------------------------|
| \( \gamma_{\rm NV} = \mu_{\rm B} g_e  \)                         |                                                               |
| \( \mu_B = 1.39962449361 \times 10^{10}\ {\rm Hz} \rm{T}^{-1} \) |  [NIST](https://physics.nist.gov/cgi-bin/cuu/Value?mubshhz)   |
| \( \mu_B = 1.399...\ {\rm MHz/G} \)                              |                                                               |
| \( g_e \approx 2.0023 \)                                         |  [Doherty2013](https://doi.org/10.1016/j.physrep.2013.02.001) |
| \( \Rightarrow  \gamma_{\rm NV} \approx 2.8 {\rm MHz/G} \)       |                                                               |

"""

# nv orientations (unit vectors) wrt lab frame [x, y, z]
NV_AXES_100_110 = [
    {"nv_number": 1, "ori": [np.sqrt(2 / 3), 0, np.sqrt(1 / 3)]},
    {"nv_number": 2, "ori": [-np.sqrt(2 / 3), 0, np.sqrt(1 / 3)]},
    {"nv_number": 3, "ori": [0, np.sqrt(2 / 3), -np.sqrt(1 / 3)]},
    {"nv_number": 4, "ori": [0, -np.sqrt(2 / 3), -np.sqrt(1 / 3)]},
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
    {"nv_number": 1, "ori": [np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]},
    {"nv_number": 2, "ori": [-np.sqrt(1 / 3), -np.sqrt(1 / 3), np.sqrt(1 / 3)]},
    {"nv_number": 3, "ori": [np.sqrt(1 / 3), -np.sqrt(1 / 3), -np.sqrt(1 / 3)]},
    {"nv_number": 4, "ori": [-np.sqrt(1 / 3), np.sqrt(1 / 3), -np.sqrt(1 / 3)]},
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

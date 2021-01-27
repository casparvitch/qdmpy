# -*- coding: utf-8 -*-
"""
This module defines global constants of the QDMPy package.

Of particular interest to the user are the SYSTEMS and AVAILABLE_FNS dictionaries.

You can customise/extend the QDMPy package by defining a subclass of the
`QDMPy.systems.System` object for the specifics of your experimental setup.
To communicate this additon to QDMPy, add it to the SYSTEMS dictionary
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
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.constants.SYSTEMS": True,
    "QDMPy.constants.AVAILABLE_FNS": True,
}

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

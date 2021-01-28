# -*- coding: utf-8 -*-
"""
This module holds the tools for loading/saving Bfield results.

Functions
---------
 - `QDMPy.io.bfield.save_bnvs_and_dshifts`

"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.io.bfield.save_bnvs_and_dshifts": True,
}

# ============================================================================

import numpy as np

# ============================================================================


def save_bnvs_and_dshifts(options, name, bnvs, dshifts):
    if bnvs:
        for i in bnvs:
            np.savetxt(options["data_dir"] / f"{name}_bnv_{i}.txt", bnvs)
    if dshifts:
        for i in dshifts:
            np.savetxt(options["data_dir"] / f"{name}_dshift_{i}.txt", dshifts)

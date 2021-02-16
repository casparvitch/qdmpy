# -*- coding: utf-8 -*-
"""
This module holds the tools for loading/saving field results.

Functions
---------
 - `QDMPy.io.fields.save_bnvs_and_dshifts`

"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.io.fields.save_bnvs_and_dshifts": True,
}

# ============================================================================

import numpy as np

# ============================================================================


def save_bnvs_and_dshifts(options, name, bnvs, dshifts):
    if bnvs:
        for i, bnv in enumerate(bnvs):
            np.savetxt(options["sub_ref_data_dir"] / f"{name}_bnv_{i}.txt", bnv)
    if dshifts:
        for i, dshift in enumerate(dshifts):
            np.savetxt(options["sub_ref_data_dir"] / f"{name}_dshift_{i}.txt", dshift)

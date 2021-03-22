# -*- coding: utf-8 -*-
"""
This module holds the tools for loading/saving source results.

Functions
---------
 - `qdmpy.io.source.`
"""

# ============================================================================

import numpy as np
import warnings
import os
from pathlib import Path

# ============================================================================

# ============================================================================


def prep_output_directories(options):
    options["source_dir"] = options["output_dir"].joinpath("field")
    if not os.path.isdir(options["source_dir"]):
        os.mkdir(options["source_dir"])


# ============================================================================


def save_source_params(options, source_params):
    if source_params:
        for param_key, result in source_params.items():
            if result is not None:
                path = options["source_dir"] / f"{param_key}.txt"
                np.savetxt(path, result)

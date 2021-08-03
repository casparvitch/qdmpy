# -*- coding: utf-8 -*-
"""
This module holds the tools for loading/saving source results.

Functions
---------
 - `qdmpy.source.io.prep_output_directories`
 - `qdmpy.source.io.save_source_params`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.source.io.prep_output_directories": True,
    "qdmpy.source.io.save_source_params": True,
}

# ============================================================================

import numpy as np
import os

# ============================================================================

# ============================================================================


def prep_output_directories(options):
    options["source_dir"] = options["output_dir"].joinpath("source")
    if not os.path.isdir(options["source_dir"]):
        os.mkdir(options["source_dir"])


# ============================================================================


def save_source_params(options, source_params):
    if source_params:
        for param_key, result in source_params.items():
            if result is not None:
                path = options["source_dir"] / f"{param_key}.txt"
                np.savetxt(path, result)

# -*- coding: utf-8 -*-
"""
This module defines some ease-of-use methods for the qdmpy package.

Functions
---------
 - `qdmpy.interface.initialize`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.interface.initialize": True,
}
# ============================================================================

import qdmpy.opt
import qdmpy.plot

# ============================================================================


def initialize(
    options_dict=None,
    options_path=None,
    ref_options_dict=None,
    ref_options_dir=None,
    set_mpl_rcparams=True,
):
    """Helped function to initialise analysis program.

    Arguments
    ---------
    options_dict : dict, default=None
        Generic options dict holding all the user options (for the main/signal experiment).
    options_path : str or path object, default=None
        Direct path to options json, i.e. will run something like 'read(options_path)'.
    ref_options_dict : dict, default=None
        Generic options dict holding all the user options (for the reference experiment).
    ref_options_dir : str or path object, default=None
        Path to read reference options from,
        i.e. will run something like 'read('ref_options_dir / saved_options.json')'.

    Returns
    -------
    options_dict : dict
        (Processed) generic options dict holding all user options.
    ref_options_dict : dict
        As options_dict, but for reference experiment (assuming pl already fit).

    """

    options = qdmpy.opt.load_options(
        options_dict=options_dict,
        options_path=options_path,
        check_for_prev_result=True,
        loading_ref=False,
    )

    if set_mpl_rcparams:
        qdmpy.plot.set_mpl_rcparams(options)

    ref_options = qdmpy.opt.load_ref_options(
        options, ref_options=ref_options_dict, ref_options_dir=ref_options_dir
    )

    return options, ref_options

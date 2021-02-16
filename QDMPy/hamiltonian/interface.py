# -*- coding: utf-8 -*-
"""
blaa
"""

# ============================================================================


import warnings

import QDMPy.hamiltonian._hamiltonians

# ============================================================================


def define_hamiltonian(options):
    from QDMPy.constants import AVAILABLE_HAMILTONIANS
    ham = AVAILABLE_HAMILTONIANS(options["hamiltonian"])

    options["ham_param_defn"] = QDMPy.hamiltonian._hamiltonians.get_param_defn(ham)

    _prep_fit_backends(options, ham)


def fit_pixels():
    pass


def _prep_fit_backends(options, ham):
    """
    Prepare all possible fit backends, checking that everything will work.

    Also attempts to import relevant modules into global scope.

    This is a wrapper around specific functions for each backend. All possible fit
    backends are loaded - these are decided in the config file for this system,
    i.e. system.option_choices("fit_backend")

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    ham : `QDMPy.hamiltonian._hamiltonians.Hamiltonian`
        Model we're fitting to.
    """
    # only scipyfit supported currently
    global fit_scipyfit
    _temp = __import__("QDMPy.hamiltonian._scipyfit", globals(), locals())
    fit_scipyfit = _temp.hamiltonian._scipyfit

    fit_scipyfit.prep_scipyfit_backend(options, ham)
    # TODO implement above. WAIT not sure that's necessary?

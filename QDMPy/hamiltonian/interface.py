# -*- coding: utf-8 -*-
"""
blaa
"""

# ============================================================================

import numpy as np

# ============================================================================

import QDMPy.hamiltonian._hamiltonians

# ============================================================================


class Chooser:
    """ DOCSTRING HERE """

    def __init__(self, chooser_ar):
        self.chooser_ar = chooser_ar

    def __call__(self, some_ar):
        return np.array([some_ar[i] for i, do_use in enumerate(self.chooser_ar) if do_use])


# ============================================================================


def define_hamiltonian(options, chooser_obj, unv_frames):
    from QDMPy.constants import AVAILABLE_HAMILTONIANS

    ham = AVAILABLE_HAMILTONIANS[options["hamiltonian"]](chooser_obj, unv_frames)

    options["ham_param_defn"] = QDMPy.hamiltonian._hamiltonians.get_param_defn(ham)

    _prep_fit_backends(options, ham)

    return ham


# ============================================================================


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


# ============================================================================


def fit_hamiltonian_pixels(options, data, hamiltonian):
    """
    Fit all pixels in image with chosen fit backend. We're fitting the hamiltonian
    to our previous fit result (i.e. the ODMR/PL fit result).

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    data : np array, 3D
        Normalised measurement array, shape: [sweep_list, y, x]. E.g. bnvs or freqs


    hamiltonian : `QDMPy.hamiltonian._hamiltonians.Hamiltonian`
        Model we're fitting to.

    Returns
    -------
    ham_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    """

    return fit_scipyfit.fit_hamiltonian_scipyfit(options, data, hamiltonian)

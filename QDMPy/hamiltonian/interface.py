# -*- coding: utf-8 -*-
"""
blaa
"""

# ============================================================================


# ============================================================================

import QDMPy.hamiltonian._hamiltonians

# ============================================================================


def define_hamiltonian(options):
    from QDMPy.constants import AVAILABLE_HAMILTONIANS

    ham = AVAILABLE_HAMILTONIANS(options["hamiltonian"])

    options["ham_param_defn"] = QDMPy.hamiltonian._hamiltonians.get_param_defn(ham)

    _prep_fit_backends(options, ham)


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


    fit_model : `QDMPy.hamiltonian._hamiltonians.Hamiltonian`
        Model we're fitting to.

    Returns
    -------
    ham_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    """
    # NOTE this shouldn't really be a user-facing function should it?
    # eh I guess it is ok. We want to reshape data though.
    # I.e. take bnvs and freqs as they are in nb, reshape into arrays, pass on
    # depending on fit method.
    #       -> I guess that would be held in bfield though!

    # Change naming -> change to 'fields' -> that way holds strain, elec, bfield etc.
    # OK that means a bit of a larger DOCUMENTATION REWRITE
    return fit_scipyfit.fit_hamiltonian_scipyfit(options, data, hamiltonian)

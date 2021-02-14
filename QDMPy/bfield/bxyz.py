# -*- coding: utf-8 -*-
"""
This module holds tools for calculating Bxyz from Bnv.

Functions
---------
 - ``
"""


# FIXME
# - This should be a propagation for just one bnv.
# -->> why?
def bxyz_from_bnv(bnvs):
    """
    Arguments
    ---------
    bnvs : list
        List of np arrays (2D) giving B_NV for each NV family/orientation.
        If num_peaks is odd, the bnv is given as the shift of that peak,
        and the dshifts is left as np.nans.

    Returns
    -------

    """
    if len(bnvs) == 1:
        # only use one bnv
        pass



import numpy as np


# ###############################################################
# ###############################################################
#
# ###############################################################


def add_bins(coll):

    # ---------------
    # check if needed

    wbins = coll._which_bins
    if coll.dobj.get(wbins) is not None:
        return

    # -------------------------
    # define bins from scratch
    # -------------------------

    # linear uniform 1d
    coll.add_bins('bin0', edges=np.linspace(0, 1, 10), units='m')

    # log uniform 1d
    coll.add_bins(edges=np.logspace(0, 1, 10), units='eV')

    # non-uniform 1d
    coll.add_bins(edges=np.r_[1, 2, 5, 10, 12, 20], units='s')

    # linear uniform 2d
    coll.add_bins('bin0', edges=np.linspace(0, 1, 10), units='m')

    # log uniform 2d
    coll.add_bins(edges=np.logspace(0, 1, 10), units='eV')

    # non-uniform 2d
    coll.add_bins(edges=np.r_[1, 2, 5, 10, 12, 20], units='s')

    # -------------------------
    # define bins pre-existing
    # -------------------------

    return

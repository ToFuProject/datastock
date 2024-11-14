

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

    # ---------------
    # define bins

    # linear uniform
    coll.add_bins('bin0', edges=np.linspace(0, 1, 10), units='m')

    return

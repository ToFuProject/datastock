

import numpy as np


# library-specific
from ._class import DataStock


__all__ = ['plot_as_array']


# #############################################################################
# #############################################################################
#           Plotting directly callable without using class (shortcuts)
# #############################################################################


def plot_as_array(data=None):
    """ Interactive plotting of any 2d np.ndarray """

    # ------------
    # check inputs

    try:
        data = np.asarray(data)
        assert data.ndim == 2
    except Exception as err:
        msg = (
            str(err)
            + "\nArg data must be a 2d np.ndarray!\n"
            f"Provided:\n{data}"
        )
        raise Exception(msg)

    # ---------------------
    # Instanciate datastock

    st = DataStock()
    st.add_data(data)

    return st.plot_as_array(inplace=True)

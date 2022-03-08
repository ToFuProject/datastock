

import numpy as np


# library-specific
from ._class import DataStock


__all__ = [
    'plot_as_array',
    'plot_BvsA_as_distribution',
]


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
        assert data.ndim in [1, 2, 3]
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


def plot_BvsA_as_distribution(dataA=None, dataB=None):
    """ Interactive plotting of any 2d np.ndarray """

    # ------------
    # check inputs

    try:
        dataA = np.asarray(dataA)
        dataB = np.asarray(dataB)
        assert dataA.ndim == 1
        assert dataA.shape == dataB.shape
    except Exception as err:
        msg = (
            str(err)
            + "\nArg dataA and dataB must be 1d np.ndarray of same shape!\n"
            f"Provided:\n{dataA}\n{dataB}"
        )
        raise Exception(msg)

    # ---------------------
    # Instanciate datastock

    st = DataStock()
    st.add_data(key='dataA', data=dataA)
    st.add_data(key='dataB', data=dataB)

    return st.plot_BvsA_as_distribution(
        keyA='dataA', keyB='dataB', inplace=True,
    )

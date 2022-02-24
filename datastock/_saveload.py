

import os
import getpass
import datetime as dtm


import numpy as np


from . import _generic_check
from . import _generic_utils


# #################################################################
# #################################################################
#                   Save
# #################################################################


def save(
    dflat=None,
    name=None,
    path=None,
    clsname=None,
    return_pfe=None,
    verb=None,
):
    """ Save flattened dict """

    # ------------
    # check inputs

    # path
    path = _generic_check._check_var(
        path, 'path',
        default=os.path.abspath('./'),
        types=str,
    )
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        msg = f"Arg path must be a valid path!\nProvided: {path}"
        raise Exception(msg)

    # clsname
    clsname = _generic_check._check_var(
        clsname, 'clsname',
        default='DataCollection',
        types=str,
    )

    # name
    name = _generic_check._check_var(
        name, 'name',
        default='name',
        types=str,
    )

    # verb
    verb = _generic_check._check_var(
        verb, 'verb',
        default=True,
        types=bool,
    )

    # return_pfe
    return_pfe = _generic_check._check_var(
        return_pfe, 'return_pfe',
        default=False,
        types=bool,
    )

    # ----------------------
    # store initial data type

    dtypes = {
        f'{k0}_type': v0.__class__.__name__
        for k0, v0 in dflat.items()
    }
    dflat.update(dtypes)

    # ----------------------
    # save / print / return

    user = getpass.getuser()
    dt = dtm.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f'{clsname}_{name}_{user}_{dt}.npz'

    # save
    pfe = os.path.join(path, name)
    np.savez(pfe,  **dflat)

    # print
    if verb:
        msg = f"Saved in:\n\t{pfe}"
        print(msg)

    # return
    if return_pfe is True:
        return pfe


# #################################################################
# #################################################################
#                   load
# #################################################################


def load(
    pfe=None,
    clsname=None,
    allow_pickle=None,
    sep=None,
    verb=None,
):

    # -------------
    # check inputs

    if not os.path.isfile(pfe):
        msg = f"Arg pfe must be a valid path to a file!\n\t- Provided: {pfe}"
        raise Exception(msg)

    allow_pickle = _generic_check._check_var(
        allow_pickle, 'allow_pickle',
        default=True,
        types=bool,
    )

    # verb
    verb = _generic_check._check_var(
        verb, 'verb',
        default=True,
        types=bool,
    )

    # --------------
    # load flat dict

    dflat = dict(np.load(pfe, allow_pickle=allow_pickle))

    # ----------
    # reshape

    dout = {}
    for k0, v0 in dflat.items():

        if k0.endswith('_type'):
            continue
        k0typ = f'{k0}_type'
        typ = dflat[k0typ].tolist()

        if v0.shape == ():
            dflat[k0] = v0.tolist()

        if typ == 'tuple':
            dout[k0] = tuple(dflat[k0])
        elif typ == 'list':
            dout[k0] = list(dflat[k0])
        elif typ == 'str':
            dout[k0] = str(dflat[k0])
        elif typ == 'int':
            dout[k0] = int(dflat[k0])
        elif typ == 'float':
            dout[k0] = float(dflat[k0])
        elif typ == 'bool':
            dout[k0] = bool(dflat[k0])
        elif typ == 'NoneType':
            dout[k0] = None
        elif typ == 'ndarray':
            dout[k0] = dflat[k0]
        else:
            msg = (
                f"Don't know how to deal with dflat['{k0}']: {typ}"
            )
            raise Exception(msg)

    dout = _generic_utils.reshape_dict(dout, sep=sep)

    # -----------
    # Instanciate

    from ._DataCollection_class import DataCollection

    obj = DataCollection.from_dict(dout)

    if verb:
        msg = f"Loaded from\n\t{pfe}"
        print(msg)

    return obj

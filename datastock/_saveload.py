

import os
import getpass
import datetime as dtm


import numpy as np
import astropy.units as asunits


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
    cls=None,
    allow_pickle=None,
    sep=None,
    verb=None,
):

    # -------------
    # check inputs

    # pfe
    if not os.path.isfile(pfe):
        msg = f"Arg pfe must be a valid path to a file!\n\t- Provided: {pfe}"
        raise Exception(msg)

    # cls
    if cls is None:
        from ._class import DataStock
        cls = DataStock

    if not (type(cls) is type and hasattr(cls, 'from_dict')):
        msg = (
            "Arg cls must be a class with method 'from_dict()'\n"
            f"\t- Provided: {cls}"
        )
        raise Exception(msg)

    # allow_pickle
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

        if k0.endswith('__type'):
            continue
        k0typ = f'{k0}__type'
        typ = dflat[k0typ].tolist()

        if v0.shape == ():
            dflat[k0] = v0.tolist()

        if typ == 'tuple':
            dout[k0] = tuple(dflat[k0])
        elif typ == 'list':
            dout[k0] = list(dflat[k0])
        elif typ == 'str':
            dout[k0] = str(dflat[k0])
        elif typ in ['int']:
            dout[k0] = int(dflat[k0])
        elif typ.startswith('int') and typ[3:].isnumeric():
            dout[k0] = np.array([dflat[k0]]).astype(typ)[0]
        elif typ in ['float']:
            dout[k0] = float(dflat[k0])
        elif typ.startswith('float') and typ[5:].isnumeric():
            dout[k0] = np.array([dflat[k0]]).astype(typ)[0]
        elif typ == 'bool':
            dout[k0] = bool(dflat[k0])
        elif typ == 'NoneType':
            dout[k0] = None
        elif typ == 'ndarray':
            dout[k0] = dflat[k0]
        elif 'Unit' in typ:
            dout[k0] = asunits.Unit(v0.tolist())
        elif typ == 'type':
            dout[k0] = dflat[k0]
        else:
            msg = (
                f"Don't know how to deal with dflat['{k0}']: {typ}"
            )
            raise Exception(msg)

    dout = _generic_utils.reshape_dict(dout, sep=sep)

    # -----------
    # Instanciate

    obj = cls.from_dict(dout)

    if verb:
        msg = f"Loaded from\n\t{pfe}"
        print(msg)

    return obj


# #################################################################
# #################################################################
#                   Find files
# #################################################################


def get_files(
    path=None,
    patterns=None,
    pfe=None,
    dpath=None,
):
    """ Return a dict of path keys associated to list of file names

    A pfe is a str describing the path, file name and extension

    If pfe is provided, it is just checked

    If path / patterns is provided, return all files in path matching patterns

    If dpath is provided, must be a dict with:
        - keys: valid path
        - values: dict with 'patterns' or 'pfe'

    If pattern is (or contains) tuples, the str in tuples are exclusive

    """

    # ------
    # pick

    lc = [
        pfe is not None,
        patterns is not None,
        dpath is not None,
    ]

    if np.sum(lc) != 1:
        msg = "Please provide pfe xor pattern xor case!"
        raise Exception(msg)

    # -----------
    # check path

    if path is not None:
        if not (isinstance(path, str) and os.path.isdir(path)):
            msg = f"Arg path must be a valid path name!\n{path}"
            return Exception(msg)
    else:
        path = './'
    path = os.path.abspath(path)

    # -----------
    # check pfe

    if isinstance(pfe, str):
        pfe = [pfe]

    if pfe is not None:

        err = False
        assert isinstance(pfe, (list, tuple))
        lout = [pp for pp in pfe if not os.path.isfile(pp)]

        # check that each file exists
        if len(lout) == len(pfe):
            pfe = [os.path.join(path, pp) for pp in pfe]
            lout = [pp for pp in pfe if not os.path.isfile(pp)]
            if len(lout) > 0:
                err = True
        elif len(lout) > 0:
            err = True

        # Exception
        if err is True:
            msg = f"The following files do not exist:\n{lout}"
            raise Exception(msg)

    # ---------------------------
    # check pattern

    if patterns is not None:

        if isinstance(patterns, (str, tuple)):
            patterns = [patterns]

        if not all([isinstance(pp, (str, tuple)) for pp in patterns]):
            msg = f"Arg patterns must be a list of str / tuple!\n{patterns}"
            raise Exception(msg)

        pfe = sorted([
            os.path.join(path, ff) for ff in os.listdir(path)
            if os.path.isfile(os.path.join(path, ff))
            if all([
                    p0 in ff if isinstance(p0, str)
                    else all([p1 not in ff for p1 in p0])
                    for p0 in patterns
                ])
        ])

    # ---------------------
    # format pfe into dpfe

    if dpath is None:

        lpf = [os.path.split(ff) for ff in pfe]
        lpu = sorted(set([ff[0] for ff in lpf]))

        dpfe = {
            os.path.abspath(k0): [ff[1] for ff in lpf if ff[0] == k0]
            for k0 in lpu
        }

    # ----------------
    # check case

    if dpath is not None:

        # check format
        c0 = (
            isinstance(dpath, dict)
            and all([
                isinstance(k0, str)
                and os.path.isdir(k0)
                and (
                    isinstance(v0, (str, list))
                    or (
                        isinstance(v0, dict)
                        and isinstance(v0.get('patterns', ''), (list, str, tuple))
                        and isinstance(v0.get('pfe', ''), (list, str))
                    )
                )
                for k0, v0 in dpath.items()
            ])
        )
        if not c0:
            msg = (
                "Arg dpath must be a dict with:\n"
                "\t- keys: valid paths\n"
                "\t- values: dict with 'patterns' xor 'pfe'\n"
                f"Provided:\n{dpath}"
            )
            raise Exception(msg)

        # str => patterns
        for k0, v0 in dpath.items():
            if isinstance(v0, (str, list)):
                dpath[k0] = {'patterns': v0}

        # append list of files
        dpfe = {}
        for k0, v0 in dpath.items():
            dpfe.update(get_files(
                path=k0,
                pfe=v0.get('pfe'),
                patterns=v0.get('patterns'),
                dpath=None,
            ))

    return dpfe

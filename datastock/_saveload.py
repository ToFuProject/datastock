

import os
import getpass
import datetime as dtm
import itertools as itt
import warnings


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
    dpfe=None,
    returnas=None,
    strict=None,
):
    """ Return a dict of path keys associated to list of file names, or a list

    dpfe can be:
        - str: a valid file path
        - list of str: a list of valid file paths
        - dict with:
            keys = valid path str
            values =
                - str: valid file names in the associated path
                - str: pattern to be found in the files names in that path
                - list of str: list of the above (file names or patterns)

    If pattern is (or contains) tuples, the str in tuples are exclusive

    """

    # ------------------------------------------
    # check inputs
    # ------------------------------------------

    # -----------------
    # check returnas

    returnas = _generic_check._check_var(
        returnas, 'returnas',
        allowed=[dict, list],
        default=list,
    )

    # -----------------
    # check dpfe

    lc = [
        isinstance(dpfe, (str, tuple)),
        isinstance(dpfe, list) and all([isinstance(pp, (str, tuple)) for pp in dpfe]),
        isinstance(dpfe, dict) and all([isinstance(pp, str) for pp in dpfe.keys()])
    ]

    if not any(lc):
        msg = (
            "Please provide dpfe as\n"
            "\t- str: a valid file path\n"
            "\t- dict with:\n"
            "\t\tkeys = valid path str\n"
            "\t\tvalues =\n"
            "\t\t\t- str: valid file names in the associated path\n"
            "\t\t\t- str: pattern to be found in the files names in that path\n"
            "\t\t\t- list of str: list of the above (file names or patterns)\n"
        )
        raise Exception(msg)

    # --------------------------------------------
    # sort cases
    # --------------------------------------------

    # -----------
    # str or list

    # str
    if lc[0]:
        dpfe = [dpfe]
        lc[0] = False
        lc[1] = True

    # list of str
    if lc[1]:
        # call check on list of files
        lpfe = _get_files_from_path(
            lpfe=dpfe,
            path=None,
            strict=strict,
        )
        dpfe = None

    # -----------
    # dict

    if lc[2]:

        dout = {}
        for k0, v0 in dpfe.items():

            # back-compatibility
            if isinstance(v0, dict):
                if v0.get('patterns') is not None:
                    v0 = v0['patterns']
                elif v0.get('pfe') is not None:
                    v0 = v0['pfe']
                else:
                    msg = ()

            # call pfe / pattern recognition
            lpfe = _get_files_from_path(
                lpfe=v0,
                path=k0,
                strict=strict,
            )
            dout[k0] = [os.path.split(pfe)[1] for pfe in lpfe]

        lpfe = None
        dpfe = dout

    # -------------------------------------------------
    # returnas
    # -------------------------------------------------

    if returnas is list:
        if lpfe is None:
            lpfe = list(itt.chain.from_iterable([
                [os.path.join(k0, v1) for v1 in v0]
                for k0, v0 in dpfe.items()
            ]))
        out = lpfe

    else:
        if dpfe is None:
            lpath = {pfe.split()[0] for pfe in lpfe}
            dpfe = {
                path: [pfe for pfe in lpfe if path == pfe.split()[0]]
                for path in lpath
            }
        out = dpfe

    return out


def _get_files_from_path(
    lpfe=None,
    path=None,
    strict=None,
):

    # -----------------
    # preliminary check

    if isinstance(lpfe, (str, tuple)):
        lpfe = [lpfe]

    if not all([isinstance(pfe, (str, tuple)) for pfe in lpfe]):
        msg = (
            "Please provide a list of str (pfe or patterns)!\n"
            f"\t- Provided: {lpfe}"
        )
        raise Exception(msg)

    # path
    if path is None:
        path = os.path.abspath('.')
    if not os.path.isdir(path):
        msg = f"Provided path is not valid!\n\t- path: {path}"
        raise Exception(msg)

    # strict
    strict = _generic_check._check_var(
        strict, 'strict',
        types=bool,
        default=True,
    )

    # ---------------
    # pfe vs patterns

    lc = [
        any([os.path.isfile(pfe) for pfe in lpfe if isinstance(pfe, str)]),
        any([os.path.isfile(os.path.join(path, pfe)) for pfe in lpfe if isinstance(pfe, str)]),
    ]

    # ---------------------
    # valid pfe

    if lc[0] or lc[1]:

        if lc[0]:
            out = [pfe for pfe in lpfe if os.path.isfile(pfe)]
            lfail = [pfe for pfe in lpfe if not os.path.isfile(pfe)]

        else:
            out = [
                os.path.join(path, pfe) for pfe in lpfe
                if os.path.isfile(os.path.join(path, pfe))
            ]
            lfail = [
                os.path.join(path, pfe) for pfe in lpfe
                if not os.path.isfile(os.path.join(path, pfe))
            ]

        if len(lfail) > 0:
            msg = (
                "The following files do not exist:\n"
                + "\n".join([f"\t- {k0}" for k0 in lfail])
            )
            if strict is True:
                raise Exception(msg)
            else:
                warnings.warn(msg)

    # ---------------------
    # patterns

    else:

        if not all([isinstance(pp, (str, tuple)) for pp in lpfe]):
            msg = f"Arg patterns must be a list of str / tuple!\n{lpfe}"
            raise Exception(msg)

        out = sorted([
            os.path.join(path, ff) for ff in os.listdir(path)
            if os.path.isfile(os.path.join(path, ff))
            if all([
                    p0 in ff if isinstance(p0, str)
                    else all([p1 not in ff for p1 in p0])
                    for p0 in lpfe
                ])
        ])

        # safety check
        if len(out) == 0:
            msg = (
                "The following list of files is empty:\n"
                f"\t- lpfe = {lpfe}\n"
                f"\t- path: {path}"
            )
            if strict is True:
                raise Exception(msg)
            else:
                warnings.warn(msg)

    return out
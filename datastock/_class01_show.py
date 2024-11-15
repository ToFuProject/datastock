# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:16:42 2024

@author: dvezinet
"""


import numpy as np


from . import _generic_utils
from . import _generic_check


#############################################
#############################################
#       Main
#############################################


def main(
    coll=None,
    # options
    show_which=None,
    show=None,
    # print parameters
    sep=None,
    line=None,
    justify=None,
    table_sep=None,
    # bool options
    verb=True,
    returnas=False,
):

    # -------------
    # check inputs
    # -------------

    show_which, show = _check_inputs(
        coll=coll,
        show_which=show_which,
        show=show,
    )

    # intialize
    lcol, lar = [], []

    # -----------------------
    # Build for dref
    # -----------------------

    if 'ref' in show_which and len(coll._dref) > 0:
        lcol, lar = _show_ref(coll, lcol=lcol, lar=lar, show=show)

    # -----------------------
    # Build for ddata
    # -----------------------

    if 'data' in show_which and len(coll._ddata) > 0:
        lcol, lar = _show_data(coll, lcol=lcol, lar=lar, show=show)

    # -----------------------
    # Build for dobj
    # -----------------------

    anyobj = (
        len(coll._dobj) > 0
        and any([
            ss in show_which
            for ss in ['obj'] + list(coll._dobj.keys())
        ])
    )
    if anyobj:
        for k0 in coll._dobj.keys():
            if 'obj' in show_which or k0 in show_which:
                func = coll._get_show_obj(k0)
                lcol, lar = func(
                    coll=coll,
                    which=k0,
                    lcol=lcol,
                    lar=lar,
                    show=show,
                )

    return _generic_utils.pretty_print(
        headers=lcol,
        content=lar,
        sep=sep,
        line=line,
        table_sep=table_sep,
        verb=verb,
        returnas=returnas,
    )


###########################################################
###########################################################
#       check
###########################################################


def _check_inputs(
    coll=None,
    show_which=None,
    show=None,
):

    # -------------
    # show_which
    # -------------

    if show_which is None:
        show_which = ['ref', 'data', 'obj']

    if isinstance(show_which, str):
        show_which = [show_which]

    lok = ['ref', 'data'] + list(coll._dobj.keys())
    show_which = _generic_check._check_var_iter(
        show_which, 'show_which',
        types=(list, tuple),
        types_iter=str,
        allowed=lok + ['obj'],
    )

    # tuple => exclusion
    if isinstance(show_which, tuple):

        if 'obj' in show_which:
            show_which = [
                k0 for k0 in ['ref', 'data'] if k0 not in show_which
            ]

        else:
            show_which = [
                k0 for k0 in lok
                if k0 not in show_which
            ]
    else:
        if 'obj' in show_which:
            show_which = (
                [k0 for k0 in ['ref', 'data'] if k0 in show_which]
                + list(coll._dobj.keys())
            )

    # -------------
    # show
    # -------------

    if len(show_which) == 1:

        if isinstance(show, str):
            show = [show]

        if show_which[0] == 'ref':
            lok = list(coll.dref.keys())
        elif show_which[0] == 'data':
            lok = list(coll.ddata.keys())
        else:
            lok = list(coll.dobj.get(show_which[0], {}).keys())

        show = _generic_check._check_var_iter(
            show, 'show',
            types=(list, tuple),
            types_iter=str,
            allowed=lok,
        )

    else:
        show = None

    return show_which, show


###########################################################
###########################################################
#       specific show
###########################################################


def _show_ref(coll=None, lcol=None, lar=None, show=None):

    # ----------------
    # column names
    # ----------------

    lcol.append(['ref key', 'size', 'nb. data', 'nb. data monot.'])

    # ---------------
    # prepare array
    # ---------------

    lk0 = [
        k0 for k0 in coll._dref.keys()
        if show is None or k0 in show
    ]

    lar.append([
        [
            k0,
            str(coll._dref[k0]['size']),
            str(len(coll._dref[k0]['ldata'])),
            str(len(coll._dref[k0]['ldata_monot'])),
        ]
        for k0 in lk0
    ])

    # ---------------------------
    # indices (for interactivity)
    # ---------------------------

    lp = coll.get_lparam(which='ref')
    if 'indices' in lp:
        lcol[0].append('indices')
        for ii, (k0, v0) in enumerate(coll._dref.items()):
            if coll._dref[k0]['indices'] is None:
                lar[0][ii].append(str(v0['indices']))
            else:
                lar[0][ii].append(str(list(v0['indices'])))

    # ---------------------------
    # group (for interactivity)
    # ---------------------------

    if 'group' in lp:
        lcol[0].append('group')
        for ii, (k0, v0) in enumerate(coll._dref.items()):
            lar[0][ii].append(str(coll._dref[k0]['group']))

    # ---------------------------
    # inc (for interactivity)
    # ---------------------------

    if 'inc' in lp:
        lcol[0].append('increment')
        for ii, (k0, v0) in enumerate(coll._dref.items()):
            lar[0][ii].append(str(coll._dref[k0]['inc']))

    return lcol, lar


def _show_data(coll=None, lcol=None, lar=None, show=None):

    # ----------------
    # parameters
    # ----------------

    lk = _show_get_fields(
        which='data',
        lparam=coll.get_lparam(which='data', for_show=True),
        dshow=coll._dshow,
    )

    # ---------------------------
    # column names
    # ---------------------------

    lcol.append(['data'] + [pp.split('.')[-1] for pp in lk])

    # ---------------------------
    # data
    # ---------------------------

    lk0 = [
        k0 for k0 in coll._ddata.keys()
        if show is None or k0 in show
    ]

    lar.append([
        [k0] + _show_extract(dobj=coll._ddata[k0], lk=lk)
        for k0 in lk0
    ])

    return lcol, lar


def _show_obj_def(coll=None, which=None, lcol=None, lar=None, show=None):

    # ----------------
    # parameters
    # ----------------

    lk = _show_get_fields(
        which=which,
        lparam=coll.get_lparam(which=which, for_show=True),
        dshow=coll._dshow,
    )

    # ---------------------------
    # column names
    # ---------------------------

    lcol.append([which] + [pp.split('.')[-1] for pp in lk])

    # ---------------------------
    # data
    # ---------------------------

    lkey = [
        k1 for k1 in coll._dobj.get(which, {}).keys()
        if show is None or k1 in show
    ]

    lar.append([
        [k1] + _show_extract(dobj=coll.dobj[which][k1], lk=lk)
        for k1 in lkey
    ])

    return lcol, lar


###########################################################
###########################################################
#       Utilities
###########################################################


def _get_lparam_show_append(which, key, val, lparam, for_show):

    c0 = (
        callable(val)
        or 'class' in key
        or 'handle' in key
        or (which == 'axes' and key == 'bck')
        or isinstance(val, dict)
    )
    if key not in lparam and ((not for_show) or (for_show and not c0)):
        lparam.append(key)


def _get_lparam(which=None, dd=None, for_show=None):

    if for_show:
        lparam = []
        for k0, v0 in dd.items():
            for k1, v1 in v0.items():
                if isinstance(v1, dict):
                    for k2, v2 in v1.items():
                        k3 = f'{k1}.{k2}'
                        _get_lparam_show_append(
                            which, k3, v2, lparam, for_show,
                        )
                else:
                    _get_lparam_show_append(
                        which, k1, v1, lparam, for_show,
                    )

    else:
        lparam = list(list(dd.values())[0].keys())

    return lparam


def _show_get_fields(which=None, lparam=None, dshow=None):

    # show dict
    if which not in dshow.keys():
        lk = lparam

    else:
        lk = dshow[which]

        if isinstance(lk, list):
            lk = [
                kk for kk in dshow[which]
                if kk in lparam
            ]
        elif isinstance(lk, tuple):
            lk = [
                kk for kk in lparam
                if kk not in dshow[which]
            ]
        else:
            msg = f"Unreckognized dshow['{which}']"
            raise Exception(msg)

    return lk


def _show_extract(dobj=None, lk=None):

    lv0 = []
    for k0 in lk:

        lk0 = k0.split('.')
        for ii in range(len(lk0)):
            if ii == 0:
                v0 = dobj[lk0[ii]]
            elif v0 is not None:
                v0 = v0[lk0[ii]]

        # formatting
        if isinstance(v0, float):
            lv0.append(f'{v0:.2e}')

        elif isinstance(v0, np.ndarray) and v0.size == 3:
            if v0.dtype == float:
                lv0.append(
                    np.array2string(
                        v0,
                        formatter={'float': lambda x: f'{x:.3e}'},
                    ),
                )
            else:
                lv0.append(str(v0))
        else:
            lv0.append(str(v0))

    return lv0


#############################################
#############################################
#       Main - details
#############################################


def main_details(
    coll=None,
    # options
    which=None,
    key=None,
    # print parameters
    sep=None,
    line=None,
    justify=None,
    table_sep=None,
    # bool options
    verb=True,
    returnas=False,
):

    # -------------
    # check inputs
    # -------------

    which, key = _check_details(
        coll=coll,
        which=which,
        key=key,
    )

    # intialize
    lcol, lar = [], []

    # -----------------------
    # Build for dobj
    # -----------------------

    func = coll._get_show_details(which=which)
    lcol, lar = func(
        coll=coll,
        key=key,
        lcol=lcol,
        lar=lar,
    )

    return _generic_utils.pretty_print(
        headers=lcol,
        content=lar,
        sep=sep,
        line=line,
        table_sep=table_sep,
        verb=verb,
        returnas=returnas,
    )


###########################################################
###########################################################
#       check details
###########################################################


def _check_details(
    coll=None,
    which=None,
    key=None,
):

    # -------------
    # key
    # -------------

    lok_which = list(coll._dobj.keys())
    dkey = {}
    for kw in lok_which:
        dkey.update({
            k1: kw for k1 in coll.dobj[kw].keys()
        })

    # check
    try:
        key = _generic_check._check_var(
            key, 'key',
            types=str,
            allowed=list(dkey.keys()),
        )

        which = dkey[key]

    except Exception:

        # -------------
        # show_which

        lok = list(coll._dobj.keys())
        which = _generic_check._check_var(
            which, 'which',
            types=str,
            allowed=lok,
        )

        lok = [k1 for k1, kw in dkey.items() if kw == which]
        key = _generic_check._check_var(
            key, 'key',
            types=str,
            allowed=lok,
        )

    return which, key

# -*- coding: utf-8 -*-


import numpy as np


# Common
from . import _generic_check


# #############################################################################
# #############################################################################
#                          bins generic check
# #############################################################################


def check(
    coll=None,
    key=None,
    edges=None,
    # custom names
    key_edges=None,
    key_cents=None,
    key_ref=None,
    # additional attributes
    **kwdargs,
):

    # -------------
    # key
    # -------------

    key = _generic_check._obj_key(
        d0=coll._dobj.get(coll._which_bins, {}),
        short='b',
        key=key,
    )

    # ------------
    # edges
    # ------------

    # -----------------------
    # first conformity check

    lc = [
        _check_edges_str(edges, coll),
        _check_edges_array(edges),
        isinstance(edges, tuple)
        and len(edges) in (1, 2)
        and all([
            _check_edges_str(ee, coll) or _check_edges_array(ee)
            for ee in edges
        ])
    ]

    if np.sum(lc) != 1:
        msg = (
            f"For Bins '{key}', arg edges must be:\n"
            "\t- a str pointing to a n existing monotonous vector\n"
            "\t- an array/list/tuple of unique increasing values\n"
            "\t- a tuple of 1 or 2 of the above\n"
            "Provided:\n\t{edges}"
        )
        raise Exception(msg)

    if lc[0] or lc[1]:
        edges = (edges,)

    # ----------------------------
    # make tuple of 1d flat arrays

    edges_new = []
    for ii, ee in enumerate(edges):
        if isinstance(ee, str):
            edges_new.append(ee)
        else:
            edges_new.append(_generic_check._check_flat1darray(
                ee, f'edges[{ii}]',
                dtype=float,
                unique=True,
                can_be_None=False,
            ))

    edges = edges_new
    nd = f"{len(edges)}d"

    # -----------------
    # kwdargs
    # -----------------

    for k0, v0 in kwdargs.items():
        if isinstance(v0, str) or v0 is None:
            if nd == 1:
                kwdargs[k0] = (v0,)
            else:
                kwdargs[k0] = (v0, v0)

        c0 = (
            isinstance(kwdargs[k0], tuple)
            and len(kwdargs[k0]) == nd
            and all([isinstance(vv, str) or vv is None for vv in kwdargs[k0]])
        )
        if not c0:
            msg = (
                f"Bins '{key}', arg kwdargs must be dict of data attributes\n"
                "Where each attribute is provided as a tuple of "
                "len() = len(edges)\n"
                f"Provided:\n\t{kwdargs}"
            )
            raise Exception(msg)

    # -----------------
    # other keys
    # -----------------

    # -----------------
    # key_ref

    dref = {}
    ddata = {}
    cents = [None for ii in edges]
    for ii, ee in enumerate(edges):

        edges[ii], cents[ii] = _to_dict(
            coll=coll,
            key=key,
            ii=ii,
            edge=ee,
            # custom names
            key_cents=key_cents,
            key_ref=key_ref,
            # dict
            dref=dref,
            ddata=ddata,
            # attributes
            **{kk: vv[ii] for kk, vv in kwdargs.items()},
        )

    # --------------
    # dobj
    # --------------

    # dobj
    dobj = {
        coll._which_bins: {
            key: {
                'nd': '1d',
                'edges': tuple(edges),
                'cents': (key_cents,),
                'ref': (key_ref,),
                # 'shape': (nb,),
            },
        },
    }

    return key, dref, ddata, dobj


def _check_edges_str(edges, coll):
    return (
        isinstance(edges, str)
        and edges in coll.ddata.keys()
        and coll.ddata[edges]['monot'] == (True,)
    )


def _check_edges_array(edges):
    return (
    )


# ##############################################################
# ###############################################################
#                           to_dict
# ###############################################################


# TBF
def _to_dict(
    coll=None,
    key=None,
    ii=None,
    ee=None,
    # dict
    dref=None,
    ddata=None,
    # custom names
    key_edge=None,
    key_cents=None,
    key_ref=None,
    # additional attributes
    **kwdargs,
):

    # attributes
    latt = ['dim', 'quant', 'name', 'units']
    dim, quant, name, units = [kwdargs.get(ss) for ss in latt]

    # -------------
    # prepare dict

    # ref
    if isinstance(ee, str):
        pass
    else:
        defk = f"{key}_ne{ii}"
        lout = [k0 for k0, v0 in coll.dref.items()]
        key_ref = _generic_check._check_var(
            key_ref[ii], defk,
            types=str,
            default=defk,
            excluded=lout,
        )
        dref[key_ref] = {'size': ee.size}

        #
        defk = f"{key}_e{ii}"
        key_edge = _generic_check._check_var(
            key_edge, defk,
            types=str,
            default=defk,
            excluded=lout,
        )
        ddata[key_edge] = {
            'data': ee,
            'ref': key_ref,
            **kwdargs,
        }

    defk = f"{key}_nc{ii}"
    lout = [k0 for k0, v0 in coll.dref.items()]
    key_ref = _generic_check._check_var(
        key_ref, defk,
        types=str,
        default=defk,
        excluded=lout,
    )
    dref[key_ref] = {'size': ee.size - 1}

    # dref
    if key_ref not in coll.dref.keys():
        dref = {
            key_ref: {
                'size': ee.size,
            },
        }
    else:
        dref = None

    # ddata
    key_cent = None
    if key_cents not in coll.ddata.keys():
        ddata = {
            key_cents: {
                # 'data': cents,
                'units': units,
                # 'source': None,
                'dim': dim,
                'quant': quant,
                'name': name,
                'ref': key_ref,
            },
        }
    else:
        ddata = None

    return key_edge, key_cent


# ##############################################################
# ###############################################################
#                   remove bins
# ###############################################################


def remove_bins(coll=None, key=None, propagate=None):

    # ----------
    # check

    # key
    wbins = coll._which_bins
    if wbins not in coll.dobj.keys():
        return

    if isinstance(key, str):
        key = [key]
    key = _generic_check._check_var_iter(
        key, 'key',
        types=(list, tuple),
        types_iter=str,
        allowed=coll.dobj.get(wbins, {}).keys(),
    )

    # propagate
    propagate = _generic_check._check_var(
        propagate, 'propagate',
        types=bool,
        default=True,
    )

    # ---------
    # remove

    for k0 in key:

        # specific data
        kdata = list(coll.dobj[wbins][k0]['cents'])
        coll.remove_data(kdata, propagate=propagate)

        # specific ref
        lref = list(coll.dobj[wbins][k0]['ref'])
        for rr in lref:
            if rr in coll.dref.keys():
                coll.remove_ref(rr, propagate=propagate)

        # obj
        coll.remove_obj(which=wbins, key=k0, propagate=propagate)

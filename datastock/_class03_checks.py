# -*- coding: utf-8 -*-


# Common
import numpy as np
import datastock as ds


# #############################################################################
# #############################################################################
#                          bins generic check
# #############################################################################


def check(
    coll=None,
    key=None,
    edges=None,
    # custom names
    key_cents=None,
    key_ref=None,
    # additional attributes
    **kwdargs,
):

    # --------
    # keys

    # key
    key = ds._generic_check._obj_key(
        d0=coll._dobj.get(coll._which_bins, {}),
        short='b',
        key=key,
    )

    # ------------
    # edges

    edges = ds._generic_check._check_flat1darray(
        edges, 'edges',
        dtype=float,
        unique=True,
        can_be_None=False,
    )

    nb = edges.size - 1
    cents = 0.5*(edges[:-1] + edges[1:])

    # --------------------
    # safety check on keys

    # key_ref
    defk = f"{key}_nb"
    lout = [k0 for k0, v0 in coll.dref.items() if v0['size'] != nb]
    key_ref = ds._generic_check._check_var(
        key_ref, 'key_ref',
        types=str,
        default=defk,
        excluded=lout,
    )

    # key_cents
    defk = f"{key}_c"
    lout = [
        k0 for k0, v0 in coll.ddata.items()
        if not (
            v0['shape'] == (nb,)
            and key_ref in coll.dref.keys()
            and v0['ref'] == (key_ref,)
            and v0['monot'] == (True,)
        )
    ]
    key_cents = ds._generic_check._check_var(
        key_cents, 'key_cents',
        types=str,
        default=defk,
        excluded=lout,
    )

    # --------------
    # to dict

    dref, ddata, dobj = _to_dict(
        coll=coll,
        key=key,
        edges=edges,
        nb=nb,
        cents=cents,
        # custom names
        key_cents=key_cents,
        key_ref=key_ref,
        # attributes
        **kwdargs,
    )

    return key, dref, ddata, dobj


# ##############################################################
# ###############################################################
#                           to_dict
# ###############################################################


def _to_dict(
    coll=None,
    key=None,
    edges=None,
    nb=None,
    cents=None,
    # custom names
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

    # dref
    if key_ref not in coll.dref.keys():
        dref = {
            key_ref: {
                'size': nb,
            },
        }
    else:
        dref = None

    # ddata
    if key_cents not in coll.ddata.keys():
        ddata = {
            key_cents: {
                'data': cents,
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

    # dobj
    dobj = {
        coll._which_bins: {
            key: {
                'nd': '1d',
                'edges': edges,
                'cents': (key_cents,),
                'ref': (key_ref,),
                'shape': (nb,),
            },
        },
    }

    # additional attributes
    for k0, v0 in kwdargs.items():
        if k0 not in latt:
            dobj[coll._which_bins][key][k0] = v0

    return dref, ddata, dobj


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
    key = ds._generic_check._check_var_iter(
        key, 'key',
        types=(list, tuple),
        types_iter=str,
        allowed=coll.dobj.get(wbins, {}).keys(),
    )

    # propagate
    propagate = ds._generic_check._check_var(
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

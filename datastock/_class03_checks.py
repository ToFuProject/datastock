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
    key_ref_edges=None,
    key_ref_cents=None,
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

    edges_new = [None for ee in edges]
    for ii, ee in enumerate(edges):
        if isinstance(ee, str):
            edges_new[ii] = ee
        else:
            edges_new[ii] = _generic_check._check_flat1darray(
                ee, f'edges[{ii}]',
                dtype=float,
                unique=True,
                can_be_None=False,
            )

    # ---------------------
    # safety check for NaNs

    for ii, ee in enumerate(edges_new):
        if isinstance(ee, str):
            ee = coll.ddata[ee]['data']

        isnan = np.any(np.isnan(ee))
        if isnan:
            msg = (
                f"Bins '{key}', provided edges have NaNs!\n"
                f"\t- edges[{ii}]: {ee}"
            )
            raise Exception(msg)

    # --------------
    # wrap up

    edges = edges_new
    nd = f"{len(edges)}d"

    # -----------------
    # kwdargs
    # -----------------

    for k0, v0 in kwdargs.items():
        if isinstance(v0, str) or v0 is None:
            if nd == '1d':
                kwdargs[k0] = (v0,)
            else:
                kwdargs[k0] = (v0, v0)

        c0 = (
            isinstance(kwdargs[k0], tuple)
            and len(kwdargs[k0]) == len(edges)
            and all([isinstance(vv, str) or vv is None for vv in kwdargs[k0]])
        )
        if not c0:
            msg = (
                f"Bins '{key}', arg kwdargs must be dict of data attributes\n"
                "Where each attribute is provided as a tuple of "
                f"len() = len(edges) = ({len(edges)})\n"
                f"Provided:\n\t{kwdargs}"
            )
            raise Exception(msg)

    # -----------------
    # other keys
    # -----------------

    key_edges = _check_keys_ref(key_edges, edges, key, 'key_edges')
    key_cents = _check_keys_ref(key_cents, edges, key, 'key_cents')
    key_ref_edges = _check_keys_ref(key_ref_edges, edges, key, 'key_ref_edges')
    key_ref_cents = _check_keys_ref(key_ref_cents, edges, key, 'key_ref_cents')

    # -----------------
    # edges, cents
    # -----------------

    # -----------------
    # key_ref

    dref = {}
    ddata = {}
    shape_edges = [None for ee in edges]
    is_linear = [None for ee in edges]
    is_log = [None for ee in edges]
    units = [None for ee in edges]
    for ii, ee in enumerate(edges):
        (
            key_edges[ii], key_cents[ii],
            key_ref_edges[ii], key_ref_cents[ii],
            shape_edges[ii],
            is_linear[ii], is_log[ii],
            units[ii],
        ) = _to_dict(
            coll=coll,
            key=key,
            ii=ii,
            ee=ee,
            # custom names
            key_edge=key_edges[ii],
            key_cent=key_cents[ii],
            key_ref_edge=key_ref_edges[ii],
            key_ref_cent=key_ref_cents[ii],
            # dict
            dref=dref,
            ddata=ddata,
            # attributes
            **{kk: vv[ii] for kk, vv in kwdargs.items()},
        )

    # -------------
    # ref and shape

    shape_cents = tuple([ss - 1 for ss in shape_edges])

    # --------------
    # dobj
    # --------------

    # dobj
    dobj = {
        coll._which_bins: {
            key: {
                'nd': nd,
                'edges': tuple(key_edges),
                'cents': tuple(key_cents),
                'ref_edges': tuple(key_ref_edges),
                'ref_cents': tuple(key_ref_cents),
                'shape_edges': tuple(shape_edges),
                'shape_cents': tuple(shape_cents),
                'units': tuple(units),
                'is_linear': tuple(is_linear),
                'is_log': tuple(is_log),
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
        isinstance(edges, (list, tuple, np.ndarray))
        and np.array(edges).ndim == 1
        and np.array(edges).size > 1
    )


def _check_keys_ref(keys, edges, key, keys_name):
    if keys is None:
        keys = [None for ee in edges]
    elif isinstance(keys, str):
        keys = [keys for ee in edges]
    elif isinstance(keys, (list, tuple)):
        c0 = (
            len(keys) == len(edges)
            and all([isinstance(ss, str) or ss is None for ss in keys])
        )
        if not c0:
            msg = (
                f"Bins '{key}', arg '{keys_name}' should be either:\n"
                "\t- None (automatically set)\n"
                "\t- str to existing key\n"
                "\t- tuple of the above of len() = {len(edges)}\n"
                "Provided:\n\t{keys}"
            )
            raise Exception(msg)
    return keys


# ##############################################################
# ###############################################################
#                           to_dict
# ###############################################################


def _to_dict(
    coll=None,
    key=None,
    ii=None,
    ee=None,
    # custom names
    key_edge=None,
    key_cent=None,
    key_ref_edge=None,
    key_ref_cent=None,
    # dict
    dref=None,
    ddata=None,
    # additional attributes
    **kwdargs,
):
    """ check key_edge, key_cents, key_ref_edge, key_ref_cent

    If new, append to dref and ddata
    """

    # -------------
    # attributes
    # -------------

    latt = ['dim', 'quant', 'name', 'units']
    dim, quant, name, units = [kwdargs.get(ss) for ss in latt]

    # -------------
    # edges
    # -------------

    # ref
    if isinstance(ee, str):
        key_edge = ee
        ee = coll.ddata[key_edge]['data']
        units = coll.ddata[key_edge]['units']

    else:

        # ------------------
        # key_ref_edge

        defk = f"{key}_ne{ii}"
        lout = [k0 for k0, v0 in coll.dref.items()]
        key_ref_edge = _generic_check._check_var(
            key_ref_edge, defk,
            types=str,
            default=defk,
        )
        if key_ref_edge in lout:
            size = coll.dref[key_ref_edge]['size']
            c0 = size == ee.size
            if not c0:
                msg = (
                    f"Bins '{key}', arg key_ref_edges[{ii}]"
                    " conflicts with existing ref:\n"
                    f"\t- coll.dref['{key_ref_edge}']['size'] = {size}"
                    f"\t- edges['{ii}'].size = {ee.size}\n"
                )
                raise Exception(msg)
        else:
            dref[key_ref_edge] = {'size': ee.size}

        # ---------------
        # key_edge

        defk = f"{key}_e{ii}"
        lout = [k0 for k0, v0 in coll.ddata.items()]
        key_edge = _generic_check._check_var(
            key_edge, defk,
            types=str,
            default=defk,
            excluded=lout,
        )
        ddata[key_edge] = {
            'data': ee,
            'ref': key_ref_edge,
            **kwdargs,
        }

        units = kwdargs.get('units')

    # shape
    shape_edge = ee.size

    # ------------------
    # is_linear, is_log
    # ------------------

    is_log = (
        np.all(ee > 0.)
        and np.allclose(ee[1:] / ee[:-1], ee[1]/ee[0], atol=0, rtol=1e-6)
    )

    is_linear = np.allclose(np.diff(ee), ee[1] - ee[0], atol=0, rtol=1e-6)
    assert not (is_log and is_linear), ee

    # ------------
    # cents
    # ------------

    # ------------
    # key_ref_cent

    defk = f"{key}_nc{ii}"
    lout = [k0 for k0, v0 in coll.dref.items()]
    key_ref_cent = _generic_check._check_var(
        key_ref_cent, defk,
        types=str,
        default=defk,
    )
    if key_ref_cent in lout:
        size = coll.dref[key_ref_cent]['size']
        c0 = size == (ee.size - 1)
        if not c0:
            msg = (
                f"Bins '{key}', arg key_ref_cents[{ii}]"
                " conflicts with existing ref:\n"
                f"\t- coll.dref['{key_ref_edge}']['size'] = {size}"
                f"\t- edges['{ii}'].size - 1 = {ee.size-1}\n"
            )
            raise Exception(msg)
    else:
        dref[key_ref_cent] = {'size': ee.size - 1}

    # ------------
    # key_cent

    defk = f"{key}_c{ii}"
    lout = [k0 for k0, v0 in coll.ddata.items()]
    key_cent = _generic_check._check_var(
        key_cent, defk,
        types=str,
        default=defk,
    )
    if key_cent in lout:
        ref = coll.ddata[key_cent]['ref']
        c0 = ref == (key_ref_cent,)
        if not c0:
            msg = (
                f"Bins '{key}', arg key_ref_cents[{ii}]"
                " conflicts with existing ref:\n"
                f"\t- coll.ddata['{key_ref_cent}']['ref'] = {ref}"
                f"\t- key_ref_cent = {key_ref_cent}\n"
            )
            raise Exception(msg)

    else:
        if is_log:
            cents = np.sqrt(ee[:-1] * ee[1:])
        else:
            cents = 0.5 * (ee[1:] + ee[:-1])

        ddata[key_cent] = {
            'data': cents,
            'ref': (key_ref_cent,),
            **kwdargs,
        }

    return (
        key_edge, key_cent,
        key_ref_edge, key_ref_cent,
        shape_edge,
        is_linear, is_log,
        units,
    )


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
        kdata = (
            coll.dobj[wbins][k0]['cents']
            + coll.dobj[wbins][k0]['edges']
        )
        coll.remove_data(kdata, propagate=propagate)

        # specific ref
        lref = (
            coll.dobj[wbins][k0]['ref_cents']
            + coll.dobj[wbins][k0]['ref_edges']
        )
        for rr in lref:
            if rr in coll.dref.keys():
                coll.remove_ref(rr, propagate=propagate)

        # obj
        coll.remove_obj(which=wbins, key=k0, propagate=propagate)

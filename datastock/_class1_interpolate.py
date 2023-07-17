# -*- coding: utf-8 -*-
""" Module holding interpolation routines """


# Builtin
import warnings
import itertools as itt


# Common
import numpy as np
import scipy.interpolate as scpinterp


# local
from . import _generic_check


# ##################################################################
# ##################################################################
#               Main routine
# ##################################################################


def interpolate(
    coll=None,
    # interpolation base
    keys=None,
    ref_key=None,
    # interpolation pts
    x0=None,
    x1=None,
    grid=None,
    # domain limitation
    domain=None,
    # common ref
    ref_com=None,
    ref_vector_strategy=None,
    # parameters
    deg=None,
    deriv=None,
    log_log=None,
    nan0=None,
    # store vs return
    return_params=None,
    returnas=None,
    store=None,
    store_keys=None,
    inplace=None,
    # debug or unit tests
    debug=None,
):
    """ Interpolate at desired points on desired data

    Interpolate quantities (keys) on coordinates (ref_keys)
    All provided keys should share the same refs
    They should have dimension 2 or less

    ref_keys should be a list of monotonous data keys to be used as coordinates
    It should have one element per dimension in refs

    The interpolation points are provided as np.ndarrays in each dimension
    They should all have the same shape except if grid = True

    deg is the degree of the interpolation

    It is an interpolation, not a smoothing, so it will pass through all points
    Uses scpinterp.InterpolatedUnivariateSpline() for 1d

    """

    # -------------
    # check inputs

    # keys
    keys, ref_key, daxis, dunits, _ = _check_keys(
        coll=coll,
        keys=keys,
        ref_key=ref_key,
        only1d=False,
        ref_vector_strategy=ref_vector_strategy,
    )

    # params
    (
        deg, deriv,
        kx0, kx1, x0, x1, refx, dref_com,
        ddata, dout, dsh_other, sli_c, sli_x, sli_v,
        log_log, nan0, grid, ndim, xunique,
        returnas, return_params, store, inplace,
    ) = _check(
        coll=coll,
        # interpolation base
        keys=keys,
        ref_key=ref_key,      # ddata keys
        # interpolation pts
        x0=x0,
        x1=x1,
        # useful for shapes
        daxis=daxis,
        dunits=dunits,
        domain=domain,
        # common ref
        ref_com=ref_com,
        # parameters
        grid=grid,
        deg=deg,
        deriv=deriv,
        log_log=log_log,
        nan0=nan0,
        # return vs store
        return_params=return_params,
        returnas=returnas,
        store=store,
        inplace=inplace,
    )

    # -----------
    # interpolate

    _interp(
        coll=coll,
        keys=keys,
        ref_key=ref_key,
        x0=x0,
        x1=x1,
        ndim=ndim,
        log_log=log_log,
        dout=dout,
        ddata=ddata,
        dsh_other=dsh_other,
        daxis=daxis,
        deg=deg,
        deriv=deriv,
        dref_com=dref_com,
        sli_x=sli_x,
        sli_c=sli_c,
        sli_v=sli_v,
        nan0=nan0,
    )

    # ------------------------------
    # adjust data and ref if xunique

    if xunique:
        _xunique(dout)

    # --------
    # store

    if store is True:
        coll2 = _store(
            coll=coll,
            dout=dout,
            inplace=inplace,
            store_keys=store_keys,
        )

    # -------
    # return

    if returnas is object:
        return coll2
    elif returnas is dict:
        if return_params is True:
            dparam = {
                'keys': keys,
                'ref_key': ref_key,
                'deg': deg,
                'deriv': deriv,
                'log_log': log_log,
                'kx0': kx0,
                'kx1': kx1,
                'x0': x0,
                'x1': x1,
                'grid': grid,
                'refx': refx,
                'dref_com': dref_com,
                'daxis': daxis,
                'dsh_other': dsh_other,
                'domain': domain,
            }
            return dout, dparam

        return dout


# #############################################################################
# #############################################################################
#               Main checking routine
# #############################################################################


def _check_keys(
    coll=None,
    keys=None,
    ref_key=None,
    only1d=None,
    ref_vector_strategy=None,
):

    # only1d
    only1d = _generic_check._check_var(
        only1d, 'only1d',
        types=bool,
        default=True,
    )

    maxd = 1 if only1d else 2

    # ---------------
    # keys vs ref_key

    # ref_key
    if ref_key is not None:

        # basic checks
        if isinstance(ref_key, str):
            ref_key = (ref_key,)

        lref = list(coll.dref.keys())
        ldata = list(coll.ddata.keys())

        ref_key = list(_generic_check._check_var_iter(
            ref_key, 'ref_key',
            types=(list, tuple),
            types_iter=str,
            allowed=lref + ldata,
        ))

        # check vs maxd
        if len(ref_key) > maxd:
            msg = (
                f"Arg ref_key shall have no more than {maxd} elements!\n"
                f"Provided: {ref_key}"
            )
            raise Exception(msg)

        # check vs valid vectors
        for ii, rr in enumerate(ref_key):
            if rr in lref:
                kwd = {'ref': rr}
            else:
                kwd = {'key': rr}
            hasref, hasvect, ref, ref_key[ii] = coll.get_ref_vector(
                **kwd,
            )[:4]

            if not (hasref and hasvect):
                msg = (
                    f"Provided ref_key[{ii}] not a valid ref or ref vector!\n"
                    "Provided: {rr}"
                )
                raise Exception(msg)

        lok_keys = [
            k0 for k0, v0 in coll.ddata.items()
            if all([coll.ddata[rr]['ref'][0] in v0['ref'] for rr in ref_key])
        ]

        if keys is None:
            keys = lok_keys
    else:
        lok_keys = list(coll.ddata.keys())

    # keys
    if isinstance(keys, str):
        keys = [keys]

    keys = _generic_check._check_var_iter(
        keys, 'keys',
        types=list,
        types_iter=str,
        allowed=lok_keys,
    )

    # ref_key
    if ref_key is None:
        hasref, ref, ref_key, val, dkeys = coll.get_ref_vector_common(
            keys=keys,
            strategy=ref_vector_strategy,
        )
        if ref_key is None:
            msg = (
                f"No matching ref vector found for:\n"
                f"\t- keys: {keys}\n"
                f"\t- hasref: {hasref}\n"
                f"\t- ref: {ref}\n"
                f"\t- ddata['{keys[0]}']['ref'] = {coll.ddata[keys[0]]['ref']} "
            )
            raise Exception(msg)
        ref_key = (ref_key,)

    # ------------------------
    # daxis, dunits, units_ref

    # daxis
    daxis = {
        k0: [
            coll.ddata[k0]['ref'].index(coll.ddata[rr]['ref'][0])
            for rr in ref_key
        ]
        for k0 in keys
    }

    # dunits
    dunits = {k0: coll.ddata[k0]['units'] for k0 in keys}

    # units_ref
    units_ref = [coll.ddata[rr]['units'] for rr in ref_key]

    return keys, ref_key, daxis, dunits, units_ref


def _check(
    coll=None,
    # interpolation base
    keys=None,
    ref_key=None,      # ddata keys
    # interpolation pts
    x0=None,
    x1=None,
    # useful for shapes
    daxis=None,
    dunits=None,
    domain=None,
    # common ref
    ref_com=None,
    # parameters
    grid=None,
    deg=None,
    deriv=None,
    log_log=None,
    nan0=None,
    # return vs store
    return_params=None,
    returnas=None,
    store=None,
    inplace=None,
    x0_str=None,
):

    ndim = len(ref_key)

    # --------------
    # x0, x1

    lc = [
        isinstance(x0, str) and (ndim == 1 or isinstance(x1, str)),
        not isinstance(x0, str) and (ndim == 1 or not isinstance(x1, str)),
    ]
    if np.sum(lc) != 1:
        msg = (
            "Please x0 (and x1) either as 2 np.ndarrays or as 2 data keys!\n"
            "Provided:\n"
            f"\t- x0: {x0}\n"
            f"\t- x1: {x1}\n"
        )
        raise Exception(msg)


    if lc[0]:
        kx0, kx1, x0, x1, refx0, refx1, ix0, ix1 = _check_x01_str(
            coll=coll,
            x0=x0,
            x1=x1,
            ndim=ndim,
            ref_com=ref_com,
        )

    else:
        kx0, kx1, x0, x1, refx0, refx1, ix0, ix1 = _check_x01_nostr(
            x0=x0,
            x1=x1,
            grid=grid,
            ndim=ndim,
            ref_com=ref_com,
        )

    # -----------------------
    # x0, x1 vs ndim and grid

    x0, x1, refx, ix, xunique = _x01_grid(
        x0=x0,
        x1=x1,
        ndim=ndim,
        grid=grid,
        refx0=refx0,
        refx1=refx1,
        ix0=ix0,
        ix1=ix1,
    )

    # ---------------------
    # get dvect from domain

    domain, dvect = _get_dvect(
        coll=coll,
        domain=domain,
        ref_key=ref_key,
    )

    # ----------------------------------
    # apply domain to coefs (input data)

    ddata = _get_ddata(
        coll=coll,
        keys=keys,
        ref_key=ref_key,
        dvect=dvect,
    )

    # --------
    # dref_com

    ref_com, dref_com = _get_dref_com(
        coll=coll,
        keys=keys,
        ref_key=ref_key,
        ref_com=ref_com,
        ix=ix,
    )

    if ref_com is not None and domain is not None:
        if ref_com in [coll.ddata[k0]['ref'][0] for k0 in dvect.keys()]:
            msg = (
                "Arg ref_com and domain cannot be applied to the same ref!\n"
                f"\t- ref_com: {ref_com}\n"
                f"\t- domain: {domain}\n"
            )
            raise Exception(msg)

    # --------------------------------
    # prepare output shape, units, ref

    dout, dsh_other = _get_dout(
        coll=coll,
        keys=keys,
        refx=refx,
        x0=x0,
        daxis=daxis,
        dunits=dunits,
        dref_com=dref_com,
        dvect=dvect,
    )

    # --------------
    # get drefshape

    sli_c, sli_x, sli_v = _get_slices(
        ndim=ndim,
        x0=x0,
        dref_com=dref_com,
    )

    # ----------------
    # check parameters

    if x0_str is None:
        x0_str = kx0 is not None
    (
        deg, ndim, deriv, log_log, nan0,
        return_params, returnas, store, inplace,
    ) = _check_params(
        coll=coll,
        # interpolation base
        keys=keys,
        ref_key=ref_key,      # ddata keys
        # parameters
        grid=grid,
        deg=deg,
        deriv=deriv,
        log_log=log_log,
        nan0=nan0,
        # return vs store
        return_params=return_params,
        returnas=returnas,
        store=store,
        inplace=inplace,
        # for store
        x0_str=x0_str,
        xunique=xunique,
    )

    return (
        deg, deriv,
        kx0, kx1, x0, x1, refx, dref_com,
        ddata, dout, dsh_other, sli_c, sli_x, sli_v,
        log_log, nan0, grid, ndim, xunique,
        returnas, return_params, store, inplace,
    )


# ###############################################################
# ###############################################################
#               Secondary checking routines
# ###############################################################


def _check_params(
    coll=None,
    # interpolation base
    keys=None,
    ref_key=None,      # ddata keys
    # parameters
    grid=None,
    deg=None,
    deriv=None,
    log_log=None,
    nan0=None,
    # return vs store
    return_params=None,
    returnas=None,
    store=None,
    inplace=None,
    # for store
    x0_str=None,
    xunique=None,
):

    ndim = len(ref_key)

    # ---
    # deg

    deg = _generic_check._check_var(
        deg, 'deg',
        default=1,
        types=int,
        allowed=[0, 1, 2, 3],
    )

    # ----
    # ndim

    if deriv not in [None, 0] and ndim > 1:
        msg = (
            "Arg deriv can only be used for 1d interpolations!\n"
            f"\t- ndim: {ndim}\n"
            f"\t- deriv: {deriv}\n"
        )
        raise Exception(msg)

    if ndim > 2:
        msg = (
            "Interpolations of more than 2 dimensions not implemented!\n"
            f"\t- ref_key: {ref_key}\n"
        )
        raise Exception(msg)

    # -----
    # deriv

    deriv = _generic_check._check_var(
        deriv, 'deriv',
        default=0,
        types=int,
        allowed=[ii for ii in range(deg + 1)],
    )

    # -------
    # log_log

    log_log = _generic_check._check_var(
        log_log, 'log_log',
        default=False,
        types=bool,
    )

    if log_log is True:
        lkout = [k0 for k0 in ref_key if np.any(coll.ddata[k0]['data'] <= 0)]
        if len(lkout) > 0:
            msg = (
                "The following keys cannot be used as ref / coordinates "
                "with log_log=True because they have <=0 values:\n"
                f"\t- {lkout}"
            )
            raise Exception(msg)
        lkout = [k0 for k0 in keys if np.any(coll.ddata[k0]['data'] <= 0)]
        if len(lkout) > 0:
            msg = (
                "The following keys cannot be used as data "
                "with log_log=True because they have <=0 values:\n"
                f"\t- {lkout}"
            )
            raise Exception(msg)

    # -------
    # nan0

    nan0 = _generic_check._check_var(
        nan0, 'nan0',
        default=False,
        types=bool,
    )

    # -------------
    # store

    lok = [False]
    if x0_str or xunique:
        lok.append(True)
    store = _generic_check._check_var(
        store, 'store',
        default=False,
        types=bool,
        allowed=lok,
    )

    # ------------
    # inplace

    lok = [False]
    if store is True:
        lok.append(True)
    inplace = _generic_check._check_var(
        inplace, 'inplace',
        default=False,
        types=bool,
        allowed=lok,
    )

    # -------------
    # returnas
    if store is True:
        if inplace is True:
            ddef = False
            lok = [False, dict, object]
        else:
            ddef = object
            lok = [dict, object]
    else:
        ddef = dict
        lok = [dict]
    returnas = _generic_check._check_var(
        returnas, 'returnas',
        default=ddef,
        allowed=lok,
    )

    # -------------
    # return_params

    return_params = _generic_check._check_var(
        return_params, 'return_params',
        default=False,
        types=bool,
    )

    return (
        deg, ndim, deriv, log_log, nan0,
        return_params, returnas, store, inplace,
    )


def _check_x01_str(
    coll=None,
    x0=None,
    x1=None,
    ndim=None,
    ref_com=None,
):

    # ----
    # x0

    lok = list(coll.ddata.keys())
    x0 = _generic_check._check_var(
        x0, 'x0',
        types=str,
        allowed=lok,
    )

    refx0 = coll.ddata[x0]['ref']

    # ----
    # x1

    if ndim == 2:
        lsame = [
            k0 for k0, v0 in coll.ddata.items()
            if v0['ref'] == refx0
        ]
        x1 = _generic_check._check_var(
            x1, 'x1',
            types=str,
            allowed=list(coll.ddata.keys()),
        )
        refx1 = coll.ddata[x1]['ref']
    else:
        refx1 = None

    # ---------
    # extract

    kx0, x0 = x0, coll.ddata[x0]['data']
    if ndim == 2:
        kx1, x1 = x1, coll.ddata[x1]['data']
    else:
        kx1 = None

    # -----------------------------
    # get potential co-varying refs

    ix0, ix1 = None, None
    if ref_com is not None:

        lok = refx0
        if ndim == 1:
            lok = refx0
        else:
            lok = [k0 for k0 in refx0 if k0 in refx1]

        ref_com = _generic_check._check_var(
            ref_com, 'ref_com',
            types=str,
            allowed=lok,
        )

        # check ref_com is first or last
        ix0 = refx0.index(ref_com)
        if ix0 not in [0, len(refx0) - 1]:
            msg = (
                "cannot handle common ref not as first or last for x\n"
                f"\t- refx0: {refx0}\n"
                f"\t- ref_com: {ref_com}\n"
                f"\t- ix0: {ix0}\n"
            )
            raise Exception(msg)

        # check ref_com is first or last
        if ndim == 2:
            ix1 = refx1.index(ref_com)
            if ix1 not in [0, len(refx1) - 1]:
                msg = (
                    "cannot handle common ref not as first or last for x\n"
                    f"\t- refx1: {refx1}\n"
                    f"\t- ref_com: {ref_com}\n"
                    f"\t- ix1: {ix1}\n"
                )
                raise Exception(msg)

    return kx0, kx1, x0, x1, refx0, refx1, ix0, ix1


def _check_x01_nostr(
    x0=None,
    x1=None,
    grid=None,
    ndim=None,
    ref_com=None,
):

    kx0, kx1 = None, None
    refx0, refx1 = None, None

    x0 = _check_pts(pts=x0, pts_name='x0')
    sh0 = x0.shape
    if ndim >= 2:
        x1 = _check_pts(pts=x1, pts_name='x1')
        sh1 = x1.shape

        # grid
        grid = _generic_check._check_var(
            grid, 'grid',
            default=sh0 != sh1,
            types=bool,
        )

    # ref_com
    if ref_com is not None:
        msg = "Arg ref_com can only be provided for x0 as key1"
        raise Exception(msg)

    return kx0, kx1, x0, x1, refx0, refx1, None, None


def _get_dref_com(
    coll=None,
    keys=None,
    ref_key=None,
    ref_com=None,
    ix=None,
):

    # create dref_com
    lref = [coll.ddata[kk]['ref'][0] for kk in ref_key]
    dref_com = {}
    for k0 in keys:

        # ik0
        refk0 = coll.ddata[k0]['ref']
        if ref_com in refk0:
            ik0 = refk0.index(ref_com)
        else:
            ik0 = None

        # others (taking into account domain)
        ref_other = [rr for rr in refk0 if rr not in lref]
        if ref_com in ref_other:
            iother = ref_other.index(ref_com)
        else:
            iother = None

        # derive shape_other
        # shape_other = [
            # dvect[drv[rr]].size if rr is None else coll.dref[rr]['size']
            # for rr in ref_other
        # ]

        # populate
        dref_com[k0] = {
            'ix': ix,
            # 'ref_other': ref_other,
            # 'shape_other': shape_other,
            'ik0': ik0,
            'iother': iother,
        }

    return ref_com, dref_com


def _x01_grid(
    x0=None,
    x1=None,
    ndim=None,
    grid=None,
    refx0=None,
    refx1=None,
    ix0=None,
    ix1=None,
):


    # ------------
    # trivial case

    if ndim == 1:
        xunique = x0.size == 1
        return x0, x1, refx0, ix0, xunique

    # -----------
    # get shapes

    sh0 = list(x0.shape)
    sh1 = list(x1.shape)

    refx, ix = None, None

    # -------------
    # check vs grid

    if grid is True:

        if ix0 is None:

            sh = sh0 + sh1
            resh = sh0 + [1 for ss in sh1]
            pts0 = np.full(sh, np.nan)
            pts0[...] = x0.reshape(resh)
            resh = [1 for ss in sh0] + sh1
            pts1 = np.full(sh, np.nan)
            pts1[...] = x1.reshape(resh)

            if refx0 is not None:
                refx = tuple(list(refx0) + list(refx1))

        else:
            if ix0 != ix1:
                msg = (
                    "ref_com is not at same index for ref of x0 and x1!\n"
                    f"\t- refx0: {refx0}\n"
                    f"\t- refx1: {refx1}\n"
                )
                raise Exception(msg)

            ix = ix0
            r0 = [rr for ii, rr in enumerate(refx0) if ii != ix]
            r1 = [rr for ii, rr in enumerate(refx1) if ii != ix]
            s0 = [ss for ii, ss in enumerate(sh0) if ii != ix]
            s1 = [ss for ii, ss in enumerate(sh1) if ii != ix]

            refx = r0 + r1
            sh = s0 + s1
            rsh0 = s0 + [1 for ss in s1]
            rsh1 = [1 for ss in s0] + s1
            if ix == 0:
                refx = tuple([refx0[ix]] + refx)
                sh = [sh0[ix]] + sh
                rsh0 = [sh0[ix]] + rsh0
                rsh1 = [sh1[ix]] + rsh1
                ix = 0
            else:
                refx = tuple(refx + [refx0[ix]])
                sh = sh + [sh0[ix]]
                rsh0 = rsh0 + [sh0[ix]]
                rsh1 = rsh1 + [sh1[ix]]
                ix = len(refx) - 1

            pts0 = np.full(sh, np.nan)
            pts1 = np.full(sh, np.nan)
            pts0[...] = x0.reshape(rsh0)
            pts1[...] = x1.reshape(rsh1)

        x0, x1 = pts0, pts1

    elif sh0 == sh1:

        refx = refx0
        if ix0 is not None:
            ix = ix0

    # reshape if necessary
    else:

        if refx0 is not None:
            msg = (
                "Please use grid=True to use data keys for x0 and x1 "
                "of different shapes!\n"
            )
            raise Exception(msg)

        if sh0 == (1,):
            x0 = np.full(sh1, x0[0])

        elif sh1 == (1,):
            x1 = np.full(sh0, x1[0])

        else:
            lc = [
                sh0 == tuple([ss for ss in sh1 if ss in sh0]),
                sh1 == tuple([ss for ss in sh0 if ss in sh1]),
            ]

            if lc[0]:
                resh = [ss if ss in sh0 else 1 for ss in sh1]
                pts0 = np.full(sh1, np.nan)
                pts0[...] = x0.reshape(resh)
                x0 = pts0

            elif lc[1]:
                resh = [ss if ss in sh1 else 1 for ss in sh0]
                pts1 = np.full(sh0, np.nan)
                pts1[...] = x1.reshape(resh)
                x1 = pts1

            else:
                msg = (
                    "No broadcasting solution identified!\n"
                    f"\t- x0.shape: {sh0}\n"
                    f"\t- x1.shape: {sh1}\n"
                )
                raise Exception(msg)

    xunique = x0.size == 1
    return x0, x1, refx, ix, xunique


def _get_dvect(
    coll=None,
    domain=None,
    ref_key=None,
):
    # ----------------
    # domain => dvect

    if domain is not None:

        # get domain
        domain = coll.get_domain_ref(domain)

        # derive dvect
        lvectu = sorted({
            v0['vect'] for v0 in domain.values() if v0['vect'] not in ref_key
        })

        dvect = {
            k0: [k1 for k1, v1 in domain.items() if v1['vect'] == k0]
            for k0 in lvectu
        }

        # check unicity of vect
        dfail = {k0: v0 for k0, v0 in dvect.items() if len(v0) > 1}
        if len(dfail) > 0:
            lstr = [f"\t- '{k0}': {v0}" for k0, v0 in dfail.items()]
            msg = (
                "Some ref vector have been specified with multiple domains!\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

        # build final dvect
        dvect = {
            k0: domain[v0[0]]['ind']
            for k0, v0 in dvect.items()
        }

    else:
        dvect = None

    return domain, dvect


def _get_ddata(
    coll=None,
    keys=None,
    ref_key=None,
    dvect=None,
):

    # --------
    # ddata

    ddata = {}
    for k0 in keys:

        data = coll.ddata[k0]['data']

        # apply domain
        if dvect is not None:
            for k1, v1 in dvect.items():
                ref = coll.ddata[k1]['ref'][0]
                if ref in coll.ddata[k0]['ref']:
                    ax = coll.ddata[k0]['ref'].index(ref)
                    sli = tuple([
                        v1 if ii == ax else slice(None)
                        for ii in range(data.ndim)
                    ])
                    data = data[sli]

        ddata[k0] = data

    return ddata


def _get_dout(
    coll=None,
    keys=None,
    refx=None,
    x0=None,
    daxis=None,
    dunits=None,
    # common refs
    dref_com=None,
    # domain
    dvect=None,
):

    # -------------
    # shape and ref

    dsh = {}
    dref = {}
    dsho = {}

    for k0 in keys:

        # ------------------
        # data shape and ref

        sh = list(coll.ddata[k0]['data'].shape)
        rd = list(coll.ddata[k0]['ref'])

        # apply domain
        if dvect is not None:
            for k1, v1 in dvect.items():
                if coll.ddata[k1]['ref'][0] in rd:
                    ax = rd.index(coll.ddata[k1]['ref'][0])
                    sh[ax] = len(v1) if v1.dtype == int else v1.sum()
                    rd[ax] = None

        # ------------------------
        # fill dshape_other (dsho)

        dsho[k0] = tuple([
            ss for ii, ss in enumerate(sh) if ii not in daxis[k0]
        ])

        # ---------------
        # x shape and ref

        # shx, rx
        shx = x0.shape
        if refx is not None:
            rx = refx

        # ref_com for shx and rx
        if dref_com[k0]['iother'] is not None:
            shx = [ss for ii, ss in enumerate(shx) if ii != dref_com[k0]['ix']]
            if refx is not None:
                rx = [k1 for ii, k1 in enumerate(refx) if ii != dref_com[k0]['ix']]
        # rx
        if refx is None:
            rx = [None]*len(shx)

        # --------------------
        # concatenate data + x

        # dshape
        dsh[k0] = tuple(
            np.r_[sh[:daxis[k0][0]], shx, sh[daxis[k0][-1] + 1:]].astype(int)
        )

        # dref
        dref[k0] = tuple(itt.chain.from_iterable(
            (rd[:daxis[k0][0]], rx, rd[daxis[k0][-1] + 1:])
        ))

    # --------------------
    # prepare output dict

    dout = {
        k0: {
            'data': np.full(dsh[k0], np.nan),
            'units': dunits[k0],
            'ref': dref[k0],
        }
        for k0 in keys
    }

    return dout, dsho


def _get_slices(
    ndim=None,
    x0=None,
    dref_com=None,
):

    # ------------------------
    # coefs (i.e.: input data)

    def sli_c(ind, k0=None, axis=None, ddim=None, ndim=ndim):
        return tuple([
            slice(None) if ii in axis
            else ind[ii - ndim*(ii>axis[0])]
            for ii in range(ddim)
        ])

    # ------------------------
    # x0 (i.e.: interpolation coordinates)

    def sli_x(
        ind,
        ix=None,
        indokx0=None,
        iother=None,
        x0dim=x0.ndim,
    ):
        if iother is None:
            return indokx0
        else:
            ioki = np.take(indokx0, ind[iother], axis=ix)
            if ix == 0:
                return (ind[iother], ioki)
            else:
                return (ioki, ind[iother])

    # ------------------------
    # val (i.e.: interpolated data)

    def sli_v(
        ind,
        indokx0=None,
        ddim=None,
        axis=None,
        ix=None,
        iother=None,
        ndim=ndim,
    ):

        if iother is None:
            return tuple([
                indokx0 if ii == axis[0]
                else ind[ii - ndim*(ii>axis[0])]
                for ii in range(ddim)
                if ii not in axis[1:]
            ])

        else:
            return tuple([
                np.take(indokx0, ind[iother], axis=ix) if ii == axis[0]
                else ind[ii - ndim*(ii>axis[0])]
                for ii in range(ddim)
                if ii not in axis[1:]
            ])

    return sli_c, sli_x, sli_v


def _check_pts(pts=None, pts_name=None):
    if pts is None:
        msg = f"Please provide the interpolation points {pts_name}!"
        raise Exception(msg)

    if not isinstance(pts, np.ndarray):
        try:
            pts = np.atleast_1d(pts)
        except Exception:
            msg = f"{pts_name} should be convertible to a np.ndarray!"
            raise Exception(msg)
    return pts


# ###############################################################
# ###############################################################
#                   interpolate
# ###############################################################


def _interp(
    coll=None,
    keys=None,
    ref_key=None,
    x0=None,
    x1=None,
    ndim=None,
    log_log=None,
    dout=None,
    ddata=None,
    dsh_other=None,
    daxis=None,
    deg=None,
    deriv=None,
    dref_com=None,
    sli_x=None,
    sli_c=None,
    sli_v=None,
    nan0=None,
):

    # ------------
    # Prepare

    # prepare derr
    derr = {}

    # x must be increasing
    x = coll.ddata[ref_key[0]]['data']
    dx = x[1] - x[0]
    if dx < 0:
        x = x[::-1]

    # indokx
    indokx0 = (
        np.isfinite(x0)
        & (x0 >= x.min()) & (x0 <= x.max())
    )

    if log_log is True:
        indokx0 &= (x0 > 0)

    # ------------
    # Interpolate

    # treat oer dimnesionality
    if ndim == 1:

        # loop on keys
        for ii, k0 in enumerate(keys):

            try:
                dout[k0]['data'][...] = _interp1d(
                    out=dout[k0]['data'],
                    data=ddata[k0],
                    shape_other=dsh_other[k0],
                    x=x,
                    x0=x0,
                    axis=daxis[k0],
                    dx=dx,
                    log_log=log_log,
                    deg=deg,
                    deriv=deriv,
                    indokx0=indokx0,
                    dref_com=dref_com[k0],
                    sli_c=sli_c,
                    sli_x=sli_x,
                    sli_v=sli_v,
                )

                # nan0
                if nan0 is True:
                    dout[k0]['data'][dout[k0]['data'] == 0.] = np.nan

            except Exception as err:
                derr[k0] = str(err)
                # raise err

    elif ndim == 2:

        # x, y
        y = coll.ddata[ref_key[1]]['data']
        dy = y[1] - y[0]
        if dy < 0:
            y = y[::-1]

        indokx0 &= (
            np.isfinite(x1)
            & (x1 >= y.min()) & (x1 <= y.max())
        )
        if log_log is True:
            indokx0 &= (x1 > 0)

        # loop on keys
        for ii, k0 in enumerate(keys):

            try:
                dout[k0]['data'][...] = _interp2d(
                    out=dout[k0]['data'],
                    data=ddata[k0],
                    shape_other=dsh_other[k0],
                    x=x,
                    y=y,
                    x0=x0,
                    x1=x1,
                    axis=daxis[k0],
                    dx=dx,
                    dy=dy,
                    log_log=log_log,
                    deg=deg,
                    deriv=deriv,
                    indokx0=indokx0,
                    dref_com=dref_com[k0],
                    sli_c=sli_c,
                    sli_x=sli_x,
                    sli_v=sli_v,
                )

            except Exception as err:
                derr[k0] = str(err)
                # raise err

    else:
        raise NotImplementedError()

    # ----------------------------
    # raise warning if any failure

    if len(derr) > 0:
        lstr = [f"\t- '{k0}': {v0}" for k0, v0 in derr.items()]
        msg = (
            "The following keys could not be interpolated:\n"
            + "\n".join(lstr)
        )
        warnings.warn(msg)


def _interp1d(
    data=None,
    out=None,
    shape_other=None,
    x=None,
    x0=None,
    axis=None,
    dx=None,
    log_log=None,
    deg=None,
    deriv=None,
    indokx0=None,
    dref_com=None,
    sli_c=None,
    sli_x=None,
    sli_v=None,
):

    # ------------
    # trivial case

    if not np.any(indokx0):
        return out

    # x must be strictly increasing
    if dx > 0:
        y = data
    else:
        y = np.flip(data, axis)

    # slicing
    linds = [range(nn) for nn in shape_other]

    for ind in itt.product(*linds):

        slic = sli_c(
            ind,
            axis=axis,
            ddim=data.ndim,
        )

        slix = sli_x(
            ind,
            indokx0=indokx0,
            ix=dref_com['ix'],
            iother=dref_com['iother'],
        )

        sliv = sli_v(
            ind,
            indokx0=indokx0,
            ddim=data.ndim,
            axis=axis,
            ix=dref_com['ix'],
            iother=dref_com['iother'],
        )

        # only keep finite y
        indoki = np.isfinite(y[slic])
        slic = tuple([
            indoki if ii == axis[0] else ss
            for ii, ss in enumerate(slic)
        ])


        xi = x[indoki]
        yi = y[slic]

        # Instanciate interpolation, using finite values only
        if log_log is True:
            out[sliv] = np.exp(
                scpinterp.InterpolatedUnivariateSpline(
                    np.log(xi),
                    np.log(yi),
                    k=deg,
                    ext='zeros',
                )(np.log(x0[slix]), nu=deriv)
            )

        else:
            out[sliv] = scpinterp.InterpolatedUnivariateSpline(
                xi,
                yi,
                k=deg,
                ext='zeros',
            )(x0[slix], nu=deriv)

    return out


def _interp2d(
    data=None,
    out=None,
    shape_other=None,
    x=None,
    y=None,
    x0=None,
    x1=None,
    axis=None,
    dx=None,
    dy=None,
    log_log=None,
    deg=None,
    deriv=None,
    indokx0=None,
    dref_com=None,
    sli_c=None,
    sli_x=None,
    sli_v=None,
):

    # adjust z order
    z = data
    if dx < 0:
        z = np.flip(z, axis=axis[0])
    if dy < 0:
        z = np.flip(z, axis=axis[1])

    # slicing
    linds = [range(nn) for nn in shape_other]
    indi = list(range(data.ndim - 2))
    for ii, aa in enumerate(axis):
        indi.insert(aa + ii, None)

    # -----------
    # interpolate

    for ind in itt.product(*linds):

        slic = sli_c(
            ind,
            axis=axis,
            ddim=data.ndim,
        )

        slix = sli_x(
            ind,
            indokx0=indokx0,
            ix=dref_com['ix'],
            iother=dref_com['iother'],
        )

        sliv = sli_v(
            ind,
            indokx0=indokx0,
            ddim=data.ndim,
            axis=axis,
            ix=dref_com['ix'],
            iother=dref_com['iother'],
        )

        # only keep finite y
        indoki = np.isfinite(z[slic])
        indokix = np.all(indoki, axis=1)
        indokiy = np.all(indoki, axis=0)

        xi = x[indokix]
        yi = y[indokiy]
        zi = z[slic][indokix, :][:, indokiy]

        # Instanciate interpolation, using finite values only
        if log_log is True:
            out[sliv] = np.exp(
                scpinterp.RectBivariateSpline(
                    np.log(xi),
                    np.log(yi),
                    np.log(zi),
                    kx=deg,
                    ky=deg,
                    s=0,
                )(
                    np.log(x0[slix]),
                    np.log(x1[slix]),
                    grid=False,
                )
            )

        else:
            out[sliv] = scpinterp.RectBivariateSpline(
                xi,
                yi,
                zi,
                kx=deg,
                ky=deg,
                s=0,
            )(
                x0[slix],
                x1[slix],
                grid=False,
            )

    return out


# ###############################################################
# ###############################################################
#                   xunique
# ###############################################################


def _xunique(dout=None):
    """ interpolation on a single point => eliminates a ref  """

    # ----------
    # safety check
    
    dind = {
        k0: [jj for jj, rr in enumerate(v0['ref']) if rr is None]
        for k0, v0 in dout.items()
    }

    dwrong = {k0: v0 for k0, v0 in dind.items() if len(v0) != 1}
    if len(dwrong) > 0:
        lstr = [
            f"\t- {k0}: {dout[k0]['ref']} => {v0}" for k0, v0 in dwrong.items()
        ]
        msg = (
            "Interpolation at unique point => ref should have one None:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)
    
    # --------------
    # ajusting dout
    
    for k0, v0 in dout.items():

        i0 = dind[k0][0]
        sli = [slice(None) for jj in range(v0['data'].ndim)]
        sli[i0] = 0
        dout[k0]['data'] = v0['data'][tuple(sli)]
        dout[k0]['ref'] = tuple([
            rr for jj, rr in enumerate(v0['ref'])
            if jj != i0
        ])


# ###############################################################
# ###############################################################
#                   store
# ###############################################################


def _store(
    coll=None,
    dout=None,
    inplace=None,
    store_keys=None,
):

    # -----------
    # inplace

    if inplace is True:
        coll2 = coll

    else:
        ldata = list(set(itt.chain.from_iterable([
            v0['ref'] for v0 in dout.values()
        ])))
        coll2 = coll.extract(keys=ldata, vectors=True)

    # -------------
    # store_keys
    
    if store_keys is None:
        store_keys = [f"{k0}_interp" for k0 in dout.keys()]
    if isinstance(store_keys, str):
        store_keys = [store_keys]
    
    lout = list(coll.ddata.keys())
    store_keys = _generic_check._check_var_iter(
        store_keys, 'store_keys',
        types=list,
        types_iter=str,
        excluded=lout,
    )
    
    assert len(store_keys) == len(dout)

    # ---------
    # add data

    for ii, (k0, v0) in enumerate(dout.items()):

        coll2.add_data(
            key=store_keys[ii],
            data=v0['data'],
            ref=v0['ref'],
            units=v0['units'],
        )

    return coll2

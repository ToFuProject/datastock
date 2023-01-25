# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 20:14:40 2023

@author: dvezinet
"""


import itertools as itt


import numpy as np
import scipy.stats as scpst


# specific
from . import _generic_check


# ############################################################
# ############################################################
#               interpolate spectral
# ############################################################


def binning(
    coll=None,
    keys=None,
    ref_key=None,
    bins=None,
):
    """ Return the binned data

    """

    # ----------
    # checks

    # keys
    keys, ref_key, daxis, dunits, units_ref = _check_keys(
        coll=coll,
        keys=keys,
        ref_key=ref_key,
        only1d=True,
    )

    # because 1d only
    ref_key = ref_key[0]
    for k0, v0 in daxis.items():
        daxis[k0] = v0[0]
    units_ref = units_ref[0]

    # bins
    bins, units_bins, db, _ = _check_bins(
        coll=coll,
        keys=keys,
        ref_key=ref_key,
        bins=bins,
        vect=coll.ddata[ref_key]['data'],
        strict=True,
    )

    # units
    dout = _units(
        dunits=dunits,
        units_ref=units_ref,
        units_bins=units_bins,
    )

    # --------------
    # actual binning

    for k0, v0 in dout.items():
        dout[k0]['data'] = _bin(
            bins=bins,
            db=db,
            vect=coll.ddata[ref_key]['data'],
            data=coll.ddata[k0]['data'],
            axis=daxis[k0],
        )
        ref = list(coll.ddata[k0]['ref'])
        ref[daxis[k0]] = None
        dout[k0]['ref'] = tuple(ref)

    return dout


# ####################################
#       check
# ####################################


def _check_keys(
    coll=None,
    keys=None,
    ref_key=None,
    only1d=None,
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
            hasref, hasvect, ref, ref_key[ii] = coll.get_ref_vector(**kwd)[:4]

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


def _check_bins(
    coll=None,
    keys=None,
    ref_key=None,
    bins=None,
    vect=None,
    # if bsplines
    strict=None,
    deg=None,
):

    # check
    strict = _generic_check._check_var(
        strict, 'strict',
        types=bool,
        default=True,
    )

    # ---------
    # bins

    if isinstance(bins, str):
        lok = [k0 for k0, v0 in coll.ddata.items() if v0['monot'] == (True,)]
        bins = _generic_check._check_var(
            bins, 'bins',
            types=str,
            allowed=lok,
        )

        bins_units = coll.ddata[bins]['units']
        bins = coll.ddata[bins]['data']

    else:
        bins_units = None

    bins = _generic_check._check_flat1darray(
        bins, 'bins',
        dtype=float,
        unique=True,
        can_be_None=False,
    )

    # -----------------
    # check uniformity

    db = np.abs(np.diff(bins))
    if not np.allclose(db[0], db):
        msg = (
            "Arg bins must be a uniform bin edges vector!"
            f"Provided diff(bins) = {db}"
        )
        raise Exception(msg)
    db = db[0]

    # ----------
    # bin method

    dv = np.abs(np.mean(np.diff(vect)))

    if strict is True:
        if db < 2*dv:
            msg = (
                f"Uncertain binning for '{sorted(keys)}', ref vect '{ref_key}':\n"
                f"Binning steps ({db}) are smaller than 2*ref ({2*dv}) vector step"
            )
            raise Exception(msg)

        else:
            npts = None

    else:
        npts = (deg + 3) * max(1, dv / db)

    return bins, bins_units, db, npts


def _units(
    dunits=None,
    units_ref=None,
    units_bins=None,
):

    dout = {}
    for k0, v0 in dunits.items():
        if v0 in [None, '']:
            dout[k0] = {'units': ''}

        elif units_ref in [None, ''] and units_bins in [None, '']:
            dout[k0] = {'units': ''}

        elif units_ref in [None, '']:
            dout[k0] = {'units': v0 * units_bins}

        elif units_bins in [None, '']:
            dout[k0] = {'units': v0 * units_ref}

        elif units_ref == units_bins:
            dout[k0] = {'units': v0 * units_ref}

        else:
            msg = (
                "Units do not agree between ref vector and bins for '{k0}'!\n"
                f"\t- units     : {v0}\n"
                f"\t- units_ref : {units_ref}\n"
                f"\t- units_bins: {units_bins}\n"
            )
            raise Exception(msg)

    return dout


# ####################################
# ####################################
#       bin
# ####################################


def _bin(
    bins=None,
    db=None,
    vect=None,
    data=None,
    axis=None,
):

    indin = (vect >= bins[0]) & (vect <= bins[-1])
    vect = vect[indin]

    if data.ndim == 1:

        data = data[indin]

        val = scpst.binned_statistic(
            vect,
            data,
            bins=bins,
            statistic=np.nansum,
        )[0]

    else:

        # remove out
        sli = tuple([
            indin if ii == axis else slice(None)
            for ii in range(data.ndim)
        ])

        data = data[sli]

        # shape
        shape = list(data.shape)
        shape[axis] = int(bins.size - 1)
        shape_other = np.r_[shape[:axis], shape[axis+1:]].astype(int)

        # indices
        linds = [range(nn) for nn in shape_other]
        indi = list(range(data.ndim-1))
        indi.insert(axis, None)

        # initialize val
        val = np.zeros(tuple(shape), dtype=data.dtype)

        for ind in itt.product(*linds):

            sli = tuple([
                slice(None) if ii == axis else ind[indi[ii]]
                for ii in range(len(shape))
            ])

            # bin
            val[sli] = scpst.binned_statistic(
                vect,
                data[sli],
                bins=bins,
                statistic=np.nansum,
            )[0]

    return val * db

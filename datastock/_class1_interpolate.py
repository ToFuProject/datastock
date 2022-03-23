# -*- coding: utf-8 -*-


# Builtin
import warnings

# Common
import numpy as np
import scipy.interpolate as scpinterp

from . import _generic_check


# #############################################################################
# #############################################################################
#           Interpolate
# #############################################################################



def interpolate(
    keys_ref=None,
    keys=None,
    pts_axis0=None,
    pts_axis1=None,
    pts_axis2=None,
    grid=None,
    ddata=None,
    dref=None,
    deg=None,
):
    """ Interpolate at desried points on desired data """

    # --------------
    # check inputs

    keys_ref, keys, deg, pts_axis0, pts_axis1, pts_axis2, grid, ndim = _check(
        keys_ref=keys_ref,
        keys=keys,
        deg=deg,
        pts_axis0=pts_axis0,
        pts_axis1=pts_axis1,
        pts_axis2=pts_axis2,
        grid=grid,
        ddata=ddata,
        dref=dref,
    )

    # ----------------
    # Interpolate

    if ndim == 1:

        for k0 in keys:
            spl = scpinterp.InterpolatedUnivariateSpline(
                ddata[keys_ref[0]]['data'],
                ddata[k0]['data'],
            )

    elif ndim == 2:
        pass

    else:
        pass

    return values

# #############################################################################
# #############################################################################
#           Utilities
# #############################################################################


def _check_pts(pts=None):
    if pts_axis0 is None:
        msg = "Please provide the interpolation points pts_axis0!"
        raise Exception(msg)

    if not isinstance(pts_axis0, np.ndarray):
        try:
            pts_axis0 = np.atleast_1d(pts_axis0)
        except Exception:
            msg = "pts_axis0 should be convertible to a np.ndarray!"
            raise Exception(msg)
    return pts


def _check(
    keys_ref=None,
    keys=None,
    pts_axis0=None,
    pts_axis1=None,
    pts_axis2=None,
    grid=None,
    ddata=None,
    dref=None,
    deg=None,
):

    # ---
    # key

    lkok = list(ddata.keys())
    if isinstance(keys, str):
        keys = [keys]
    key = _generic_check._check_var_iter(
        keys, 'keys',
        types=list,
        types_iter=str,
        allowed=lkok,
    )

    lrefs = set([ddata[kk]['ref'] for kk in keys])
    if len(lrefs) != 1:
        msg = (
            "All interpolation keys must share the same refs!\n"
            f"\t- keys: {keys}\n"
            f"\t- refs: {lrefs}\n"
        )
        raise Exception(msg)

    ref = ddata[keys[0]]['ref']
    ndim = ddata['data'].ndim
    assert ndim == len(ref)
    if ndim > 2:
        msg = "Interpolation not implemented for more than 3 dimensions!"
        raise NotImplementedError(msg)

    # --------
    # keys_ref

    if keys_ref is None:
        keys_ref = [None for rr in ref]

    if ndim == 1 and (keys_ref is None or isinstance(keys_ref, str)):
        keys_ref = [keys_ref]
    c0 = (
        isinstance(keys_ref, list)
        and len(keys_ref) == ndim
        and all([kk is None or kk in ddata.keys() for kk in keys_ref])
    )
    if not c0:
        msg = (
            "Arg keys_ref must be a list of keys to monotonous data\n"
            f"One for each component in data['ref']: {ref}\n"
            f"Provided: {keys_ref}"
        )
        raise Exception(msg)

    # check for each dimension
    for ii, rr in enumerate(ref):
        lok = [
            k1 for k1, v1 in ddata.items()
            if v1['ref'] == (rr,)
            and v1['monotonous'] == True
        ]
        keys_ref[ii] = _generic_check._check_var(
            keys_ref[ii], f'keys_ref[{ii}]',
            types=str,
            allowed=lok,
        )

    # ---
    # deg

    deg = _generic_check._check_var(
        deg, 'deg',
        default=1,
        types=int,
        allowed=[0, 1, 2, 3],
    )

    # ---
    # grid

    grid = _generic_check._check_var(
        grid, 'grid',
        default=False,
        types=bool,
    )

    # ---
    # pts_axis0

    pts_axis0 = _check_pts(pts=pts_axis0)
    if ndim == 1:

        scpinterp.InterpolatedUnivariateSpline()

    elif ndim == 2:

        pts_axis1 = _check_pts(pts=pts_axis1)
        sh0 = pts_axis0.shape
        sh1 = pts_axis1.shape

        if grid is True:
            sh = list(sh0) + list(sh1)
            resh = list(sh0) + [1 for ss in sh1]
            pts0 = np.full(sh, np.nan)
            pts0[...] = pts_axis0.reshape(resh)
            resh = [1 for ss in sh0] + list(sh1)
            pts1 = np.full(sh, np.nan)
            pts1[...] = pts_axis1.reshape(resh)
            pts_axis0, pts_axis1 = pts0, pts1

        elif sh0 != sh1:

            if sh0 == (1,):
                pts_axis0 = np.full(sh1, pts_axis0[0])

            elif sh1 == (1,):
                pts_axis1 = np.full(sh0, pts_axis1[0])

            else:
                lc = [
                    sh0 = tuple([ss for ss in sh1 if ss in sh0]),
                    sh1 = tuple([ss for ss in sh0 if ss in sh1]),
                ]

                if lc[0]:
                    resh = [ss if ss in sh0 else 1 for ss in sh1]
                    pts0 = np.full(sh1, np.nan)
                    pts0[...] = pts_axis0.reshape(resh)
                    pts_axis0 = pts0

                elif lc[1]:
                    resh = [ss if ss in sh1 else 1 for ss in sh0]
                    pts1 = np.full(sh0, np.nan)
                    pts1[...] = pts_axis1.reshape(resh)
                    pts_axis1 = pts1

                else:
                    msg = (
                        "No broadcasting solution identified!\n"
                        f"\t- pts_axis0.shape: {sh0}\n"
                        f"\t- pts_axis1.shape: {sh1}\n"
                    )
                    raise Exception(msg)

    else:

        pts_axis1 = _check_pts(pts=pts_axis1)
        pts_axis2 = _check_pts(pts=pts_axis2)



    return keys_ref, keys, deg, pts_axis0, pts_axis1, pts_axis2, grid, ndim

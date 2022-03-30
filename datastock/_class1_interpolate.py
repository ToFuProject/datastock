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
    deriv=None,
    log_log=None,
):
    """ Interpolate at desired points on desired data

    Interpolate quantities (keys) on coordinates (keys_ref)
    All provided keys should share the same refs
    They should have dimension 2 or less

    keys_ref should be a list of monotonous data keys to be used as coordinates
    It should have one element per dimension in refs

    The interpolation points are provided as np.ndarrays in each dimension
    They should all have the same shape except if grid = True

    deg is the degree of the interpolation

    It is an interpolation, not a smoothing, so it will pass through all points
    Uses scpinterp.InterpolatedUnivariateSpline() for 1d

    """

    # --------------
    # check inputs

    (
        keys_ref, keys,
        deg, deriv,
        pts_axis0, pts_axis1, pts_axis2,
        log_log, grid, ndim,
    ) = _check(
        keys_ref=keys_ref,
        keys=keys,
        deg=deg,
        deriv=deriv,
        pts_axis0=pts_axis0,
        pts_axis1=pts_axis1,
        pts_axis2=pts_axis2,
        grid=grid,
        log_log=log_log,
        ddata=ddata,
        dref=dref,
    )

    # ----------------
    # Interpolate

    shape = pts_axis0.shape
    dvalues = {k0: np.full(shape, np.nan) for k0 in keys}

    if ndim == 1:

        for k0 in keys:

            # x must be strictly increasing
            if ddata[keys_ref[0]]['data'][1] > ddata[keys_ref[0]]['data'][0]:
                x = ddata[keys_ref[0]]['data']
                y = ddata[k0]['data']
            else:
                x = ddata[keys_ref[0]]['data'][::-1]
                y = ddata[k0]['data'][::-1]

            # only keep finite y
            indok = np.isfinite(y)
            x = x[indok]
            y = y[indok]

            # Interpolate on finite values within boundaries only
            indok = (
                np.isfinite(pts_axis0)
                & (pts_axis0 >= x[0]) & (pts_axis0 <= x[-1])
            ).nonzero()[0]
            # sort for more efficient evaluation
            indok = indok[np.argsort(pts_axis0[indok])]

            # Instanciate interpolation, using finite values only
            if log_log is True:
                dvalues[k0][indok] = np.exp(
                    scpinterp.InterpolatedUnivariateSpline(
                        np.log(x),
                        np.log(y),
                        k=deg,
                        ext='zeros',
                    )(np.log(pts_axis0[indok]), nu=deriv)
                )

            else:
                dvalues[k0][indok] = scpinterp.InterpolatedUnivariateSpline(
                    x,
                    y,
                    k=deg,
                    ext='zeros',
                )(pts_axis0[indok], nu=deriv)

    elif ndim == 2:
        pass

    else:
        pass

    return dvalues

# #############################################################################
# #############################################################################
#           Utilities
# #############################################################################


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


def _check(
    keys_ref=None,
    keys=None,
    pts_axis0=None,
    pts_axis1=None,
    pts_axis2=None,
    grid=None,
    deg=None,
    deriv=None,
    log_log=None,
    ddata=None,
    dref=None,
):

    # ---
    # keys

    lkok = list(ddata.keys())
    if isinstance(keys, str):
        keys = [keys]
    keys = _generic_check._check_var_iter(
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
    ndim = ddata[keys[0]]['data'].ndim
    assert ndim == len(ref)
    if ndim > 2:
        msg = "Interpolation not implemented for more than 3 dimensions!"
        raise NotImplementedError(msg)

    # --------
    # keys_ref

    if keys_ref is None:
        keys_ref = [None for rr in ref]

    if keys_ref == 1 and (keys_ref is None or isinstance(keys_ref, str)):
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
            and v1['monot'] == (True,)
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
    # deriv

    deriv = _generic_check._check_var(
        deriv, 'deriv',
        default=0,
        types=int,
        allowed=[ii for ii in range(deg + 1)],
    )

    # ---
    # grid

    grid = _generic_check._check_var(
        grid, 'grid',
        default=False,
        types=bool,
    )

    # ---
    # log_log

    log_log = _generic_check._check_var(
        log_log, 'log_log',
        default=False,
        types=bool,
    )

    # ---
    # pts_axis0

    pts_axis0 = _check_pts(pts=pts_axis0, pts_name='pts_axis0')
    if ndim == 1:

        pass

    elif ndim == 2:

        pts_axis1 = _check_pts(pts=pts_axis1, pts_name='pts_axis1')
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
                    sh0 == tuple([ss for ss in sh1 if ss in sh0]),
                    sh1 == tuple([ss for ss in sh0 if ss in sh1]),
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
        raise NotImplementedError()

    return (
        keys_ref, keys,
        deg, deriv,
        pts_axis0, pts_axis1, pts_axis2,
        log_log, grid, ndim,
    )

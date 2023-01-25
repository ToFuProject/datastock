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
from . import _class1_binning


# ##################################################################
# ##################################################################
#           Interpolate
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
    # parameters
    deg=None,
    deriv=None,
    log_log=None,
    return_params=None,
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
    keys, ref_key, daxis, dunits, _ = _class1_binning._check_keys(
        coll=coll,
        keys=keys,
        ref_key=ref_key,
        only1d=False,
    )

    # params
    (
        deg, deriv,
        x0, x1,
        log_log, grid, ndim, return_params,
    ) = _check_params(
        coll=coll,
        # interpolation base
        keys=keys,
        ref_key=ref_key,      # ddata keys
        # interpolation pts
        x0=x0,
        x1=x1,
        # parameters
        grid=grid,
        deg=deg,
        deriv=deriv,
        log_log=log_log,
        return_params=return_params,
    )

    # ------------
    # prepare

    dshape, dout = _prepare_dshape_dout(
        coll=coll,
        keys=keys,
        daxis=daxis,
        dunits=dunits,
        x0=x0,
    )

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
                dout[k0]['data'] = _interp1d(
                    out=dout[k0]['data'],
                    data=coll.ddata[k0]['data'],
                    dshape=dshape[k0],
                    x=x,
                    x0=x0,
                    axis=daxis[k0][0],
                    dx=dx,
                    log_log=log_log,
                    deg=deg,
                    deriv=deriv,
                    indokx0=indokx0,
                )
                dout[k0]['units'] = dunits[k0]

            except Exception as err:
                derr[k0] = str(err)

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
                dout[k0]['data'] = _interp2d(
                    out=dout[k0]['data'],
                    data=coll.ddata[k0]['data'],
                    dshape=dshape[k0],
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
                )
                dout[k0]['units'] = dunits[k0]

            except Exception as err:
                derr[k0] = str(err)

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

    # -------
    # return

    if return_params is True:
        dparam = {
            'keys': keys,
            'ref_key': ref_key,
            'deg': deg,
            'deriv': deriv,
            'log_log': log_log,
            'x0': x0,
            'x1': x1,
            'grid': grid,
        }
        return dout, dparam

    return dout


# #############################################################################
# #############################################################################
#           Utilities
# #############################################################################


def _check_params(
    coll=None,
    # interpolation base
    keys=None,
    ref_key=None,      # ddata keys
    # interpolation pts
    x0=None,
    x1=None,
    # parameters
    grid=None,
    deg=None,
    deriv=None,
    log_log=None,
    return_params=None,
):

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

    ndim = len(ref_key)
    if deriv not in [None, 0] and ndim > 1:
        msg = (
            "Arg deriv can only be used for 1d interpolations!\n"
            f"\t- ndim: {ndim}\n"
            f"\t- deriv: {deriv}\n"
        )
        raise Exception(msg)

    deriv = _generic_check._check_var(
        deriv, 'deriv',
        default=0,
        types=int,
        allowed=[ii for ii in range(deg + 1)],
    )

    # ---
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

    # -------------
    # return_params

    return_params = _generic_check._check_var(
        return_params, 'return_params',
        default=False,
        types=bool,
    )

    # --------------------------
    # x0, x1, grid

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

    # cases
    if ndim == 1:

        pass

    elif ndim == 2:

        if grid is True:
            sh = list(sh0) + list(sh1)
            resh = list(sh0) + [1 for ss in sh1]
            pts0 = np.full(sh, np.nan)
            pts0[...] = x0.reshape(resh)
            resh = [1 for ss in sh0] + list(sh1)
            pts1 = np.full(sh, np.nan)
            pts1[...] = x1.reshape(resh)
            x0, x1 = pts0, pts1

        elif sh0 != sh1:

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

    else:
        raise NotImplementedError()

    return (
        deg, deriv,
        x0, x1,
        log_log, grid, ndim, return_params,
    )


def _prepare_dshape_dout(
    coll=None,
    keys=None,
    daxis=None,
    dunits=None,
    x0=None,
):

    # dshape
    dshape = {
        k0: _get_shapes_axis_ind(
            axis=daxis[k0],
            shape_coefs=coll.ddata[k0]['data'].shape,
            shape_x=x0.shape,
            shape_bs=[coll.ddata[k0]['data'].shape[aa] for aa in daxis[k0]],
        )
        for k0 in keys
    }

    # dout
    dout = {
        k0: {
            'data': np.full(dshape[k0]['shape_val'], np.nan),
            'units': dunits[k0],
        }
        for k0 in keys
    }
    return dshape, dout


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


# ##################################################################
# ##################################################################
#                   get shapes dict
# ##################################################################


def _get_shapes_axis_ind(axis=None, shape_coefs=None, shape_x=None, shape_bs=None):

    # -------------------------------------------------
    # initial safety check on coefs vs shapebs vs axis

    c0 = (
        len(shape_coefs) >= len(shape_bs)
        and len(axis) == len(shape_bs)
        and all([aa == axis[0] + ii for ii, aa in enumerate(axis)])
        and all([shape_coefs[aa] == shape_bs[ii] for ii, aa in enumerate(axis)])
    )
    if not c0:
        msg = (
            f"Arg shape_coefs must include {shape_bs}\n"
            f"\t- axis: {axis}\n"
            f"\t- shape_coefs: {shape_coefs}\n"
            f"\t- shape_bs: {shape_bs}\n"
            f"\t- shape_x: {shape_x}\n"
        )
        raise Exception(msg)

    # ----------------
    # shape for output

    shape_val, axis_x, ind_coefs, ind_x = [], [], [], []
    ij = 0
    for ii in range(len(shape_coefs)):
        if ii == axis[0]:
            for jj in range(len(shape_x)):
                shape_val.append(shape_x[jj])
                axis_x.append(ii + jj)
                ind_x.append(None)
            ind_coefs.append(None)

        elif len(axis) > 1 and ii in axis[1:]:
            ind_coefs.append(None)
        else:
            shape_val.append(shape_coefs[ii])
            ind_coefs.append(ij)
            ind_x.append(ij)
            ij += 1

    # ----------------
    # shape_other

    shape_other = tuple([
        ss for ii, ss in enumerate(shape_coefs)
        if ii not in axis
    ])

    return {
        'shape_val': tuple(shape_val),
        'shape_other': shape_other,
        'axis_x': axis_x,
        'ind_coefs': ind_coefs,
        'ind_x': ind_x,
    }


# ##################################################################
# ##################################################################
#                   interpolate
# ##################################################################


def _interp1d(
    data=None,
    out=None,
    dshape=None,
    x=None,
    x0=None,
    axis=None,
    dx=None,
    log_log=None,
    deg=None,
    deriv=None,
    indokx0=None,
):

    # x must be strictly increasing
    if dx > 0:
        y = data
    else:
        y = np.flip(data, axis)

    # slicing
    linds = [range(nn) for nn in dshape['shape_other']]
    indi = list(range(data.ndim - 1))
    indi.insert(axis, None)

    print(x.shape, x0.shape, y.shape)
    print(dshape)
    print(linds, indi)

    for ind in itt.product(*linds):

        sli = [
            slice(None) if ii == axis else ind[indi[ii]]
            for ii in range(len(y.shape))
        ]
        sli_val = tuple([
            indokx0 if ii == axis else ind[indi[ii]]
            for ii in range(len(y.shape))
        ])

        # only keep finite y
        indoki = np.isfinite(y[tuple(sli)])
        sli[axis] = indoki

        xi = x[indoki]
        yi = y[tuple(sli)]

        # Instanciate interpolation, using finite values only
        if log_log is True:
            out[sli_val] = np.exp(
                scpinterp.InterpolatedUnivariateSpline(
                    np.log(xi),
                    np.log(yi),
                    k=deg,
                    ext='zeros',
                )(np.log(x0[indokx0]), nu=deriv)
            )

        else:
            out[sli_val] = scpinterp.InterpolatedUnivariateSpline(
                xi,
                yi,
                k=deg,
                ext='zeros',
            )(x0[indokx0], nu=deriv)

    return out


def _interp2d(
    data=None,
    out=None,
    dshape=None,
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
):

    # adjust z order
    z = data
    if dx < 0:
        z = np.flip(z, axis=axis[0])
    if dy < 0:
        z = np.flip(z, axis=axis[1])

    # slicing
    linds = [range(nn) for nn in dshape['shape_other']]
    indi = list(range(data.ndim - 2))
    for ii, aa in enumerate(axis):
        indi.insert(aa + ii, None)

    # -----------
    # interpolate

    for ind in itt.product(*linds):

        sli = [
            slice(None) if ii in axis else ind[indi[ii]]
            for ii in range(len(z.shape))
        ]
        sli_val = tuple([
            indokx0 if ii == axis[0] else ind[indi[ii]]
            for ii in range(len(z.shape))
            if ii != axis[1]
        ])

        # only keep finite y
        indoki = np.isfinite(z[tuple(sli)])
        indokix = np.all(indoki, axis=1)
        indokiy = np.all(indoki, axis=0)

        xi = x[indokix]
        yi = y[indokiy]
        zi = z[tuple(sli)][indokix, :][:, indokiy]

        # Instanciate interpolation, using finite values only
        if log_log is True:
            out[sli_val] = np.exp(
                scpinterp.RectBivariateSpline(
                    np.log(xi),
                    np.log(yi),
                    np.log(zi),
                    kx=deg,
                    ky=deg,
                    s=0,
                )(
                    np.log(x0[indokx0]),
                    np.log(x1[indokx0]),
                    grid=False,
                )
            )

        else:
            out[sli_val] = scpinterp.RectBivariateSpline(
                xi,
                yi,
                zi,
                kx=deg,
                ky=deg,
                s=0,
            )(
                x0[indokx0],
                x1[indokx0],
                grid=False,
            )

    return out

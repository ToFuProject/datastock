# -*- coding: utf-8 -*-


# Builtin
import warnings


# Common
import numpy as np
import scipy.interpolate as scpinterp


# local
from . import _generic_check


# #############################################################################
# #############################################################################
#           Interpolate
# #############################################################################


def interpolate(
    # interpolation base
    keys=None,
    ref_keys=None,
    ref_quant=None,
    # interpolation pts
    pts_axis0=None,
    pts_axis1=None,
    pts_axis2=None,
    grid=None,
    # parameters
    deg=None,
    deriv=None,
    log_log=None,
    return_params=None,
    # ressources
    ddata=None,
    dref=None,
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

    # --------------
    # check inputs

    (
        ref_keys, keys,
        deg, deriv,
        pts_axis0, pts_axis1, pts_axis2,
        log_log, grid, ndim, return_params,
    ) = _check(
        # interpolation base
        keys=keys,
        ref_keys=ref_keys,
        ref_quant=ref_quant,
        # interpolation pts
        pts_axis0=pts_axis0,
        pts_axis1=pts_axis1,
        pts_axis2=pts_axis2,
        # parameters
        deg=deg,
        deriv=deriv,
        grid=grid,
        log_log=log_log,
        return_params=return_params,
        # ressources
        ddata=ddata,
        dref=dref,
    )

    # ----------------
    # Interpolate

    shape = pts_axis0.shape
    dvalues = {k0: np.full(shape, np.nan) for k0 in keys}
    derr = {}

    # x must be increasing
    x = ddata[ref_keys[0]]['data']
    dx = x[1] - x[0]
    if dx < 0:
        x = x[::-1]

    # treat oer dimnesionality
    if ndim == 1:

        # loop on keys
        for ii, k0 in enumerate(keys):

            try:
                # x must be strictly increasing
                if dx > 0:
                    y = ddata[k0]['data']
                else:
                    y = ddata[k0]['data'][::-1]

                # only keep finite y
                indok = np.isfinite(y)
                xi = x[indok]
                y = y[indok]

                # Interpolate on finite values within boundaries only
                indok = (
                    np.isfinite(pts_axis0)
                    & (pts_axis0 >= xi[0]) & (pts_axis0 <= xi[-1])
                )
                if log_log is True:
                    indok &= pts_axis0 > 0
                indok = indok.nonzero()[0]

                # sort for more efficient evaluation
                indok = indok[np.argsort(pts_axis0[indok])]

                # Instanciate interpolation, using finite values only
                if log_log is True:
                    dvalues[k0][indok] = np.exp(
                        scpinterp.InterpolatedUnivariateSpline(
                            np.log(xi),
                            np.log(y),
                            k=deg,
                            ext='zeros',
                        )(np.log(pts_axis0[indok]), nu=deriv)
                    )

                else:
                    dvalues[k0][indok] = scpinterp.InterpolatedUnivariateSpline(
                        xi,
                        y,
                        k=deg,
                        ext='zeros',
                    )(pts_axis0[indok], nu=deriv)

            except Exception as err:
                derr[k0] = str(err)

    elif ndim == 2:

        # x, y
        y = ddata[ref_keys[1]]['data']
        dy = y[1] - y[0]
        if dy < 0:
            y = y[::-1]

        # loop on keys
        for ii, k0 in enumerate(keys):

            try:

                # adjust z order
                z = ddata[k0]['data']
                if dx < 0:
                    z = np.flip(z, axis=0)
                if dy < 0:
                    z = np.flip(z, axis=1)

                # only keep finite y
                indok = np.isfinite(z)
                indokx = np.all(indok, axis=1)
                indoky = np.all(indok, axis=0)
                xi = x[indokx]
                yi = y[indoky]
                z = z[indokx, :][:, indoky]

                # Interpolate on finite values within boundaries only
                indok = (
                    np.isfinite(pts_axis0)
                    & np.isfinite(pts_axis1)
                    & (pts_axis0 >= xi[0]) & (pts_axis0 <= xi[-1])
                    & (pts_axis1 >= yi[0]) & (pts_axis1 <= yi[-1])
                )
                if log_log is True:
                    indok &= (pts_axis0 > 0) & (pts_axis1 > 0)

                # Instanciate interpolation, using finite values only
                if log_log is True:
                    dvalues[k0][indok] = np.exp(
                        scpinterp.RectBivariateSpline(
                            np.log(xi),
                            np.log(yi),
                            np.log(z),
                            kx=deg,
                            ky=deg,
                            s=0,
                        )(
                            np.log(pts_axis0[indok]),
                            np.log(pts_axis1[indok]),
                            grid=False,
                        )
                    )
                else:
                    dvalues[k0][indok] = scpinterp.RectBivariateSpline(
                        xi,
                        yi,
                        z,
                        kx=deg,
                        ky=deg,
                        s=0,
                    )(
                        pts_axis0[indok],
                        pts_axis1[indok],
                        grid=False,
                    )

            except Exception as err:
                derr[k0] = str(err)

    else:
        pass

    # ----------------------------
    # raise warning if any failure

    if len(derr) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in derr.items()]
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
            'ref_keys': ref_keys,
            'deg': deg,
            'deriv': deriv,
            'log_log': log_log,
            'pts_axis0': pts_axis0,
            'pts_axis1': pts_axis0,
            'pts_axis2': pts_axis0,
            'grid': grid,
        }
        return dvalues, dparam
    else:
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
    # interpolation base
    keys=None,
    ref_keys=None,      # ddata keys
    ref_quant=None,     # ddata[ref_key]['quant'], not used yet
    # interpolation pts
    pts_axis0=None,
    pts_axis1=None,
    pts_axis2=None,
    # parameters
    grid=None,
    deg=None,
    deriv=None,
    log_log=None,
    return_params=None,
    # ressources
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
    # ref_keys

    if ref_keys is None:
        ref_keys = [None for rr in ref]

    if isinstance(ref_keys, str):
        ref_keys = [ref_keys for rr in ref]

    c0 = (
        isinstance(ref_keys, list)
        and len(ref_keys) == ndim
        and all([kk is None or kk in ddata.keys() for kk in ref_keys])
    )
    if not c0:
        msg = (
            "Arg ref_keys must be a list of keys to monotonous data\n"
            f"One for each component in data['ref']: {ref}\n"
            f"Provided: {ref_keys}"
        )
        raise Exception(msg)

    # ref_quant
    lqok = [v0.get('quant') for v0 in ddata.values()] + [None]
    if ref_quant is None:
        ref_quant = [None for ii in range(ndim)]
    ref_quant = _generic_check._check_var_iter(
        ref_quant, 'ref_quant',
        types=list,
        allowed=lqok,
    )

    # check for each dimension
    for ii, rr in enumerate(ref):
        lok = [
            k1 for k1, v1 in ddata.items()
            if v1['ref'] == (rr,)
            and v1['monot'] == (True,)
            and (ref_quant[ii] is None or v1.get('quant') == ref_quant[ii])
        ]
        ref_keys[ii] = _generic_check._check_var(
            ref_keys[ii], f'ref_keys[{ii}]',
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
        lkout = [k0 for k0 in ref_keys if np.any(ddata[k0]['data'] <= 0)]
        if len(lkout) > 0:
            msg = (
                "The following keys cannot be used as ref / coordinates "
                "with log_log=True because they have <=0 values:\n"
                f"\t- {lkout}"
            )
            raise Exception(msg)
        lkout = [k0 for k0 in keys if np.any(ddata[k0]['data'] <= 0)]
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
    # pts_axis0, pts_axis1, grid

    pts_axis0 = _check_pts(pts=pts_axis0, pts_name='pts_axis0')
    sh0 = pts_axis0.shape
    if ndim >= 2:
        pts_axis1 = _check_pts(pts=pts_axis1, pts_name='pts_axis1')
        sh1 = pts_axis1.shape

        # grid
        grid = _generic_check._check_var(
            grid, 'grid',
            default=sh0 != sh1,
            types=bool,
        )

    if ndim == 1:

        pass

    elif ndim == 2:

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
        ref_keys, keys,
        deg, deriv,
        pts_axis0, pts_axis1, pts_axis2,
        log_log, grid, ndim, return_params,
    )


# #############################################################################
# #############################################################################
#           Monotonous vector
# #############################################################################


def _get_ref_vector_nearest(x0, x):
    x0bins = 0.5*(x0[1:] + x0[:-1])
    ind = np.digitize(x, x0bins)

    vmin = np.min(x0)
    vmax = np.max(x0)
    dmax2 = np.max(np.diff(x0)) / 2.
    indok = (x >= vmin - dmax2) & (x <= vmax + dmax2)
    return ind, indok


def get_ref_vector(
    # ressources
    ddata=None,
    dref=None,
    # inputs
    key=None,
    ref=None,
    dim=None,
    quant=None,
    name=None,
    units=None,
    # parameters
    values=None,
    indices=None,
    ind_strict=None,
):

    # ----------------
    # check inputs

    # ind_strict
    ind_strict = _generic_check._check_var(
        ind_strict, 'ind_strict',
        types=bool,
        default=True,
    )

    # key
    lkok = list(ddata.keys()) + [None]
    key = _generic_check._check_var(
        key, 'key',
        allowed=lkok,
    )

    # ref
    lkok = list(dref.keys()) + [None]
    ref = _generic_check._check_var(
        ref, 'ref',
        allowed=lkok,
    )

    if key is None and ref is None:
        msg = "Please provide key or ref at least!"
        raise Exception(msg)

    # ------------------------
    # hasref, hasvect

    hasref = None
    if ref is not None and key is not None:
        hasref = ref in ddata[key]['ref']
    elif ref is not None:
        hasref = True

    if hasref is True:
        refok = (ref,)
    elif key is not None:
        refok = ddata[key]['ref']

    # identify possible vect
    if hasref is not False:
        lp = [('dim', dim), ('quant', quant), ('name', name), ('units', units)]
        lk_vect = [
            k0 for k0, v0 in ddata.items()
            if v0['monot'] == (True,)
            and v0['ref'][0] in refok
            and all([
                (vv is None)
                or (vv is not None and v0[ss] == vv)
                for ss, vv in lp
            ])
        ]

        # cases
        if len(lk_vect) == 0:
            msg = "No matching vector found!"
            warnings.warn(msg)
            hasvect = False

        elif len(lk_vect) == 1:
            hasvect = True
            key_vector = lk_vect[0]
            if hasref is True:
                assert ref == ddata[key_vector]['ref'][0]
            else:
                ref = ddata[key_vector]['ref'][0]
                hasref = True

        else:
            msg = (
                f"Multiple possible vectors found:\n{lk_vect}"
            )
            warnings.warn(msg)
            hasvect = False
    else:
        hasvect = False

    # set hasref if not yet set
    if hasvect is False:
        key_vector = None
        if hasref is None:
            hasref = False
            ref = None

    # consistencu check
    assert hasref == (ref is not None)
    assert hasvect == (key_vector is not None)

    # nref
    if hasref:
        nref = dref[ref]['size']
    else:
        nref = None

    # -----------------
    # values vs indices

    dind = _get_ref_vector_values(
        dref=dref,
        ddata=ddata,
        hasref=hasref,
        hasvect=hasvect,
        ref=ref,
        nref=nref,
        key_vector=key_vector,
        values=values,
        indices=indices,
    )

    # val
    if dind is None:
        if key_vector is not None:
            val = ddata[key_vector]['data']
        else:
            val = None
    else:
        val = dind['data']

    return hasref, hasvect, ref, key_vector, val, dind


def _get_ref_vector_values(
    # ressources
    ddata=None,
    dref=None,
    # inputs
    hasref=None,
    hasvect=None,
    ref=None,
    nref=None,
    key_vector=None,
    # for extra keys
    dim=None,
    quant=None,
    name=None,
    units=None,
    # parameters
    values=None,
    indices=None,
    ind_strict=None,
):

    # -------------
    # check inputs

    # values vs indices
    if values is not None and indices is not None:
        msg = "Please provide values xor indices, not both!"
        raise Exception(msg)

    # values vs hasvect
    if values is not None and hasvect is not True:
        msg = "Arg values cannot be used if hasvect = False!"
        raise Exception(msg)

    # indices vs hasref
    if indices is not None and hasref is not True:
        msg = "Arg indices cannot be used if hasref = False!"
        raise Exception(msg)

    # trivial case
    if indices is None and values is None:
        return None

    # -------
    # indices

    # values
    if isinstance(values, str):
        lp = [('dim', dim), ('quant', quant), ('name', name), ('units', units)]
        lkok = [
            k0 for k0, v0 in ddata.items()
            if v0['monot'] == (True,)
            and all([
                (vv is None)
                or (vv is not None and v0[ss] == vv)
                for ss, vv in lp
            ])
        ]
        values = _generic_check._check_var(
            values, 'values',
            types=str,
            allowed=lkok,
        )
        key_values = values
        ref_values = ddata[key_values]['ref'][0]
        values = ddata[key_values]['data']

    elif isinstance(values, (np.ndarray, list, tuple)) or np.isscalar(values):
        values = np.atleast_1d(values).ravel()
        key_values = None
        ref_values = None

    elif values is not None:
        msg = f"Unexpected values: {values}"
        raise Exception(msg)

    else:
        key_values = None
        ref_values = None

    # values vs key_vector => indices
    if values is not None:
        if key_values is not None and key_values == key_vector:
            return None
        else:
            indices, indok = _get_ref_vector_nearest(
                ddata[key_vector]['data'],
                values,
            )
    else:
        indok = None

    # -------
    # indices

    # check
    if indices is not None:
        indices = np.atleast_1d(indices).ravel()

    if indices is not None:
        if 'bool' in indices.dtype.name:
            if indices.size != nref:
                msg = (
                    f"indt as bool must have shape ({nref},), "
                    "not {indices.shape}"
                )
                raise Exception(msg)

        elif 'int' in indices.dtype.name:
            if np.nanmax(indices) >= nref:
                msg = f"indices as int must be < {nref}\nProvided: {indices}"
                raise Exception(msg)

        else:
            msg = (
                "Arg indices must be a bool or int array of indices!\n"
                f"\t- indices.dtype: {indices.dtype}\n"
                f"\t- indices: {indices}\n"
            )
            raise Exception(msg)

        # convert to int
        if 'bool' in indices.dtype.name:
            indices = indices.nonzero()[0]

        # derive values
        if values is None:
            values = ddata[key_vector]['data'][indices]

    # -------------------
    # indtu, indt_reverse

    indr = None
    if indices is not None:

        # ind_strict
        if ind_strict is True and indok is not None:
            indices = indices[indok]
            if values is not None:
                values = values[indok]

        # indu, indr
        indu = np.unique(indices)
        if indu.size < indices.size:
            indr = np.array([indices == iu for iu in indu], dtype=bool)

    if indr is None:
        indu = None

    dind = {
        'key': key_values,
        'ref': ref_values,
        'data': values,
        'ind': indices,
        'indu': indu,
        'indr': indr,
        'indok': indok,
    }

    return dind


# #############################################################################
# #############################################################################
#           Monotonous vector - common
# #############################################################################


def get_ref_vector_common(
    # ressources
    ddata=None,
    dref=None,
    # inputs
    keys=None,
    # for selecting ref vector
    ref=None,
    dim=None,
    quant=None,
    name=None,
    units=None,
    # parameters
    values=None,
    indices=None,
    ind_strict=None,
):

    # ------------
    # check inputs

    # ind_strict
    ind_strict = _generic_check._check_var(
        ind_strict, 'ind_strict',
        types=bool,
        default=True,
    )

    # keys
    keys = _generic_check._check_var_iter(
        keys, 'keys',
        types=(list, tuple),
        types_iter=str,
        allowed=ddata.keys(),
    )

    # ------------
    # keys with hasvect

    dkeys = {}
    for ii, k0 in enumerate(keys):
        hasrefi, hasvecti, refi, key_vecti, vali, dindi = get_ref_vector(
            ddata=ddata,
            dref=dref,
            key=k0,
            ref=ref,
            dim=dim,
            quant=quant,
            name=name,
            units=units,
            values=None,
            indices=None,
        )
        if hasvecti:
            dkeys[k0] = {
                'ref': refi,
                'key_vect': key_vecti,
            }

    keys = list(dkeys.keys())

    # ------------
    # list unique ref, key_vector

    if len(keys) == 0:
        hasref = False

    elif len(keys) == 1:
        hasref = True
        key_vector = dkeys[keys[0]]['key_vect']

    else:
        hasref = True
        lrefu = list([v0['ref'] for k0, v0 in dkeys.items()])
        lkeyu = list([v0['key_vect'] for k0, v0 in dkeys.items()])

        if len(lkeyu) == 1:
            key_vector = lkeyu[0]
        else:
            key_vector = None

    # False
    if hasref is False:
        key_vector = None

    # --------
    # compute

    # common vector
    val = None
    if hasref:
        if key_vector is None:

            lv = [ddata[k0]['data'] for k0 in lkeyu]

            # bounds
            b0 = np.max([np.min(vv) for vv in lv])
            b1 = np.min([np.max(vv) for vv in lv])

            # check bounds
            if b0 >= b1:
                msg = "Non valid common vector values could be identified!"
                raise Exception(msg)

            # check if ready-made solution exists
            ld = [np.min(np.diff(vv)) for vv in lv]
            imin = np.argmin(np.abs(ld))

            if np.all((lv[imin] >= b0) & (lv[imin] <= b1)):
                # the finest vector is all included in bounds
                key_vector = lkeyu[imin]
                val = lv[imin]

            else:
                # increments
                val = np.linspace(b0, b1, int(np.ceil((b1-b0)/ld[imin])))
                key_vector = None

            # indices dict
            for k0, v0 in dkeys.items():
                ind, indok = _get_ref_vector_nearest(
                    ddata[v0['key_vect']]['data'],
                    val,
                )
                dkeys[k0]['ind'] = ind
                dkeys[k0]['indok'] = indok

            iok = np.all([v0['indok'] for v0 in dkeys.values()], axis=0)

            # adjust
            if not np.all(iok):
                if key_vector is None or ind_strict:
                    val = val[iok]
                    for k0, v0 in dkeys.items():
                        dind[k0]['ind'] = dind[k0]['ind'][iok]
                        dind[k0]['indok'] = dind[k0]['indok'][iok]
                    if key_vector is not None:
                        key_vector = None

        else:
            val = ddata[key_vector]['data']

    # ---------------------
    # add values / indices

    key_vector, val = _get_ref_vector_common_values(
        ddata=ddata,
        dref=dref,
        hasref=hasref,
        # identify
        ref=ref,
        dim=dim,
        quant=quant,
        name=name,
        units=units,
        # 
        dkeys=dkeys,
        key_vector=key_vector,
        val=val,
        # values, indices
        values=values,
        indices=indices,
        ind_strict=ind_strict,
    )

    if key_vector is not None:
        ref = ddata[key_vector]['ref'][0]
    else:
        ref = None

    return hasref, ref, key_vector, val, dkeys


def _get_ref_vector_common_values(
    ddata=None,
    dref=None,
    hasref=None,
    #
    ref=None,
    dim=None,
    quant=None,
    name=None,
    units=None,
    # 
    dkeys=None,
    key_vector=None,
    val=None,
    # values, indices
    values=None,
    indices=None,
    ind_strict=None,
):

    # ------------------
    # check values and indices

    if values is None and indices is None:
        return key_vector, val

    val_out = None
    if hasref:
        for k0, v0 in dkeys.items():
            hasrefi, hasvecti, refi, key_vecti, vali, dindi = get_ref_vector(
                ddata=ddata,
                dref=dref,
                key=k0,
                ref=ref,
                dim=dim,
                quant=quant,
                name=name,
                units=units,
                values=values,
                indices=indices,
            )

            if dindi is not None:

                # update ind
                if dkeys[k0].get('ind') is not None:
                    dkeys[k0]['ind'] = dindi['ind']
                    dkeys[k0]['indok'] = dindi['indok']

                # indu, indr
                dkeys[k0]['indu'] = np.unique(dkeys[k0]['ind'])
                dkeys[k0]['indr'] = np.array([
                    dkeys[k0]['ind'] == iu for iu in dkeys[k0]['indu']
                ])

                # val_out
                if val_out is None:
                    val_out = dindi['data']
                    key_vector = dindi['key']
                else:
                    assert val_out.size == dindi['data'].size
                    assert np.allclose(val_out, dindi['data'])

    return key_vector, val_out

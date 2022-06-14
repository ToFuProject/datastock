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
    hasref=None,
    lk_vect=None,
    # parameters
    values=None,
    indices=None,
    ind_strict=None,
):

    # values
    if values is not None:
        values = np.atleast_1d(values).ravel()

    # indices
    if indices is not None:
        indices = np.atleast_1d(indices).ravel()

    # values vs indices
    if values is not None and indices is not None:
        msg = "Please provide values xor indices, not both!"
        raise Exception(msg)

    # ind_strict
    ind_strict = _generic_check._check_var(
        ind_strict, 'ind_strict',
        types=bool,
        default=True,
    )

    # ------------------------
    # hasvector and key_vector

    # lt vs hasvector
    key_vector = None
    if len(lk_vect) == 0:
        if hasref is True:
            msg = (
                f"key '{key}' was found to have '{ref}' dimension, "
                "but no corresponding vector could be identified!"
            )
            warnings.warn(msg)
        else:
            hasref = False
            ref = None
        hasvector = False

    elif len(lk_vect) == 1:
        key_vector = lk_vect[0]
        if hasref is False:
            msg = (
                "Contradiction:\n"
                f"\t- '{key}' should not have a '{ref}' dimension\n"
                f"\t- vector '{key_vector}' identified"
            )
            raise Exception(msg)
        else:
            hasref = True
            ref = ddata[key_vector]['ref'][0]
            hasvector = True

    else:
        if hasref is False:
            msg = (
                "Contradiction:\n"
                f"\t- '{key}' should not have a '{ref}' dimension\n"
                f"\t- Several vectors identified: {lk_vect}"
            )
            raise Exception(msg)
        else:
            msg = (
                f"Several possible time vectors identified for '{key}'!\n"
                f"{lk_vect}"
            )
            raise Exception(msg)

    # ref
    if hasref:
        axis = ddata[key]['ref'].index(ref)
        nref = dref[ref]['size']

    # ------------------
    # consistency checks 

    if not hasref:
        assert ref is None

    if hasvector:
        assert hasref
        assert key_vector is not None
    else:
        assert key_vector is None

    if (values is not None or indices is not None) and not hasvector:
        msg = (
            "Contradtiction:\n"
            f"\t- '{key}' does not seem to have a '{ref}' vector\n"
            "=> Args values and indices cannot be provided"
        )
        raise Exception(msg)

    # -----------------
    # values vs indices

    # values vs key_vector => indices
    if values is not None:
        indices, indok = _get_ref_vector_nearest(
            ddata[key_vector]['data'],
            values,
        )
    else:
        indok = None

    # -------
    # indices

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

    elif values is None and key_vector is not None:
        values = ddata[key_vector]['data']

    # -------------------
    # indtu, indt_reverse

    ind_reverse = None
    if indices is not None:

        # ind_strict
        if ind_strict is True and indices is not None and indok is not None:
            values = values[indok]
            indices = indices[indok]

        # indu, ind_reverse
        indu = np.unique(indices)
        if indu.size < indices.size:
            ind_reverse = np.array(
                [indices == iu for iu in indu],
                dtype=bool,
            )

    if ind_reverse is None:
        indu = None

    return (
        hasref, hasvector,
        ref, key_vector,
        values, indices, indu, ind_reverse, indok,
    )


# #############################################################################
# #############################################################################
#           Monotonous vector - common
# #############################################################################


def get_ref_vector_common(
    # ressources
    ddata=None,
    dref=None,
    # inputs
    din=None,
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

    # ------------
    # cases

    err = False
    lref = list(set([v0[2] for v0 in din.values() if v0[0]]))

    # No key has ref dimension
    if all([not v0[0] for v0 in din.values()]):
        hasref = False
        hasvect = False

    # No key has ref vector (but at least one has ref dimension)
    elif len(lref) == 1:
        nref = dref[lref[0]]['size']
        hasref = True
        hasvect = any([v0[1] for v0 in din.values() if v0[0]])

    # all keys with ref dimension have a ref vector
    elif len(lref) > 1 and all([v0[1] for v0 in din.values() if v0[0]]):
        hasref = True
        hasvect = True

    # some only => error
    else:
        err = True

    # raise Error if needed
    if err:
        lstr = [f"\t- {k0}: {v0[2]}, {v0[3]}" for k0, v0 in din.items()]
        msg = (
            "Chosen keys cannot be used to extract a common ind / value:\n"
            "They should have either:\n"
            "\t- no ref\n"
            "\t- a unique common ref\n"
            "\t- a set of different ref, each with a vector\n"
            "Provided:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # values and indices
    if values is not None:
        values = np.atleast_1d(values).ravel()

    if indices is not None:
        indices = np.atleast_1d(indices).ravel()

    if values is not None and indices is not None:
        msg = "Please provide values xor indices, not both!"
        raise Exception(msg)

    if values is not None and not hasvect:
        msg = (
            "Arg values cannot be provided because no ref vector exists!"
        )
        raise Exception(msg)

    if indices is not None and len(lref) > 1:
        lstr = [f"\t- {k0}: {v0[2]}, {v0[3]}" for k0, v0 in din.items()]
        msg = (
            "Arg indices can only be used if there is a unique common ref!\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # --------
    # compute

    val = None

    # common vector
    if hasvect:

        din = {
            k0: v0[3]
            for k0, v0 in din.items()
            if v0[1]
        }

        # ------------
        # compute

        # get list if key_vector and of vectors
        lkv = list(set([v0 for v0 in din.values()]))
        nv, lv = 0, [ddata[lkv[0]]['data']]
        for ii in range(1, len(lkv)):
            c0 = (
                lv[nv].size == ddata[lkv[ii]]['data'].size
                and np.allclose(lv[nv], ddata[lkv[ii]]['data'])
            )
            if not c0:
                lv.append(ddata[lkv[ii]]['data'])
                nv += 1

        if len(lv) == 1:
            val = lv[0]
            ind = np.arange(0, len(val))
            dout = {
                k0: {
                    'ind': ind,
                    'key_vector': din[k0],
                }
                for k0 in din.keys()
            }

        else:

            # bounds
            b0 = np.max([np.min(vv) for vv in lv])
            b1 = np.min([np.max(vv) for vv in lv])

            if b0 >= b1:
                msg = (
                    "Non valid common vector values could be identified!"
                )
                warnings.warn(msg)
                val, dout = None, None

            else:
                # increments
                dv = np.min([np.min(np.diff(vv)) for vv in lv])
                val = np.linspace(b0, b1, int(np.ceil((b1-b0)/dv)))

                # indices
                dout = {
                    k0: _get_ref_vector_nearest(ddata[v0]['data'], val)
                    for k0, v0 in din.items()
                }

                # only keep all-valid indices
                iok = np.all(np.array([v0[1] for v0 in dout.values()]), axis=0)
                if not np.any(iok):
                    msg = (
                        "Non valid common vector values could be identified!"
                    )
                    warnings.warn(msg)
                    val, dout = None, None
                else:
                    # adjust
                    val = val[iok]
                    dout = {
                        k0: {
                            'ind': v0[0][iok],
                            'key_vector': din[k0],
                        }
                        for k0, v0 in dout.items()
                    }

        # values
        if values is not None:

            ind, indok = _get_ref_vector_nearest(val, values)

            if ind_strict:
                values = values[indok]
                ind = ind[indok]

            for k0, v0 in dout.items():
                v0['ind'] = v0['ind'][ind]

            val = values

        elif indices is not None:
            val = val[indices]
            for k0, v0 in dout.items():
                v0['ind'] = v0['ind'][indices]

    # common indices
    elif hasref:
        din = {
            k0: v0
            for k0, v0 in din.items()
            if v0[1]
        }
        ind = np.arange(0, nref)

        if indices is not None:
            ind = ind[indices]
        dout = dict.fromkeys(din.keys(), {'ind': ind, 'key_vector': None})

    else:
        dout = {}

    return hasref, hasvect, val, dout

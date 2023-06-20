# -*- coding: utf-8 -*-


# Builtin
import warnings


# common
import numpy as np


# local
from . import _generic_check


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
    warn=None,
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

    # warn
    warn = _generic_check._check_var(
        warn, 'warn',
        types=bool,
        default=True,
    )

    # ------------------------
    # hasref, hasvect

    hasref = None
    if ref is not None and key is not None:
        hasref = ref in ddata[key]['ref']
        if not hasref:
            ref = None
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

        # particular case
        if key is not None and key in lk_vect:
            lk_vect = [key]

        # cases
        if len(lk_vect) == 0:
            if warn is True:
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
            if warn is True:
                msg = (
                    f"Multiple possible vectors found:\n{lk_vect}\n"
                    f"\t- key: {key}\n"
                    f"\t- ref: {ref}\n"
                    f"\t- hasref: {hasref}\n"
                    f"\t- refok: {refok}\n"
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

    # consistency check
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
    warn=None,
):

    # -------------
    # check inputs

    # values vs indices
    if values is not None and indices is not None:
        if warn is True:
            msg = "Please provide values xor indices, not both!"
            warnings.warn(msg)
        indices = None

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

    # values = array => array or str
    elif isinstance(values, (np.ndarray, list, tuple)) or np.isscalar(values):
        values = np.atleast_1d(values).ravel()

        # Try to identify match with a known vector
        key_values = _get_ref_vector_find_identical(
            # ressources
            ddata=ddata,
            dref=dref,
            # for selecting ref vector
            ref=ref,
            dim=dim,
            quant=quant,
            name=name,
            units=units,
            # for comparison
            val=values,
        )

        # case
        if key_values is None:
            ref_values = None

        else:
            ref_values = ddata[key_values]['ref'][0]

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


def _get_ref_vector_find_identical(
    # ressources
    ddata=None,
    dref=None,
    # for selecting ref vector
    ref=None,
    dim=None,
    quant=None,
    name=None,
    units=None,
    # for comparison
    val=None
):

    # get list of all available ref vectors
    lkok = []
    for ii, k0 in enumerate(ddata.keys()):
        _, hasvecti, _, key_vecti = get_ref_vector(
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
        )[:4]
        if hasvecti and key_vecti not in lkok:
            lkok.append(key_vecti)

    # extract those which match
    lkok = [
        k0 for k0 in lkok
        if ddata[k0]['data'].size == val.size
        and np.allclose(ddata[k0]['data'], val)
    ]

    # cases
    if len(lkok) == 1:
        key = lkok[0]
    else:
        key = None
    return key


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
    # strategy for choosing common ref vector
    strategy=None,
    strategy_bounds=None,
    # parameters
    values=None,
    indices=None,
    ind_strict=None,
):
    """ Return a common ref vector for all data

    For example if several data are based on different time vectors
    This routine helps identify a common time vector according to strategy:
        - 'dt_min': picks the time vector with smallest time step
        - 'dt_max': picks the time vector with largest time step
        - key/ref:  force-pick the time vector of choice

    Additionally, stragety_bounds:
        -

    Additionally, ind_strict:
        - False: no additional constraint
        - True:

    Additionally, indices:
        - None
        -

    Additionally values:

    """

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
    refs = [v0['ref'] for v0 in dkeys.values() if v0['ref'] is not None]
    keysv = [v0['key_vect'] for v0 in dkeys.values() if v0['key_vect'] is not None]
    assert len(refs) == len(keysv)

    # strategy
    strategy = _generic_check._check_var(
        strategy, 'strategy',
        types=str,
        default='dt_min',
        allowed=['dt_min', 'dt_max'] + refs + keysv,
    )

    # strategy_bounds
    strategy_bounds = _generic_check._check_var(
        strategy_bounds, 'strategy_bounds',
        default='min' if strategy in ['dt_min', 'dt_max'] else False,
        allowed=['min', False],
    )

    # ------------
    # list unique ref, key_vector

    if len(keys) == 0:
        hasref = False

    elif len(keys) == 1:
        hasref = True
        key_vector = dkeys[keys[0]]['key_vect']

    else:
        hasref = True

        lkeyu = list(set(keysv))
        lrefu = []
        for ii, kv in enumerate(lkeyu):
            lrefu.append(refs[keysv.index(kv)])

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
    if hasref is True:
        if key_vector is None:

            lv = [ddata[k0]['data'] for k0 in lkeyu]

            # ---------------------------
            # Strategy for choosing ref / key_vector

            # check if ready-made solution exists
            ld = [np.min(np.diff(vv)) for vv in lv]
            if strategy == 'dt_min':
                i0 = np.argmin(np.abs(ld))

            elif strategy == 'dt_max':
                i0 = np.argmax(np.abs(ld))

            else:
                i0 = [
                    ii for ii, k0 in enumerate(lkeyu)
                    if k0 == strategy or lrefu[ii] == strategy
                ]
                assert len(i0) == 1
                i0 = i0[0]

            # -----------------------------
            # strategy regarding boundaries

            if strategy_bounds == 'min':

                # bounds
                b0 = np.max([np.min(vv) for vv in lv])
                b1 = np.min([np.max(vv) for vv in lv])

                # check bounds
                if b0 >= b1:
                    msg = "Non valid common vector values could be identified!"
                    raise Exception(msg)

                if np.all((lv[i0] >= b0) & (lv[i0] <= b1)):
                    # the finest vector is all included in bounds
                    key_vector = lkeyu[i0]
                    val = lv[i0]

                else:
                    # increments
                    val = np.linspace(b0, b1, int(np.ceil((b1-b0)/ld[i0]) + 2))
                    # val = lv[1]
                    key_vector = None
            else:
                key_vector = lkeyu[i0]
                val = lv[i0]

            # -------------
            # indices dict

            for k0, v0 in dkeys.items():
                ind, indok = _get_ref_vector_nearest(
                    ddata[v0['key_vect']]['data'],
                    val,
                )
                dkeys[k0]['ind'] = ind
                dkeys[k0]['indok'] = indok

            iok = np.all([v0['indok'] for v0 in dkeys.values()], axis=0)

            # -------------------------
            # adjust if ind_strict

            if not np.all(iok):
                if key_vector is None and ind_strict:
                    val = val[iok]
                    # TBC
                    # for k0, v0 in dkeys.items():
                        # dind[k0]['ind'] = dind[k0]['ind'][iok]
                        # dind[k0]['indok'] = dind[k0]['indok'][iok]
                    # if key_vector is not None:
                        # key_vector = None

            # ----------------------------------------------
            # try to identify identical pre-existing vector

            if key_vector is None:
                key_vector = _get_ref_vector_find_identical(
                    # ressources
                    ddata=ddata,
                    dref=dref,
                    # for selecting ref vector
                    ref=ref,
                    dim=dim,
                    quant=quant,
                    name=name,
                    units=units,
                    # for comparison
                    val=val,
                )

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

    if isinstance(values, str) and values == key_vector:
        return key_vector, val

    val_out = val
    haschanged = False
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
                if haschanged is False:
                    val_out = dindi['data']
                    key_vector = dindi['key']
                    haschanged = True
                else:
                    assert val_out.size == dindi['data'].size
                    assert np.allclose(val_out, dindi['data'])

    return key_vector, val_out


# #############################################################################
# #############################################################################
#               Uniformize
# #############################################################################


def _uniformize_check(
    coll=None,
    keys=None,
    refs=None,
    param=None,
    lparam=None,
    returnas=None,
):

    # -------------
    # keys vs refs

    # trivial and limit cases
    if keys is None and refs is None:
        keys = list(coll.ddata.keys())
        refs = list(coll.dref.keys())
    else:
        if isinstance(refs, str):
            refs = [refs]
        if isinstance(keys, str):
            keys = [keys]

    # check provided refs is ok
    if refs is not None:
        lok = list(coll.dref.keys())
        refs = _generic_check._check_var_iter(
            refs, 'refs',
            types=(list, tuple),
            types_iter=str,
            allowed=lok,
        )

    # check provided keys is ok
    if keys is not None:
        lok = list(coll.ddata.keys())
        keys = _generic_check._check_var_iter(
            keys, 'keys',
            types=(list, tuple),
            types_iter=str,
            allowed=lok,
        )

    # set keys if None
    if keys is None:
        keys = [
            k0 for k0, v0 in coll.ddata.items()
            if all([rr in refs for rr in v0['ref']])
        ]

    # set refs if None
    if refs is None:
        refs = list(set(np.r_[[coll.ddata[k0]['refs'] for k0 in keys]]))

    # check cross-consistency
    dkout = {
        k0: [k1 for k1 in coll.ddata[k0]['ref'] if k1 not in refs]
        for k0 in keys
        if len([k1 for k1 in coll.ddata[k0]['ref'] if k1 not in refs]) > 0
    }
    if len(dkout) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dkout.items()]
        msg = (
            "The following keys have non-specified refs:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # ------------------------------
    # parami: either a str or a dict

    # param
    param = _generic_check._check_var(
        param, 'param',
        types=str,
        allowed=['dim', 'quant', 'name', 'units'],
        default='dim',
    )

    # dparam
    # list availabe values for param and associated refs
    dparref = {
        k0: [
            v1[param] for k1, v1 in coll.ddata.items()
            if v1['monot'] == (True,)
            and v1['ref'][0] == k0
            and (lparam is None or v1[param] in lparam)
        ]
        for k0 in refs
    }

    dfail = {k0: v0 for k0, v0 in dparref.items() if len(v0) != 1}
    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = (
            "dparam cannot be inferred automatically due to ambiguities!\n"
            + "\n".join(lstr)
            + "\nPlease provide lparam explicitly!"
        )
        raise Exception(msg)

    # store dkeypar
    dkeypar = {
        k0: [dparref[rr][0] for rr in coll.ddata[k0]['ref']]
        for k0 in keys
    }

    # create dparam
    lparamu = set([v0[0] for v0 in dparref.values()])

    dparam = dict.fromkeys(lparamu)
    for k0 in lparamu:
        lr = [k1 for k1, v1 in dparref.items() if v1[0] == k0]
        dparam[k0] = {
            'refs': lr,
            'keys': [
                k1 for k1 in keys
                if any([k2 in coll.ddata[k1]['ref'] for k2 in lr])
            ],
        }

    # double-check consistency
    dfails = {}
    for k0, v0 in dparam.items():
        lout = [
            k1 for k1 in v0['keys']
            if coll.get_ref_vector(key=k1, **{param: k0})[3] is None
        ]
        if len(lout) > 0:
            dfails[k0] = lout

    if len(dfails) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfails.items()]
        msg = (
            f"For the following values of '{param}', "
            "some keys have no ref vector:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # -------------
    # others

    # returnas
    returnas = _generic_check._check_var(
        returnas, 'returnas',
        types=str,
        allowed=['datastock', 'dataframe'],
        default='datastock',
    )

    return keys, refs, param, dparam, dkeypar, returnas


def uniformize(
    coll=None,
    keys=None,
    refs=None,
    param=None,
    lparam=None,
    ref_vector_strategy=None,
    returnas=None,
):

    # ------------
    # check inputs

    keys, refs, param, dparam, dkeypar, returnas = _uniformize_check(
        coll=coll,
        keys=keys,
        refs=refs,
        param=param,
        lparam=lparam,
        returnas=returnas,
    )

    # ----------------
    # Treat ref by ref

    # instanciate
    stu = coll.__class__()

    # group refs according to param
    lp = ['dim', 'quant', 'name', 'units']
    dindnd = {}
    for k0, v0 in dparam.items():
        hasref, ref, key, val, dind = coll.get_ref_vector_common(
            keys=v0['keys'],
            strategy=ref_vector_strategy,
            **{param: k0},
        )

        # add ref
        if ref is not None:
            nref = int(coll.dref[ref]['size'])
        else:
            assert val is not None
            ref = f'n{k0}'
            nref = int(val.size)
        stu.add_ref(key=ref, size=nref)

        # add ref data
        if key is not None:
            val = np.copy(coll.ddata[key]['data'])
        else:
            key = k0
        stu.add_data(
            key=key,
            data=val,
            ref=ref,
            **{kp: str(coll.ddata[key][kp]) for kp in lp},
        )

        if dind is None:
            continue

        # add other data
        for k1, v1 in dind.items():

            # add 1d arrays
            if len(coll.ddata[k1]['ref']) == 1:

                cskip = (
                    coll.ddata[k1]['monot'] == (True,)
                    and coll.ddata[k1][param] == k0
                )
                if cskip:
                    continue

                val1 = np.copy(coll.ddata[k1]['data'])[v1['ind']]
                stu.add_data(
                    key=k1,
                    data=val1,
                    ref=ref,
                    **{kp: coll.ddata[k1][kp] for kp in lp},
                )

            else:

                # index of ref
                ref1 = coll.ddata[k1]['ref']
                iref1 = dkeypar[k1].index(k0)

                # store nd arrays
                if k1 not in dindnd.keys():
                    dindnd[k1] = {
                        'ref': [None for ii in ref1],
                        'data': np.copy(coll.ddata[k1]['data'])
                    }

                dindnd[k1]['ref'][iref1] = ref
                if iref1 == 0:
                    dindnd[k1]['data'] = dindnd[k1]['data'][dind[k1]['ind'], ...]
                elif iref1 == 1:
                    dindnd[k1]['data'] = dindnd[k1]['data'][:, dind[k1]['ind'], ...]
                elif iref1 == 2:
                    dindnd[k1]['data'] = dindnd[k1]['data'][:, :, dind[k1]['ind'], ...]

    # -------------
    # add nd arrays

    for k0, v0 in dindnd.items():
        stu.add_data(
            key=k0,
            data=v0['data'],
            ref=tuple(v0['ref']),
            **{kp: coll.ddata[k0][kp] for kp in lp},
        )

    # ---------
    # return

    if returnas == 'dataframe':
        pass

    return stu

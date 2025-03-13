# -*- coding: utf-8 -*-


# common
import copy
import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    '_check_var',
    '_check_var_iter',
    '_check_flat1darray',
    '_check_dict_valid_keys',
    '_check_vectbasis',
    '_obj_key',
    '_check_dax',
    '_apply_dlim',
    '_check_cmap_vminvmax',
]


_LALLOWED_AXESTYPES = [
    None,
    'cross', 'hor',
    'matrix',
    'timetrace',
    'profile1d',
    'image',
    'text',
    'misc',
]


# #############################################################################
# #############################################################################
#                           Utilities
# #############################################################################


def _complete_extra_msg(msg, extra_msg):
    if extra_msg not in [None, False, '']:
        msg += f"\n{extra_msg}"
    return msg


def _check_var(
    var,
    varname,
    types=None,
    default=None,
    allowed=None,
    excluded=None,
    sign=None,
    extra_msg=None,
):
    """ Check a variable, with options

    - check is instance of any types
    - check belongs to list of allowed values
    - check does not belong to list of excluded values

    if None:
        - set to default if provided
        - set to allowed value if only has 1 element

    Print proper error message if necessary, return the variable itself

    """

    # set to default
    if var is None:
        var = default

    if allowed is not None and not isinstance(allowed, list):
        allowed = list(allowed)

    if allowed is not None:
        if var is None and len(allowed) == 1:
            var = allowed[0]
        elif var not in allowed:
            msg = (
                f"Arg {varname} not in allowed range!\n"
                f"Provided: {var}\n"
                f"Allowed: {allowed}\n"
            )
            raise Exception(_complete_extra_msg(msg, extra_msg))

    # check type
    if types is not None:
        if not isinstance(var, types):
            msg = (
                f"Arg {varname} must be of type {types}!\n"
                f"Provided: {type(var)}"
            )
            raise Exception(_complete_extra_msg(msg, extra_msg))

    # check if excluded
    if excluded is not None:
        if var in excluded:
            msg = (
                f"Arg {varname} must not be in excluded range!\n"
                f"Provided: {var}\n"
                f"Excluded: {excluded}\n"
            )
            raise Exception(_complete_extra_msg(msg, extra_msg))

    # sign
    if sign is not None:
        err = False

        if isinstance(sign, str):
            sign = [sign]

        if np.isscalar(var):
            for ss in sign:
                if not eval(f'var {ss}'):
                    err = True
                    break
        else:
            vv = np.asarray(var)
            for ss in sign:
                if not np.all(eval(f'vv {ss}')):
                    err = True
                    break

        if err is True:
            msg = (
                f"Arg {varname} must be of sign {sign}\n"
                f"Provided: {var}"
            )
            raise Exception(_complete_extra_msg(msg, extra_msg))

    return var


def _check_var_iter(
    var,
    varname,
    types=None,
    types_iter=None,
    default=None,
    size=None,
    allowed=None,
    excluded=None,
    extra_msg=None,
):
    """ Check a variable supposed to be an iterable, with options

    - check is instance of any types
    - check each element is instance of types_iter
    - check each element belongs to list of allowed values
    - check each element does not belong to list of excluded values

    if var is not iterable, turned into a list of itself

    if None:
        - set to default if provided
        - set to list of allowed if provided

    Print proper error message if necessary, return the variable itself

    """

    # set to default
    if var is None:
        var = default
    if var is None and allowed is not None:
        var = allowed

    if var is not None and not hasattr(var, '__iter__'):
        var = [var]

    # check type
    if types is not None:
        if not isinstance(var, types):
            msg = (
                f"Arg {varname} must be of type {types}!\n"
                f"Provided: {type(var)}"
            )
            raise Exception(_complete_extra_msg(msg, extra_msg))

    # check types_iter
    if types_iter is not None and var is not None:
        if not all([isinstance(vv, types_iter) for vv in var]):
            msg = (
                f"Arg {varname} must be an iterable of types {types_iter}\n"
                f"Provided: {[type(vv) for vv in var]}"
            )
            raise Exception(_complete_extra_msg(msg, extra_msg))

    # check size
    if size is not None:
        if len(var) != size:
            msg = (
                f"Arg {varname} must be an iterable of len = {size}\n"
                f"Provided: {len(var)}"
            )
            raise Exception(_complete_extra_msg(msg, extra_msg))

    # check if allowed
    if allowed is not None:
        if any([vv not in allowed for vv in var]):
            msg = (
                f"Arg {varname} must contain elements in {allowed}!\n"
                f"Provided: {var}"
            )
            raise Exception(_complete_extra_msg(msg, extra_msg))

    # check if excluded
    if excluded is not None:
        if any([vv in excluded for vv in var]):
            msg = (
                f"Arg {varname} must contain elements not in {excluded}!\n"
                f"Provided: {var}"
            )
            raise Exception(_complete_extra_msg(msg, extra_msg))

    return var


def _check_flat1darray(
    var=None,
    varname=None,
    dtype=None,
    size=None,
    sign=None,
    norm=None,
    unique=None,
    can_be_None=None,
    extra_msg=None,
):

    # Default inputs
    if norm is None:
        norm = False

    # can_be_None
    if can_be_None is None:
        can_be_None = False

    # Format to flat 1d array and check size
    if var is None:
        if can_be_None is True:
            return
        else:
            msg = f"Arg {varname} is None!"
            raise Exception(_complete_extra_msg(msg, extra_msg))

    var = np.atleast_1d(var).ravel()

    # unique
    if unique is True:
        if not np.allclose(var, np.unique(var)):
            msg = (
                f"Arg {varname} must be a sorted array of unique values!\n"
                f"Provided: {var}"
            )
            raise Exception(_complete_extra_msg(msg, extra_msg))

    # size
    if size is not None:
        if np.isscalar(size):
            size = [size]
        if var.size not in size:
            msg = (
                f"Arg {varname} should be a 1d np.ndarray with:\n"
                f"\t- size = {size}\n"
                f"\t- dtype = {dtype}\n"
                f"Provided:\n{var}"
            )
            raise Exception(_complete_extra_msg(msg, extra_msg))

    # dtype
    if dtype is not None:
        var = var.astype(dtype)

    # sign
    if sign is not None:

        if isinstance(sign, str):
            sign = [sign]

        for ss in sign:
            if not np.all(eval(f'var {ss}')):
                msg = (
                    f"Arg {varname} must be {ss}\n"
                    f"Provided: {var}"
                )
                raise Exception(_complete_extra_msg(msg, extra_msg))

    # Normalize
    if norm is True:
        var = var / np.linalg.norm(var)

    return var


# ##################################################################
# ##################################################################
#               Utilities for checking dict
# ##################################################################


def _check_dict_valid_keys(
    var=None,
    varname=None,
    dkeys=None,
    has_all_keys=None,
    has_only_keys=None,
    keys_can_be_None=None,
    return_copy=None,
):
    """ Check dict has expected keys """

    # check type
    if not isinstance(var, dict):
        msg = f"Arg {varname} must be a dict!\nProvided: {type(var)}"
        raise Exception(msg)

    # copy
    if return_copy is True:
        var = copy.deepcopy(var)

    # derive lkeys
    if isinstance(dkeys, dict):
        lkeys = list(dkeys.keys())
    else:
        lkeys = dkeys

    # has_all_keys
    if has_all_keys is True:
        if not all([k0 in var.keys() for k0 in lkeys]):
            msg = (
                f"Arg {varname} should have all keys:\n{sorted(lkeys)}\n"
                f"Provided:\n{sorted(var.keys())}"
            )
            raise Exception(msg)

    # has_only_keys
    if has_only_keys is True:
        if not all([k0 in lkeys for k0 in var.keys()]):
            msg = (
                f"Arg {varname} should have only keys:\n{sorted(lkeys)}\n"
                f"Provided:\n{sorted(var.keys())}"
            )
            raise Exception(msg)

    # keys types constraints
    lkarray = ['dtype', 'size']
    if isinstance(dkeys, dict):

        if keys_can_be_None is not None:
            for k0, v0 in dkeys.items():
                dkeys[k0]['can_be_None'] = keys_can_be_None

        for k0, v0 in dkeys.items():

            # policy vs None
            if v0.get('can_be_None', False) is True:
                if var.get(k0) is None:
                    var[k0] = None
                    continue

            vv = var.get(k0)

            # routine to call
            if any([ss in v0.keys() for ss in lkarray]):
                var[k0] = _check_flat1darray(
                    var.get(k0),
                    f"{varname}['{k0}']",
                    **v0,
                )

            else:
                if 'can_be_None' in v0:
                    del v0['can_be_None']

                if any(['iter' in ss for ss in v0.keys()]):
                    var[k0] = _check_var_iter(
                        var.get(k0),
                        f"{varname}['{k0}']",
                        **v0,
                    )

                else:
                    var[k0] = _check_var(
                        var.get(k0),
                        f"{varname}['{k0}']",
                        **v0,
                    )

    return var


# #############################################################################
# #############################################################################
#                   Utilities for vector basis
# #############################################################################


def _get_horizontal_unitvect(ee=None):
    if np.abs(ee[2]) < 1. - 1e-10:
        eout = np.r_[ee[1], -ee[0], 0]
    else:
        eout = np.r_[1, 0, 0.]
    return _check_flat1darray(eout, 'eout', size=3, dtype=float, norm=True)


def _get_vertical_unitvect(ee=None):
    if np.abs(ee[2]) < 1. - 1e-10:
        eh = np.sum(ee[:2]**2)
        eout = np.r_[-ee[2]*ee[0], -ee[2]*ee[1], eh]
    else:
        eout = _get_horizontal_unitvect(ee=ee)
    return _check_flat1darray(eout, 'eout', size=3, dtype=float, norm=True)


def _check_vectbasis(
    e0=None,
    e1=None,
    e2=None,
    dim=None,
    tol=None,
):

    # dim
    dim = _check_var(dim, 'dim', types=int, default=3, allowed=[2, 3])

    # tol
    tol = _check_var(tol, 'tol', types=float, default=1.e-14, sign='>0.')

    # check is provided
    if e0 is not None:
        e0 = _check_flat1darray(e0, 'e0', size=dim, dtype=float, norm=True)
    if e1 is not None:
        e1 = _check_flat1darray(e1, 'e1', size=dim, dtype=float, norm=True)
    if e2 is not None:
        e2 = _check_flat1darray(e2, 'e2', size=dim, dtype=float, norm=True)

    # vectors
    if dim == 2:

        if e0 is None and e1 is None:
            msg = "Please provide e0 and/or e1!"
            raise Exception(msg)

        # complete if missing
        if e0 is None:
            e0 = np.r_[e1[1], -e1[0]]
        if e1 is None:
            e1 = np.r_[-e0[1], e0[0]]

        # perpendicularity
        if np.abs(np.sum(e0*e1)) > tol:
            msg = "Non-perpendicular"
            raise Exception(msg)

        # direct
        if np.abs(np.cross(e0, e1).tolist() - 1.) < tol:
            msg = "Non-direct basis"
            raise Exception(msg)

        return e0, e1

    else:
        if e0 is None and e1 is None and e2 is None:
            msg = "Please provide at least e0, e1 or e2!"
            raise Exception(msg)

        # complete if 2 missing
        if e0 is None and e1 is None:
            e1 = _get_horizontal_unitvect(ee=e2)
        elif e0 is None and e2 is None:
            e2 = _get_vertical_unitvect(ee=e1)
        elif e1 is None and e2 is None:
            e2 = _get_vertical_unitvect(ee=e0)

        # complete if 1 missing
        if e0 is None:
            e0 = np.cross(e1, e2)
            e0 = _check_flat1darray(e0, 'e0', size=dim, dtype=float, norm=True)
        if e1 is None:
            e1 = np.cross(e2, e0)
            e1 = _check_flat1darray(e1, 'e1', size=dim, dtype=float, norm=True)
        if e2 is None:
            e2 = np.cross(e0, e1)
            e2 = _check_flat1darray(e2, 'e2', size=dim, dtype=float, norm=True)

        # perpendicularity
        lv = [
            (('e0', 'e1'), (e0, e1)),
            (('e0', 'e2'), (e0, e2)),
            (('e1', 'e2'), (e1, e2)),
        ]
        dperp = {
            f'{eis}.{ejs}': np.abs(np.sum(ei*ej))
            for (eis, ejs), (ei, ej) in lv
            if np.abs(np.sum(ei*ej)) > tol
        }
        if len(dperp) > 0:
            lstr = [f"\t- {k0}: {v0}" for k0, v0 in dperp.items()]
            msg = "Non-perpendicular vectors:\n" + "\n".join(lstr)
            raise Exception(msg)

        # direct
        if not np.allclose(np.cross(e0, e1), e2, atol=tol, rtol=1e-6):
            msg = "Non-direct basis"
            raise Exception(msg)

        return e0, e1, e2


# #############################################################################
# #############################################################################
#                   Utilities for naming keys
# #############################################################################


def _obj_key(d0=None, short=None, key=None, ndigits=None):

    # check input
    ndigits = _check_var(
        ndigits, 'ndigits',
        types=int,
        default=2,
        sign='>0',
    )

    # get key
    lout = list(d0.keys())
    if key is None:
        if len(lout) == 0:
            nb = 0
        else:
            lnb = [
                int(k0[len(short):]) for k0 in lout if k0.startswith(short)
                and k0[len(short):].isnumeric()
            ]
            if len(lnb) == 0:
                nb = 0
            else:
                nb = min([ii for ii in range(max(lnb)+2) if ii not in lnb])
        key = f'{short}{nb:0{ndigits}.0f}'

    return _check_var(
        key, 'key',
        types=str,
        excluded=lout,
    )


# #############################################################################
# #############################################################################
#                   Utilities for plotting
# #############################################################################


def _check_all_broadcastable(
    return_full_arrays=None,
    **kwdargs,
):

    # -------------------
    # return_full_arrays
    # -------------------

    return_full_arrays = _check_var(
        return_full_arrays, 'return_full_arrays',
        types=bool,
        default=False,
    )

    # -------------------
    # Preliminary check
    # -------------------

    dout = {}
    dfail = {}
    for k0, v0 in kwdargs.items():
        try:
            dout[k0] = np.atleast_1d(v0)
        except Exception:
            dfail[k0] = f"Not convertible to np.ndarray! - {v0}"

    # Raise Exception
    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = (
            "The following kwdargs are non-conform:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # -------------------
    # check ndim
    # -------------------

    dndim = {k0: v0.ndim for k0, v0 in dout.items() if v0.shape != (1,)}
    lndim = list(set(dndim.values()))

    if len(lndim) == 0:
        # all scalar
        if return_full_arrays:
            return dout, (1,)
        else:
            return {k0: v0[0] for k0, v0 in dout.items()}, None

    elif len(lndim) == 1:
        ndim = lndim[0]

    else:
        lstr = [f"-t {k0}: {v0}" for k0, v0 in dndim.items()]
        msg = (
            "Some keyword args have non-compatible dimensions:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # -------------------
    # check shapes
    # -------------------

    dfail = {}
    shapef = np.ones((ndim,), dtype=int)
    for k0, v0 in dout.items():

        if v0.shape == (1,):
            continue

        for ii in range(ndim):
            if v0.shape[ii] == 1:
                pass
            elif shapef[ii] == 1:
                shapef[ii] = v0.shape[ii]
            elif v0.shape[ii] == shapef[ii]:
                pass
            else:
                dfail[k0] = f"Non-compatible shape = {v0.shape} (ii = {ii})"
                continue

    shapef = tuple(shapef)

    # raise Exception if needed
    if len(dfail) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = (
            "The following keywords args have non-compatible shape:\n"
            + "\n".join(lstr)
            + f"\nReference shape: {shapef}\n"
        )
        raise Exception(msg)

    # -------------------
    # reshape output
    # -------------------

    if return_full_arrays is True:
        for k0, v0 in dout.items():
            if v0.shape == (1,):
                dout[k0] = np.full(shapef, v0[0])
            elif v0.shape != shapef:
                dout[k0] = np.broadcast_to(v0, shapef)

    else:
        for k0, v0 in dout.items():
            if v0.shape == (1,):
                dout[k0] = v0[0]

    return dout, shapef


# #############################################################################
# #############################################################################
#                   Utilities for plotting
# #############################################################################

# DEPRECATED
# def _check_inplace(coll=None, keys=None):
    # """ Check key to data and inplace """

    # # -----------------------------
    # # keys of data to be extracted
    # # ----------------------------

    # if isinstance(keys, str):
        # keys = [keys]
    # keys = _check_var_iter(
        # keys, 'keys',
        # default=None,
        # types=list,
        # types_iter=str,
        # allowed=list(coll.ddata.keys()),
    # )

    # # ----------------------
    # # extract sub-collection
    # # ----------------------

    # lk0 = list(keys)
    # for key in keys:

        # # Include all data matching any single ref
        # for rr in coll._ddata[key]['ref']:
            # for k0, v0 in coll._ddata.items():
                # if v0['ref'] == (rr,):
                    # if k0 not in lk0:
                        # lk0.append(k0)

        # # include all data matching all refs
        # for k0, v0 in coll._ddata.items():
            # if v0['ref'] == coll._ddata[key]['ref']:
                # if k0 not in lk0:
                    # lk0.append(k0)

    # coll2 = coll.extract(lk0)

    # return keys, coll2


def _check_dax(dax=None, main=None):

    # ---------
    # trivial

    if dax is None:
        return dax

    # --------------
    # if axes handle

    if issubclass(dax.__class__, plt.Axes):
        if main is None:
            msg = (
            )
            raise Exception(msg)
        else:
            dax = {main: {'handle': dax, 'type': main}}

    # -------------
    # check is dict

    c0 = (
        isinstance(dax, dict)
        and all([
            isinstance(k0, str)
            and (
                (
                    issubclass(v0.__class__, plt.Axes)
                )
                or (
                    isinstance(v0, dict)
                    and issubclass(v0.get('handle').__class__, plt.Axes)
                    # and v0.get('type') in _LALLOWED_AXESTYPES
                )
            )
            for k0, v0 in dax.items()
        ])
    )
    if not c0:
        msg = (
            "Wrong dax:\n"
            "Should be a dict of plt.Axes or of dict of plt.Axes in 'handle'\n"
        )
        if isinstance(dax, dict):
            lstr = [f"\t- '{k0}': {v0}" for k0, v0 in dax.items()]
            msg += "\n" + "\n".join(lstr)
        else:
            msg += f"{dax}"
        raise Exception(msg)

    # -----------------------------------
    # make sure handle and type are there

    for k0, v0 in dax.items():

        if issubclass(v0.__class__, plt.Axes):
            dax[k0] = {'handle': v0, 'type': [k0]}

        if isinstance(v0, dict):
            if v0.get('type') is None:
                dax[k0]['type'] = [k0]

        # make sure type is a list
        if isinstance(dax[k0]['type'], str):
            dax[k0]['type'] = [dax[k0]['type']]

    return dax


# #############################################################################
# #############################################################################
#                   Utilities for setting limits
# #############################################################################


def _check_lim(lim):

    # -----------------------------------
    # if single lim interval => into list

    c0 = (
        isinstance(lim, tuple)
        or (
            isinstance(lim, list)
            and len(lim) == 2
            and all([ll is None or np.isscalar(ll) for ll in lim])
        )
    )
    if c0:
        lim = [lim]

    # ---------------------------------------------------------
    # check lim is a list of list/tuple intervals of len() == 2

    c0 = (
        isinstance(lim, list)
        and all([
            isinstance(ll, (list, tuple))
            and len(ll) == 2
            and all([
                lll is None or np.isscalar(lll)
                for lll in ll
            ])
            for ll in lim
        ])
    )
    if not c0:
        msg = (
            "lim must be a list of list/tuple intervals of len() == 2\n"
            "\t- Provided: {lim}"
        )
        raise Exception(msg)

    # ------------------------------
    # check each interval is ordered

    dfail = {}
    for ii, ll in enumerate(lim):
        if ll[0] is not None and ll[1] is not None:
            if ll[0] >= ll[1]:
                dfail[ii] = f"{ll[0]} >= ll[1]"

    if len(dfail) > 0:
        lstr = [f"\t- lim[{ii}]: {vv}" for ii, vv in dfail.items()]
        msg = (
            "The following non-conformities in lim have been identified:\n"*
            + "\n".join(lstr)
        )
        raise Exception(msg)

    return lim


def _apply_lim(lim=None, data=None, logic=None):

    # ------------
    # check inputs

    logic = _check_var(
        logic, 'logic',
        types=str,
        default='all',
        allowed=['any', 'all', 'raw']
    )

    # lim
    lim = _check_lim(lim)

    # -------------
    # apply limits

    nlim = len(lim)
    shape = tuple(np.r_[nlim, data.shape])
    ind = np.ones(shape, dtype=bool)
    for ii in range(nlim):
        if isinstance(lim[ii], (list, tuple)):

            if lim[ii][0] is not None:
                ind[ii, ...] &= (data >= lim[ii][0])
            if lim[ii][1] is not None:
                ind[ii, ...] &= (data < lim[ii][1])

            if isinstance(lim[ii], tuple):
                ind[ii, ...] = ~ind[ii, ...]
        else:
            msg = "Unknown lim type!"
            raise Exception(msg)

    # -------------
    # apply logic

    if logic == 'all':
        ind = np.all(ind, axis=0)
    elif logic == 'any':
        ind = np.any(ind, axis=0)
    else:
        pass

    return ind



def _apply_dlim(dlim=None, logic_intervals=None, logic=None, ddata=None):

    # ------------
    # check inputs

    logic = _check_var(
        logic, 'logic',
        types=str,
        default='all',
        allowed=['any', 'all', 'raw']
    )

    # raw not accessible in this case
    logic_intervals = _check_var(
        logic_intervals, 'logic_intervals',
        types=str,
        default='all',
        allowed=['any', 'all']
    )

    # dlim
    c0 = (
        isinstance(dlim, dict)
        and all([
            k0 in ddata.keys()
            and isinstance(v0, (list, tuple))
            for k0, v0 in dlim.items()
        ])
    )
    if not c0:
        msg = (
            "Arg dlim must be a dict of the form:\n"
            "\t- {k0: [lim0, lim1], ...}\n"
            "  or\n"
            "\t- {k0: [[lim0, lim1], (lim2, lim3)], ...}\n"
            "  where k0 is a valid key to ddata\n"
            + f"Provided:\n{dlim}"
        )
        raise Exception(msg)

    # data shape
    dreshape = {}
    datashapes = list(set([ddata[k0]['data'].shape for k0 in dlim.keys()]))
    if len(datashapes) > 1:
        ndim = [len(dd) for dd in datashapes]
        shape = datashapes[np.argmax(ndim)]
        dfail = {
            k0: ddata[k0]['data'].shape
            for k0, v0 in dlim.items()
            if ddata[k0]['data'].shape != shape
            and not (
                ddata[k0]['data'].shape
                == tuple(aa for aa in shape if aa in ddata[k0]['data'].shape)
            )
        }
        if len(dfail) > 0:
            lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
            msg = (
                "The following keys have non-compatible shapes:\n"
            )
            raise Exception(msg)

        # prepare dict of reshape
        dreshape = {
            k0: tuple([
                aa if aa in ddata[k0]['data'].shape else 1
                for ii, aa in enumerate(shape)
            ])
            for k0, v0 in dlim.items()
            if ddata[k0]['data'].shape != shape
        }
    else:
        shape = datashapes[0]

    # ------------
    # compute

    # trivial case
    if len(dlim) == 0:
        return np.ones(shape, dtype=bool)

    # non-trivial
    nlim = len(dlim)
    shape = tuple(np.r_[nlim, shape])
    ind = np.zeros(shape, dtype=bool)
    for ii, (k0, v0) in enumerate(dlim.items()):
        if k0 in dreshape.keys():
            ind[ii, ...] = _apply_lim(
                lim=v0,
                data=ddata[k0]['data'].reshape(dreshape[k0]),
                logic=logic_intervals,
            )
        else:
            ind[ii, ...] = _apply_lim(
                lim=v0,
                data=ddata[k0]['data'],
                logic=logic_intervals,
            )

    # -------------
    # apply logic

    if logic == 'all':
        ind = np.all(ind, axis=0)
    elif logic == 'any':
        ind = np.any(ind, axis=0)
    else:
        pass

    return ind


# #############################################################################
# #############################################################################
#                   check cmap, vmin, vmax
# #############################################################################


def _check_cmap_vminvmax(data=None, cmap=None, vmin=None, vmax=None):
    # cmap
    c0 = (
        cmap is None
        or vmin is None
        or vmax is None
    )
    if cmap is None or vmin is None or vmax is None:
        nanmax = np.nanmax(data)
        nanmin = np.nanmin(data)
        diverging = nanmin * nanmax < 0

    if cmap is None:
        if diverging:
            cmap = 'seismic'
        else:
            cmap = 'viridis'

    # vmin, vmax
    if vmin is None:
        if diverging:
            vmin = -max(abs(nanmin), nanmax)
        else:
            vmin = nanmin
    if vmax is None:
        if diverging:
            vmax = max(abs(nanmin), nanmax)
        else:
            vmax = nanmax

    return cmap, vmin, vmax

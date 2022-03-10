# -*- coding: utf-8 -*-


# common
import numpy as np
import matplotlib.pyplot as plt


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


def _check_var(
    var,
    varname,
    types=None,
    default=None,
    allowed=None,
    excluded=None,
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
    if var is None and allowed is not None and len(allowed) == 1:
        if not isinstance(allowed, list):
            allowed = list(allowed)
        var = allowed[0]

    # check type
    if types is not None:
        if not isinstance(var, types):
            msg = (
                f"Arg {varname} must be of type {types}!\n"
                f"Provided: {type(var)}"
            )
            raise Exception(msg)

    # check if allowed
    if allowed is not None:
        if var not in allowed:
            msg = (
                f"Arg {varname} must be in {allowed}!\n"
                f"Provided: {var}"
            )
            raise Exception(msg)

    # check if excluded
    if excluded is not None:
        if var in excluded:
            msg = (
                f"Arg {varname} must not be in {excluded}!\n"
                f"Provided: {var}"
            )
            raise Exception(msg)

    return var


def _check_var_iter(
    var,
    varname,
    types=None,
    types_iter=None,
    default=None,
    allowed=None,
    excluded=None,
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
            raise Exception(msg)

    # check types_iter
    if types_iter is not None and var is not None:
        if not all([isinstance(vv, types_iter) for vv in var]):
            msg = (
                f"Arg {varname} must be an iterable of types {types_iter}\n"
                f"Provided: {[type(vv) for vv in var]}"
            )
            raise Exception(msg)

    # check if allowed
    if allowed is not None:
        if any([vv not in allowed for vv in var]):
            msg = (
                f"Arg {varname} must contain elements in {allowed}!\n"
                f"Provided: {var}"
            )
            raise Exception(msg)

    # check if excluded
    if excluded is not None:
        if any([vv in excluded for vv in var]):
            msg = (
                f"Arg {varname} must contain elements not in {excluded}!\n"
                f"Provided: {var}"
            )
            raise Exception(msg)

    return var


# #############################################################################
# #############################################################################
#                   Utilities for naming keys
# #############################################################################


def _name_key(dd=None, dd_name=None, keyroot='key'):
    """ Return existing default keys and their number as a dict

    Used to automatically iterate on dict keys

    """

    dk = {
        kk: int(kk[len(keyroot):])
        for kk in dd.keys()
        if kk.startswith(keyroot)
        and kk[len(keyroot):].isnumeric()
    }
    if len(dk) == 0:
        nmax = 0
    else:
        nmax = max([v0 for v0 in dk.values()]) + 1
    return dk, nmax


# #############################################################################
# #############################################################################
#                   Utilities for plotting
# #############################################################################


def _check_inplace(coll=None, keys=None, inplace=None):
    """ Check key to data and inplace """

    # key
    if isinstance(keys, str):
        keys = [keys]
    keys = _check_var_iter(
        keys, 'keys',
        default=None,
        types=list,
        types_iter=str,
        allowed=coll.ddata.keys(),
    )

    # inplace
    inplace = _check_var(
        inplace, 'inplace',
        types=bool,
        default=False,
    )

    # extract sub-collection of necessary
    if inplace:
        coll2 = coll
    else:
        lk0 = list(keys)
        for key in keys:

            # Include all data matching any single ref
            for rr in coll._ddata[key]['ref']:
                for k0, v0 in coll._ddata.items():
                    if v0['ref'] == (rr,):
                        if k0 not in lk0:
                            lk0.append(k0)

            # include all data matching all refs
            for k0, v0 in coll._ddata.items():
                if v0['ref'] == coll._ddata[key]['ref']:
                    if k0 not in lk0:
                        lk0.append(k0)

        coll2 = coll.extract(lk0)

    return keys, inplace, coll2


def _check_dax(dax=None, main=None):

    # None
    if dax is None:
        return dax

    # Axes
    if issubclass(dax.__class__, plt.Axes):
        if main is None:
            msg = (
            )
            raise Exception(msg)
        else:
            return {main: dax}

    # dict
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
                    and v0.get('type') in _LALLOWED_AXESTYPES
                )
            )
            for k0, v0 in dax.items()
        ])
    )
    if not c0:
        msg = (
        )
        import pdb; pdb.set_trace()     # DB
        pass
        raise Exception(msg)

    for k0, v0 in dax.items():
        if issubclass(v0.__class__, plt.Axes):
            dax[k0] = {'handle': v0, 'type': k0}
        if isinstance(v0, dict):
            dax[k0]['type'] = v0.get('type')

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

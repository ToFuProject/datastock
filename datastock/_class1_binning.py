# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 20:14:40 2023

@author: dvezinet
"""


import itertools as itt


import numpy as np
# import astropy.units as asunits
import scipy.stats as scpst


# specific
from . import _generic_check
from . import _generic_utils


# Dict of statistic <=> ufunc
_DUFUNC = {
    'sum': np.add.reduceat,
    'max': np.maximum.reduceat,
    'min': np.minimum.reduceat,
}


# ############################################################
# ############################################################
#               interpolate spectral
# ############################################################


def binning(
    coll=None,
    data=None,
    data_units=None,
    axis=None,
    # binning
    bins0=None,
    bins1=None,
    bin_data0=None,
    bin_data1=None,
    bin_units0=None,
    # kind of binning
    integrate=None,
    statistic=None,
    # options
    safety_ratio=None,
    dref_vector=None,
    ref_vector_strategy=None,
    verb=None,
    returnas=None,
    # storing
    store=None,
    store_keys=None,
):
    """ Return the binned data

    data:  the data on which to apply binning, can be
        - a list of np.ndarray to be binned
            (any dimension as long as they all have the same)
        - a list of keys to ddata items sharing the same refs

    data_units: str only necessary if data is a list of arrays

    axis: int or array of int indices
        the axis of data along which to bin
        data will be flattened along all those axis priori to binning
        If None, assumes bin_data is not variable and uses all its axis

    bins0: the bins (centers), can be
        - a 1d vector of monotonous bins
        - a int, used to compute a bins vector from max(data), min(data)

    bin_data0: the data used to compute binning indices, can be:
        - a str, key to a ddata item
        - a np.ndarray
        _ a list of any of the above if each data has different size along axis

    bin_units: str
        only used if integrate = True and bin_data is a np.ndarray

    integrate: bool
        flag indicating whether binning is used for integration
        Implies that:
            Only usable for 1d binning (axis has to be a single index)
            data is multiplied by the underlying bin_data0 step prior to binning

    statistic: str
        the statistic kwd feed to scipy.stats.binned_statistic()
        automatically set to 'sum' if integrate = True

    store: bool
        If True, will sotre the result in ddata
        Only possible if all (data, bin_data and bin) are provided as keys

    """

    # ----------
    # checks

    # keys
    (
     ddata, dbins0, dbins1, axis,
     statistic, dvariable,
     dref_vector,
     verb, store, returnas,
     ) = _check(**locals())

    # --------------
    # actual binning

    if dvariable['bin0'] is False and dvariable['bin1'] is False:

        dout = {k0: {'units': v0['units']} for k0, v0 in ddata.items()}
        for k0, v0 in ddata.items():

            # handle dbins1
            if dbins1 is None:
                bins1, vect1, bin_ref1 = None, None, None
            else:
                bins1 = dbins1['edges']
                vect1 = dbins1['data']
                bin_ref1 = dbins1[k0].get('bin_ref')

            # compute
            dout[k0]['data'], dout[k0]['ref'] = _bin_fixed_bin(
                # data to bin
                data=v0['data'],
                data_ref=v0['ref'],
                # binning quantities
                vect0=dbins0[k0]['data'],
                vect1=vect1,
                # bins
                bins0=dbins0[k0]['edges'],
                bins1=bins1,
                bin_ref0=dbins0[k0].get('bin_ref'),
                bin_ref1=bin_ref1,
                # axis
                axis=axis,
                # statistic
                statistic=statistic,
                # integration
                variable_data=dvariable['data'],
            )

    else:
        msg = (
            "Variable bin vectors not implemented yet!\n"
            f"\t- axis: {axis}\n"
            f"\t- bin_data0 variable: {dvariable['bin0']}\n"
            f"\t- bin_data1 variable: {dvariable['bin1']}\n"
        )
        raise NotImplementedError(msg)

    # --------------
    # storing

    if store is True:

        _store(
            coll=coll,
            dout=dout,
            store_keys=store_keys,
        )

    # -------------
    # return

    if returnas is True:
        return dout


# ####################################
#       check
# ####################################


def _check(
    coll=None,
    data=None,
    data_units=None,
    axis=None,
    # binning
    bins0=None,
    bins1=None,
    bin_data0=None,
    bin_data1=None,
    bin_units0=None,
    # kind of binning
    integrate=None,
    statistic=None,
    # options
    safety_ratio=None,
    dref_vector=None,
    ref_vector_strategy=None,
    verb=None,
    returnas=None,
    # storing
    store=None,
    # non-used
    **kwdargs
):

    # -----------------
    # store and verb
    # -------------------

    # verb
    verb = _generic_check._check_var(
        verb, 'verb',
        types=bool,
        default=True,
    )

    # ------------------
    # data: str vs array
    # -------------------

    ddata = _check_data(
        coll=coll,
        data=data,
        data_units=data_units,
        store=store,
    )

    ndim_data = list(ddata.values())[0]['data'].ndim

    # -----------------
    # check statistic
    # -------------------

    # statistic
    if integrate is True:
        statistic = 'sum'
    else:
        statistic = _generic_check._check_var(
            statistic, 'statistic',
            types=str,
            default='sum',
        )

    # -----------
    # bins
    # ------------

    dbins0 = _check_bins(
        coll=coll,
        lkdata=list(ddata.keys()),
        bins=bins0,
        dref_vector=dref_vector,
        store=store,
    )
    if bins1 is not None:
        dbins1 = _check_bins(
            coll=coll,
            lkdata=list(ddata.keys()),
            bins=bins1,
            dref_vector=dref_vector,
            store=store,
        )

    # -----------
    # bins
    # ------------

    # dbins0
    dbins0, variable_bin0, axis = _check_bins_data(
        coll=coll,
        axis=axis,
        ddata=ddata,
        bin_data=bin_data0,
        dbins=dbins0,
        bin_units=bin_units0,
        dref_vector=dref_vector,
        safety_ratio=safety_ratio,
        store=store,
    )

    # data vs axis
    if np.any(axis > ndim_data - 1):
        msg = f"axis too large\n{axis}"
        raise Exception(msg)

    variable_data = len(axis) < ndim_data

    # dbins1
    if bin_data1 is not None:
        dbins1, variable_bin1, axis = _check_bins_data(
            coll=coll,
            axis=axis,
            ddata=ddata,
            bin_data=bin_data1,
            dbins=dbins1,
            bin_units=None,
            dref_vector=dref_vector,
            safety_ratio=safety_ratio,
            store=store,
        )

        if variable_bin0 != variable_bin1:
            msg = "bin_data0 and bin_data1 have different shapes, todo"
            raise NotImplementedError(msg)

    else:
        dbins1 = None
        variable_bin1 = False

    # -----------------
    # check integrate
    # -------------------

    # integrate
    integrate = _generic_check._check_var(
        integrate, 'integrate',
        types=bool,
        default=False,
    )

    # safety checks
    if integrate is True:

        if bin_data1 is not None:
            msg = (
                "If integrate = True, bin_data1 must be None!\n"
                "\t- bin_data1: {bin_data1}\n"
            )
            raise Exception(msg)

        if len(axis) > 1:
            msg = (
                "If integrate is true, binning can only be done on one axis!\n"
                f"\t- axis: {axis}\n"
            )
            raise Exception(msg)


    # -----------------------
    # additional safety check

    if integrate is True:

        if variable_bin0:
            axbin = axis[0]
        else:
            axbin = 0

        for k0, v0 in ddata.items():

            ddata[k0]['units'] = v0['units'] * dbins0[k0]['units']
            if dbins0[k0]['data'].size == 0:
                continue

            dv = np.diff(dbins0[k0]['data'], axis=axbin)
            dv = np.concatenate(
                (np.take(dv, [0], axis=axbin), dv),
                axis=axbin,
            )

            # reshape
            if variable_data != variable_bin0:

                if variable_data:
                    shape_dv = np.ones((ndim_data,), dtype=int)
                    shape_dv[axis[0]] = -1
                    dv = dv.reshape(tuple(shape_dv))

                if variable_bin0:
                    raise NotImplementedError()

            ddata[k0]['data'] = v0['data'] * dv

    # --------
    # variability dict

    dvariable = {
        'data': variable_data,
        'bin0': variable_bin0,
        'bin1': variable_bin1,
    }

    # --------
    # returnas

    returnas = _generic_check._check_var(
        returnas, 'returnas',
        types=bool,
        default=store is False,
    )

    return (
        ddata, dbins0, dbins1, axis,
        statistic, dvariable,
        dref_vector,
        verb, store, returnas,
    )


def _check_data(
    coll=None,
    data=None,
    data_units=None,
    store=None,
):
    # -----------
    # store

    store = _generic_check._check_var(
        store, 'store',
        types=bool,
        default=False,
    )

    # ---------------------
    # make sure it's a list

    if isinstance(data, (np.ndarray, str)):
        data = [data]
    assert isinstance(data, list)

    # ------------------------------------------------
    # identify case: str vs array, all with same ndim

    lc = [
        all([
            isinstance(dd, str)
            and dd in coll.ddata.keys()
            and coll.ddata[dd]['data'].ndim == coll.ddata[data[0]]['data'].ndim
            for dd in data
        ]),
        all([
            isinstance(dd, np.ndarray)
            and dd.ndim == data[0].ndim
            for dd in data
        ]),
    ]

    # vs store
    if store is True:
        if not lc[0]:
            msg = "If storing, all data, bin data and bins must be declared!"
            raise Exception(msg)


    # if none => err
    if np.sum(lc) != 1:
        msg = (
            "Arg data must be a list of either:\n"
            "\t- keys to ddata with identical ref\n"
            "\t- np.ndarrays with identical shape\n"
            f"Provided:\n{data}"
        )
        raise Exception(msg)

    # --------------------
    # sort cases

    # str => keys to existing data
    if lc[0]:
        ddata = {
            k0: {
                'key': k0,
                'data': coll.ddata[k0]['data'],
                'ref': coll.ddata[k0]['ref'],
                'units': coll.ddata[k0]['units'],
            }
            for k0 in data
        }

    # arrays
    else:
        ddata = {
            ii: {
                'key': None,
                'data': data[ii],
                'ref': None,
                'units': data_units,
            }
            for ii in range(len(data))
        }

    return ddata


def _check_bins(
    coll=None,
    lkdata=None,
    bins=None,
    dref_vector=None,
    store=None,
):

    dbins = {k0: {} for k0 in lkdata}
    if np.isscalar(bins) and not isinstance(bins, str):
        bins = int(bins)

    elif isinstance(bins, str):
        lok_data = list(coll.ddata.keys())
        lok_ref = list(coll.dref.keys())
        if hasattr(coll, '_which_bins'):
            wb = coll._which_bins
            lok_bins = list(coll.dobj.get(wb, {}).keys())
        else:
            lok_bins = []

        bins = _generic_check._check_var(
            bins, 'bins',
            types=str,
            allowed=lok_data + lok_ref + lok_bins,
        )

    else:
        bins = _generic_check._check_flat1darray(
            bins, 'bins',
            dtype=float,
            unique=True,
            can_be_None=False,
        )

    # --------------
    # check vs store

    if store is True and not isinstance(bins, str):
        msg = "With store=True, bins must be keys to coll.dobj['bins'] items!"
        raise Exception(msg)

    # ----------------------------
    # compute bin edges if needed

    if isinstance(bins, str):

        if bins in lok_bins:
            for k0 in lkdata:
                dbins[k0]['bin_ref'] = coll.dobj[wb][bins]['ref']
                dbins[k0]['edges'] = coll.dobj[wb][bins]['edges']

        else:

            if bins in lok_ref:

                if dref_vector is None:
                    dref_vector = {}

                bins = coll.get_ref_vector(
                    ref=bins,
                    **dref_vector,
                )[3]
                if bins is None:
                    msg = "No ref vector identified!"
                    raise Exception(msg)

            binc = coll.ddata[bins]['data']
            for k0 in lkdata:
                dbins[k0]['bin_ref'] = coll.ddata[bins]['ref']
                dbins[k0]['edges'] = np.r_[
                    binc[0] - 0.5*(binc[1] - binc[0]),
                    0.5*(binc[1:] + binc[:-1]),
                    binc[-1] + 0.5*(binc[-1] - binc[-2]),
                ]

    else:

        for k0 in lkdata:
            bin_edges = np.r_[
                bins[0] - 0.5*(bins[1] - bins[0]),
                0.5*(bins[1:] + bins[:-1]),
                bins[-1] + 0.5*(bins[-1] - bins[-2]),
            ]

            dbins[k0]['edges'] = bin_edges

    return dbins


def _check_bins_data(
    coll=None,
    axis=None,
    ddata=None,
    bin_data=None,
    dbins=None,
    bin_units=None,
    dref_vector=None,
    store=None,
    # if bsplines
    strict=None,
    safety_ratio=None,
    deg=None,
):

    # --------------
    # options
    # --------------

    # check
    strict = _generic_check._check_var(
        strict, 'strict',
        types=bool,
        default=True,
    )

    # check
    safety_ratio = float(_generic_check._check_var(
        safety_ratio, 'safety_ratio',
        types=(int, float),
        default=1.5,
        sign='>0.'
    ))

    # -------------
    # bin_data
    # --------------

    # make list
    if isinstance(bin_data, (str, np.ndarray)):
        bin_data = [bin_data for ii in range(len(ddata))]

    # check consistency
    if not (isinstance(bin_data, list) and len(bin_data) == len(ddata)):
        msg = (
            "Arg bin_data must be a list of len() == len(data)\n"
            f"\t- type(bin_data) = {type(bin_data)}\n"
        )
        if isinstance(bin_data, list):
            msg += (
                f"\t- len(data) = {len(ddata)}\n"
                f"\t- len(bin_data) = {len(bin_data)}\n"
            )
        raise Exception(msg)

    # -------------
    # case sorting

    lok_ref = list(coll.dref.keys())
    lok_data = [k0 for k0, v0 in coll.ddata.items()]

    lok = lok_data + lok_ref
    lc = [
        all([isinstance(bb, str) and bb in lok for bb in bin_data]),
        all([isinstance(bb, np.ndarray) for bb in bin_data]),
    ]
    if np.sum(lc) != 1:
        msg = (
            "Arg bin_data must be a list of:\n"
            f"\t- np.ndarrays\n"
            f"\t- keys to coll.ddata items\n"
            f"Provided:\n{bin_data}\n"
            f"Available:\n{sorted(lok)}"
        )
        raise Exception(msg)

    # --------------
    # check vs store

    if store is True and not lc[0]:
        msg = "With store=True, all bin_data must be keys to ddata or ref"
        raise Exception(msg)

    # case with all str
    if lc[0]:

        if dref_vector is None:
            dref_vector = {}

        # derive dbins
        for ii, k0 in enumerate(ddata.keys()):

            # if ref => identify vector
            if bin_data[ii] in lok_ref:

                key_vect = coll.get_ref_vector(
                    ref=bin_data[ii],
                    **dref_vector,
                )[3]

                if key_vect is None:
                    msg = "bin_data '{bin_data[ii]}' has no reference vector!"
                    raise Exception(msg)

                bin_data[ii] = key_vect

            # fill dict
            dbins[k0].update({
                'key': bin_data[ii],
                'data': coll.ddata[bin_data[ii]]['data'],
                'ref': coll.ddata[bin_data[ii]]['ref'],
                'units': coll.ddata[bin_data[ii]]['units'],
            })

    else:
        for ii, k0 in enumerate(ddata.keys()):
            dbins[k0].update({
                'key': None,
                'data': bin_data[ii],
                'ref': None,
                'units': bin_units,
            })

    # -----------------------------------
    # check nb of dimensions consistency

    ldim = list(set([v0['data'].ndim for v0 in dbins.values()]))
    if len(ldim) > 1:
        msg = (
            "All bin_data provided must have the same nb of dimensions!\n"
            f"Provided: {ldim}"
        )
        raise Exception(msg)

    # -------------------------
    # check dimensions vs axis

    # None => set to all bin (assuming variable_bin = False)
    if axis is None:
        for k0, v0 in dbins.items():

            if ddata[k0]['ref'] is not None and v0['ref'] is not None:
                seq_data = list(ddata[k0]['ref'])
                seq_bin = v0['ref']

            else:
                seq_data = list(ddata[k0]['data'].shape)
                seq_bin = v0['data'].shape

            # get start indices of subsequence seq_bin in sequence seq_data
            laxis0 = list(_generic_utils.KnuthMorrisPratt(seq_data, seq_bin))
            if len(laxis0) != 1:
                msg = (
                    "Please specify axis, ambiguous results from ref / shape\n"
                    f"\t- data '{k0}': {seq_data}\n"
                    f"\t- bin '{v0['key']}': {seq_bin}\n"
                    f"=> laxis0 = {laxis0}\n"
                )
                raise Exception(msg)

            axisi = laxis0[0] + np.arange(0, len(seq_bin))
            if axis is None:
                axis = axisi
            else:
                assert axis == axisi

    # --------------
    # axis
    # -------------------

    axis = _generic_check._check_flat1darray(
        axis, 'axis',
        dtype=int,
        unique=True,
        can_be_None=False,
        sign='>=0',
    )

    if np.any(np.diff(axis) > 1):
        msg = f"axis must be adjacent indices!\n{axis}"
        raise Exception(msg)

    # check
    ndim_bin = ldim[0]
    if ndim_bin < len(axis):
        msg = (
            "bin_data seems to have insufficient number of dimensions!\n"
            f"\t- axis: {axis}\n"
            f"\t- ndim_bin: {ndim_bin}\n"
            f"\t- bin_data: {bin_data}"
        )
        raise Exception(msg)

    variable_bin = ndim_bin > len(axis)

    # -------------------------------
    # check vs data shape along axis

    ndim_data = list(ddata.values())[0]['data'].ndim
    variable_data = len(axis) < ndim_data
    for k0, v0 in dbins.items():

        shape_data = ddata[k0]['data'].shape
        shape_bin = v0['data'].shape

        if variable_bin == variable_data and shape_data != v0['data'].shape:
            msg = (
                "variable_bin == variable_data => shapes should be the same!\n"
                f"\t- variable_data = {variable_data}\n"
                f"\t- variable_bin = {variable_bin}\n"
                f"\t- axis = {axis}\n"
                f"\t- data '{k0}' shape = {shape_data}\n"
                f"\t- bin_data '{v0['key']}' shape = {v0['data'].shape}\n"
            )
            raise Exception(msg)

        else:
            if variable_data:
                sh_var, sh_fix = shape_data, shape_bin
            else:
                sh_fix, sh_var = shape_data, shape_bin

            shape_axis = [ss for ii, ss in enumerate(sh_var) if ii in axis]
            if sh_fix != tuple(shape_axis):
                msg = (
                    f"Wrong shapes: data '{k0}' vs bin_data '{v0['key']}'!\n"
                    f"\t- shape_data: {shape_data}\n"
                    f"\t- shape_bin: {shape_bin}\n"
                    f"\t- axis: {axis}\n"
                )
                raise Exception(msg)

    # ----------------------------------------
    # safety check on bin sizes
    # ----------------------------------------

    if len(axis) == 1:

        for k0, v0 in dbins.items():

            if variable_bin:
                raise NotImplementedError()
            else:
                dv = np.abs(np.diff(v0['data']))

            dvmean = np.mean(dv) + np.std(dv)

            if strict is True:

                lim = safety_ratio * dvmean
                db = np.mean(np.diff(dbins[k0]['edges']))
                if db < lim:
                    msg = (
                        f"Uncertain binning for bin_data '{v0['key']}':\n"
                        f"Binning steps ({db}) are < {safety_ratio} * bin_data ({lim}) step"
                    )
                    raise Exception(msg)

    return dbins, variable_bin, axis


# ####################################
# ####################################
#       binning
# ####################################


def _bin_fixed_bin(
    data=None,
    data_ref=None,
    vect0=None,
    vect1=None,
    bins0=None,
    bins1=None,
    bin_ref0=None,
    bin_ref1=None,
    axis=None,
    statistic=None,
    # integration
    variable_data=None,
):

    # ----------------------------
    # select only relevant indices

    indin = np.isfinite(vect0)
    indin[indin] = (vect0[indin] >= bins0[0]) & (vect0[indin] < bins0[-1])
    if bins1 is not None:
        indin[indin] = np.isfinite(vect1[indin])
        indin[indin] = (vect1[indin] >= bins1[0]) & (vect1[indin] < bins1[-1])

    if not variable_data:
        indin[indin] = np.isfinite(data[indin])

    # -------------
    # prepare shape

    shape_data = data.shape
    ind_other = np.arange(data.ndim)
    nomit = len(axis) - 1
    ind_other_flat = np.r_[ind_other[:axis[0]], ind_other[axis[-1]+1:] - nomit]
    ind_other = np.r_[ind_other[:axis[0]], ind_other[axis[-1]+1:]]

    shape_other = [ss for ii, ss in enumerate(shape_data) if ii not in axis]

    shape_val = list(shape_other)
    shape_val.insert(axis[0], int(bins0.size - 1))
    if bins1 is not None:
        shape_val.insert(axis[0] + 1, int(bins1.size - 1))
    val = np.zeros(shape_val, dtype=data.dtype)

    if not np.any(indin):
        return val

    # -------------
    # subset

    # vect
    vect0 = vect0[indin]
    if bins1 is not None:
        vect1 = vect1[indin]

    # data
    sli = [slice(None) for ii in shape_other]
    sli.insert(axis[0], indin)

    data = data[tuple(sli)]

    # ---------------
    # custom

    if statistic == 'sum_smooth':
        stat = 'mean'
    else:
        stat = statistic

    # ------------------
    # simple case

    if variable_data is False:

        if bins1 is None:

            # compute
            val[...] = scpst.binned_statistic(
                vect0,
                data,
                bins=bins0,
                statistic=stat,
            )[0]

        else:
            val[...] = scpst.binned_statistic_2d(
                vect0,
                vect1,
                data,
                bins=[bins0, bins1],
                statistic=stat,
            )[0]

    # -------------------------------------------------------
    # variable data, but axis = int and ufunc exists (faster)

    elif len(axis) == 1 and stat in _DUFUNC.keys() and bins1 is None:

        if statistic == 'sum_smooth':
            msg = "statistic 'sum_smooth' not properly handled here yet"
            raise NotImplementedError(msg)

        # safety check
        vect0s = np.sort(vect0)
        if not np.allclose(vect0s, vect0):
            msg = (
                "Non-sorted vect0 for binning 1d with ufunc!\n"
                f"\t- axis: {axis}\n"
                f"\t- shape_data: {shape_data}\n"
                f"\t- shape_other: {shape_other}\n"
                f"\t- shape_val: {shape_val}\n"
                f"\t- vect0.shape: {vect0.shape}\n"
                f"\t- vect0: {vect0}\n"
                f"\t- vect0s: {vect0s}\n"
            )
            raise Exception(msg)

        # get ufunc
        ufunc = _DUFUNC[stat]

        # get indices
        ind0 = np.searchsorted(
            bins0,
            vect0,
            sorter=None,
        )
        ind0[ind0 == 0] = 1

        # ind
        indu = np.unique(ind0 - 1)

        # cases
        if indu.size == 1:
            sli[axis[0]] = indu[0]
            val[tuple(sli)] = np.nansum(data, axis=axis[0])

        else:

            sli[axis[0]] = indu

            # neutralize nans
            data[np.isnan(data)] = 0.
            ind = np.r_[0, np.where(np.diff(ind0))[0] + 1]

            # sum
            val[tuple(sli)] = ufunc(data, ind, axis=axis[0])

    # -----------------------------------
    # other statistic with variable data

    else:

        # indices
        linds = [range(nn) for nn in shape_other]

        # slice_data
        sli = [0 for ii in shape_other]
        sli.insert(axis[0], slice(None))
        sli = np.array(sli)

        if bins1 is None:

            for ind in itt.product(linds):
                sli[ind_other_flat] = ind

                val[tuple(sli)] = scpst.binned_statistic(
                    vect0,
                    data[tuple(sli)],
                    bins=bins0,
                    statistic=stat,
                )[0]

                if statistic == 'sum_smooth':
                    val[tuple(sli)] *= (
                        np.nansum(data[tuple(sli)]) / np.nansum(val[tuple(sli)])
                    )

        else:

            sli_val = np.copy(sli)
            sli_val = np.insert(axis[0] + 1, slice(None))

            for ind in itt.product(linds):

                sli[ind_other_flat] = ind
                sli_val[ind_other_flat] = ind

                val[tuple(sli_val)] = scpst.binned_statistic_2d(
                    vect0,
                    vect1,
                    data[tuple(sli)],
                    bins=[bins0, bins1],
                    statistic=stat,
                )[0]

                if statistic == 'sum_smooth':
                    val[tuple(sli_val)] *= (
                        np.nansum(data[tuple(sli)]) / np.nansum(val[tuple(sli_val)])
                    )

    # ---------------
    # adjust custom

    if statistic == 'sum_smooth':
        if variable_data is False:
            val[...] *= np.nansum(data) / np.nansum(val)

    # ------------
    # references

    if data_ref is not None:
        ref = [
            rr for ii, rr in enumerate(data_ref)
            if ii not in axis
        ]

        if bin_ref0 is not None:
            bin_ref0 = bin_ref0[0]
        if bin_ref1 is not None:
            bin_ref1 = bin_ref1[0]

        ref.insert(axis[0], bin_ref0)
        if bins1 is not None:
            ref.insert(axis[0] + 1, bin_ref1)

        ref = tuple(ref)
    else:
        ref = None

    return val, ref

# #######################################################
#           Store
# #######################################################


def _store(
    coll=None,
    dout=None,
    store_keys=None,
):


    # ----------------
    # check store_keys

    if len(dout) == 1 and isinstance(store_keys, str):
        store_keys = [store_keys]

    ldef = [f"{k0}_binned" for k0 in dout.items()]
    lex = list(coll.ddata.keys())
    store_keys = _generic_check._check_var_iter(
        store_keys, 'store_keys',
        types=list,
        types_iter=str,
        default=ldef,
        excluded=lex,
    )

    # -------------
    # store

    for ii, (k0, v0) in enumerate(dout.items()):
        coll.add_data(
            key=store_keys[ii],
            data=v0['data'],
            ref=v0['ref'],
            units=v0['units'],
        )
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 20:14:40 2023

@author: dvezinet
"""


import itertools as itt


import numpy as np
import astropy.units as asunits
import scipy.stats as scpst


# specific
from . import _generic_check


# ############################################################
# ############################################################
#               interpolate spectral
# ############################################################


def binning(
    coll=None,
    data=None,
    data_units=None,
    # binning
    bins0=None,
    bins1=None,
    bin_data0=None,
    bin_data1=None,
    bin_units=None,
    # kind of binning
    integrate=None,
    statistic=None,
    # options
    safety_ratio=None,
    ref_vector_strategy=None,
):
    """ Return the binned data

    """

    # ----------
    # checks

    # keys
    ddata, dbins0, dbins1, statistic = _check(
        coll=coll,
        data=data,
        data_units=data_units,
        # binning
        bins0=bins0,
        bins1=bins1,
        bin_data0=bin_data0,
        bin_data1=bin_data1,
        bin_units=bin_units,
        # kind of binning
        integrate=integrate,
        statistic=statistic,
        safety_ratio=safety_ratio,
        strict=True,
        # options
        only1d=True,
        ref_vector_strategy=ref_vector_strategy,
    )

    # --------------
    # actual binning

    if dbins1 is None:

        for k0, v0 in dout.items():
            dout[k0]['data'] = _bin(
                bins0=dbins0['edges'],
                bins1=None if dbins1 is None else dbins1['edges'],
                vect0=dbins0['data'],
                vect1=dbins1['data'],
                data=ddata[k0]['data'],
                axis=dbins0['axis'],
            )
            ref = list(coll.ddata[k0]['ref'])
            ref[daxis[k0]] = None
            dout[k0]['ref'] = tuple(ref)

    else:

    return dout


# ####################################
#       check
# ####################################


def _check(
    coll=None,
    data=None,
    data_units=None,
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
    ref_vector_strategy=None,
):

    # ---------------
    # nb of dim

    # only1d
    only1d = _generic_check._check_var(
        only1d, 'only1d',
        types=bool,
        default=True,
    )

    maxd = 1 if only1d else 2

    # integrate
    integrate = _generic_check._check_var(
        integrate, 'integrate',
        types=bool,
        default=False,
    )

    # safety checks
    if integrate is True and bin_data1 is not None:
        msg = (
            "If integrate = True, can only provide one bin dimension!"
        )
        raise Exception(msg)

    # statistic
    if integrate is True:
        statistic = 'sum'
    else:
        statistic = _generic_check._check_var(
            statistic, 'statistic',
            types=str,
            default='sum',
        )

    # ------------------
    # data: str vs array

    # make sure it's a list
    if isinstance(data, (np.ndarray, str)):
        data = [data]
    assert isinstance(data, list)

    # identify case: str vs array
    lc = [
        all([
            isinstance(dd, str)
            and dd in coll.ddata.keys()
            and coll.ddata[dd]['ref'] == coll.ddata[data[0]]['ref']
            for dd in ddata
        ]),
        all([isinstance(dd, np.ndarray) and dd.shape == data[0].shape]),
    ]

    # if none => err
    if np.sum(lc) != 1:
        msg = (
            "Arg data must be a list of either:\n"
            "\t- keys to ddata with identical ref\n"
            "\t- np.ndarrays with identical shape\n"
            f"Provided:\n{data}"
        )
        raise Exception(msg)

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

    # -----------
    # bins

    # dbins0
    dbins0, _ = _check_bins(
        coll=coll,
        dref=dref,
        key_diag=key_diag,
        dref_cam=dref_cam,
        key_cam=key_cam,
        bin_data=bin_data0,
        bins=bins0,
        bin_units=bins_units0,
    )

    # dbins1
    if bin_data1 is not None:
        dbins1, _ = _check_bins(
            coll=coll,
            dref=dref,
            dref_cam=dref_cam,
            key_diag=key_diag,
            key_cam=key_cam,
            bin_data=bin_data1,
            bins=bins1,
        )
    else:
        dbins1 = None

    # -----------------------
    # additional safety check

    if integrate is True:

        if dbins0['data'].ndim > 1:
            msg = "Binned integration => only 1d bin_data usable"
            raise Exception(msg)

        for k0, v0 in ddata.items():
            dv = np.diff(dbins0['data'])
            dv = np.r_[dv[0], dv]

            # reshape
            if v0['data'].ndim > 1:
                shape_dv = [
                    -1 if ii == axis[0] for ii in range(v0['data'].shape)
                ]
                dv = dv.reshape(shape_dv)

            ddata[k0]['data'] = v0['data'] * dv
            ddata[k0]['units'] = v0['units'] * dbins0['units']

    return ddata, dbins0, dbins1, statistic


def _check_bins(
    coll=None,
    dref=None,
    dref_cam=None,
    key_diag=None,
    key_cam=None,
    bin_data=None,
    bins=None,
    bin_units=None,
    # if bsplines
    safety_ratio=None,
    strict=None,
    deg=None,
):

    # --------------
    # options

    # check
    strict = _generic_check._check_var(
        strict, 'strict',
        types=bool,
        default=True,
    )

    # check
    safety_ratio = _generic_check._check_var(
        safety_ratio, 'safety_ratio',
        types=(int, float),
        default=1.5,
        sign='>0.'
    )

    # --------------
    # bins

    if bins is None:
        bins = 100

    if np.isscalar(bins):
        bins = int(bins)

    else:
        bins = ds._generic_check._check_flat1d_array(
            bins, 'bins',
            dtype=float,
            unique=True,
            can_be_None=False,
        )

    # -------------
    # bin data

    if isinstance(bin_data, str):

        lquant = ['etendue', 'amin', 'amax']  # 'los'
        lcomp = ['length', 'tangency radius', 'alpha']
        llamb = ['lamb', 'lambmin', 'lambmax', 'dlamb', 'res']
        lok_fixed = ['x0', 'x1'] + lquant + lcomp + llamb

        lok_sig_static = []
        lok_sig_var = []
        for k0 in coll.dobj['diagnostic'][key_diag]['signal']:
            cams = coll.dobj['synth sig'][k0]['camera']
            ref = coll.ddata[coll.dobj['synth sig'][k0]['data'][0]]['ref']
            if ref == dref_cam[cams[0]]:
                lok_sig_static.append(k0)
            elif ref == dref[cams[0]]:
                lok_sig_var.append(k0)

        bin_key = ds._generic_check._check_var(
            bin_data, 'bin_data',
            types=str,
            allowed=lok_fixed + lok_sig_static + lok_sig_var,
        )
        bin_data = coll.ddata[bin_key]['data']
        bin_ref = coll.ddata[bin_key]['ref']
        bin_units = coll.ddata[bin_key]['units']

        variable = bin_data in lok_sig_var

    elif isinstance(bin_data, np.ndarray):
        bin_key = None
        shape = tuple([ss for ss in data_shape if ss in bin_data.shape])
        if bin_data.shape != shape:
            msg = "Arg bin_data must have "
            raise Exception(msg)

    elif bin_data is None:
        hasref, ref, bin_data, val, dkeys = coll.get_ref_vector_common(
            keys=keys,
            strategy=ref_vector_strategy,
        )
        if bin_data is None:
            msg = (
                f"No matching ref vector found for:\n"
                f"\t- keys: {keys}\n"
                f"\t- hasref: {hasref}\n"
                f"\t- ref: {ref}\n"
                f"\t- ddata['{keys[0]}']['ref'] = {coll.ddata[keys[0]]['ref']} "
            )
            raise Exception(msg)

    else:
        msg = f"Invalid bin_data:\n{bin_data}"
        raise Exception(msg)

    # ----------------
    # get axis

    if axis is None:
        if bin_key is None:
            axis = np.array([
                ii for ii, ss in enumerate(data_shape)
                if ss in bin_data.shape
            ])
        else:
            axis = np.array([
                ii for ii, rr in enumerate(data_ref)
                if rr in bin_ref
            ])

    axis = ds._generic_check._check_flat1d_array(
        axis, 'axis',
        dtype=int,
        unique=True,
        can_be_None=False,
        sign='>=0',
    )

    if np.any(axis > len(data_shape)-1):
        msg = f"axis too large\n{axis}"
        raise Exception(msg)

    if np.any(np.diff(axis) > 1):
        msg = f"axis must be adjacent indices!\n{axis}"
        raise Exception(msg)

    # --------------
    # bins

    # bins
    if isinstance(bins, int):
        bin_min = np.nanmin(bin_data)
        bin_max = np.nanmax(bin_data)
        bin_edges = np.linspace(bin_min, bin_max, bins + 1)

    else:
        bin_edges = np.r_[
            bins[0] - 0.5*(bins[1] - bins[0]),
            0.5*(bins[1:] + bins[:-1]),
            bins[-1] + 0.5*(bins[-1] - bins[-2]),
        ]

    # ----------
    # bin method

    if bin_data.ndim == 1:
        dv = np.abs(np.diff(bin_data))
        dv = np.append(dv, dv[-1])
        dvmean = np.mean(dv) + np.std(dv)

        if strict is True:
            lim = safety_ratio*dvmean
            if not np.mean(np.diff(bin_edges)) > lim:
                msg = (
                    f"Uncertain binning for '{sorted(keys)}', ref vect '{ref_key}':\n"
                    f"Binning steps ({db}) are < {safety_ratio}*ref ({lim}) vector step"
                )
                raise Exception(msg)

            else:
                npts = None

        else:
            npts = (deg + 3) * max(1, dvmean / db)

    # ----------
    # dbins

    dbins = {
        'key': bin_key,
        'bins': bins,
        'edges': edges,
        'data': bin_data,
        'ref': bin_ref,
        'units': bin_units,
        'axis': axis,
    }

    return dbins, npts


# ####################################
# ####################################
#       bin
# ####################################


def _bin(
    data=None,
    vect0=None,
    vect1=None,
    bins0=None,
    bins1=None,
    axis=None,
    # integration
    integrate=None,
    dv=None,
):

    # ----------------------------
    # select only relevant indices

    indin = np.isfinite(vect0)
    indin[indin] = (vect0[indin] >= bins[0]) & (vect0[indin] < bins[-1])

    # -------------
    # prepare shape

    shape_data = data.shape
    shape_other = [ss for ii, ss in shape_data if ii not in axis]
    shape_val = tuple(list(shape_other).insert(axis[0], int(bins.size - 1)))
    val = np.zeros(shape_val, dtype=data.dtype)

    if not np.any(indin):
        return val

    # -------------
    # subset

    # vect, dv
    vect0 = vect0[indin]

    # data
    sli = tuple([slice(None) for ii in shape_other].insert(axis[0], indin))
    data = data[sli]

    # integrate
    if integrate is True:
        if vect0.shape != data0.shape:
            # reshape dv
            shape_dv = [
                ss if ii in axis else 1
                for ii, ss in enumerate(shape_data)
            ]
            dv = dv.reshape(dvshape)

        data = data * dv[indin]

    # ------------
    # dim == 1

    if data.ndim == 1:

        if dbins1 is None:
            val = scpst.binned_statistic(
                vect,
                data,
                bins=dbins0['edges'],
                statistic=statistic,
            )[0]

        else:
            val = scpst.binned_statistic_2d(
                vect,
                data,
                bins=bins,
                statistic=statistic,
            )[0]

    # ---------------------------------
    # data dim > 1 but vect dim minimal

    elif vect.ndim == 1:

        # indices
        linds = [range(nn) for nn in shape_other]

        # get indices
        ind0 = np.searchsorted(
            bins,
            vect,
            sorter=None,
        )
        ind0[ind0 == 0] = 1
        assert np.allclose(np.unique(vect), vect)

        # ind
        indu = np.unique(ind0 - 1)

        # cases
        if indu.size == 1:
            sli[axis[0]] = indu[0]
            val[sli] = np.nansum(data, axis=axis)

        elif indu.size > 1:

            sli[axis[0]] = indu

            # neutralize nans
            data[np.isnan(data)] = 0.
            ind = np.r_[0, np.where(np.diff(ind0))[0] + 1]

            # sum
            val[sli] = np.add.reduceat(data, ind, axis=axis)

        else:
            import pdb; pdb.set_trace()     # DB
            pass

    # -----------------------------
    # data and vect have same shape

    else:
        assert data.shape == vect.shape




    return val

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
    axis=None,
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
    
    data:  the data on which to apply binning, can be
        - a list of np.ndarray to be binned
            (any dimension as long as they all have the same)
        - a list of keys to ddata items sharing the same refs
        
    data_units: str only necessary if data is a list of arrays
    
    axis: int or array of int indices
        the axis of data along which to bin
        data will be flattened along all those axis priori to binning
    
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
        pass
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
    ref_vector_strategy=None,
):

    # ------------------
    # data: str vs array
    # -------------------

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
            for dd in data
        ]),
        all([
            isinstance(dd, np.ndarray) and dd.shape == data[0].shape
            for dd in data
        ]),
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
        
    shape_data = list(ddata.values())[0]['data'].shape
        
    # --------------
    # axis
    # -------------------
    
    axis = _generic_check._check_flat1d_array(
        axis, 'axis',
        dtype=int,
        unique=True,
        can_be_None=False,
        sign='>=0',
    )
    
    if np.any(axis > len(shape_data) - 1):
        msg = f"axis too large\n{axis}"
        raise Exception(msg)
    
    if np.any(np.diff(axis) > 1):
        msg = f"axis must be adjacent indices!\n{axis}"
        raise Exception(msg)
    
    variable_data = len(axis) < shape_data

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

    # dbins0
    dbins0, _ = _check_bins(
        coll=coll,
        ddata=ddata,
        coll=coll,
        bin_data=bin_data0,
        bins=bins0,
        bin_units=bin_units0,
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
                    -1 if ii == axis[0] else 1
                    for ii in range(v0['data'].shape)
                ]
                dv = dv.reshape(shape_dv)

            ddata[k0]['data'] = v0['data'] * dv
            ddata[k0]['units'] = v0['units'] * dbins0['units']

    return ddata, dbins0, dbins1, statistic


def _check_bins(
    coll=None,
    axis=None,
    ddata=None,
    shape_data=None,
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
    # --------------

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

    # -------------
    # bin_data
    # --------------

    # make list
    if isinstance(bin_data, (str, np.ndarray)):
        bin_data = [bin_data for ii in range(len(ddata))]
    
    # check consistency
    if not (isinstance(bin_data, list) and len(bin_data) == len(ddata)):
        msg = "Arg bin_data must be a list of len() == len(data)"
        raise Exception(msg)
    
    # case sorting
    lc = [
        all([isinstance(bb, str) for bb in bin_data]),
        all([isinstance(bb, np.ndarray) for bb in bin_data]),
    ]
    if np.sum(lc) != 1:
        msg = "Arg bin_data must be a list of arrays or str matching len(data)"
        raise Exception(msg)
    
    # case with all str
    if lc[0]:

        # lquant = ['etendue', 'amin', 'amax']  # 'los'
        # lcomp = ['length', 'tangency radius', 'alpha']
        # llamb = ['lamb', 'lambmin', 'lambmax', 'dlamb', 'res']
        # lok_fixed = ['x0', 'x1'] + lquant + lcomp + llamb

        lok = list(coll.ddata.keys())
        bin_keys = _generic_check._check_var_iter(
            bin_data, 'bin_data',
            types=list,
            types_iter=str,
            allowed=lok,
        )
        
        # derive dbins
        dbins = {
            k0: {
                'key': bin_keys[ii],
                'data': coll.ddata[bin_keys[ii]]['data'],
                'ref': coll.ddata[bin_keys[ii]]['ref'],
                'units': coll.ddata[bin_keys[ii]]['units'],
            }
            for ii, k0 in enumerate(ddata.keys())
        }

    else:
        
        c0 = all([
            
        ])
        shape = tuple([ss for ii, ss in shape_data if ii in axis])
        if bin_data.shape != shape:
            msg = (
                "Arg bin_data.shape must be contained in adjacent indices  "
                "of shape_data:\n"
                f"\t- shape_bin = {bin_data.shape}\n"
                f"\t- shape_data = {shape_data}"
            )
            raise Exception(msg)
            
        # derive dbins
        dbins = {
            k0: {
                'key': None,
                'data': bin_data[ii],
                'ref': None,
                'units': bin_units,
            }
            for ii, k0 in enumerate(ddata.keys())
        }

    # --------------------------------------------
    # check non-axis shapes => should be the same
    # --------------------------------------------
    
    lshape_bin_other = [
        tuple([ss for ii, ss in enumerate(v0['data'].shape) if ii not in axis])
        for k0, v0 in dbins.items()
    ]
    
    shape_bin_otheru = list(set(lshape_bin_other))
    if len(shape_bin_otheru) != 1:
        msg = (
        )
        raise Exception(msg)
    
    # derive variable_bin
    shape_bin_other = shape_bin_otheru[0]
    variable_bin = len(shape_bin_other) > 0    

    # --------------
    # bins
    # --------------

    if bins is None:
        bins = 100

    if np.isscalar(bins):
        bins = int(bins)

    else:
        bins = _generic_check._check_flat1d_array(
            bins, 'bins',
            dtype=float,
            unique=True,
            can_be_None=False,
        )

    # bins
    for k0, v0 in dbins.items():
        if isinstance(bins, int):
            bin_min = np.nanmin(v0['data'])
            bin_max = np.nanmax(v0['data'])
            bin_edges = np.linspace(bin_min, bin_max, bins + 1)

        else:
            bin_edges = np.r_[
                bins[0] - 0.5*(bins[1] - bins[0]),
                0.5*(bins[1:] + bins[:-1]),
                bins[-1] + 0.5*(bins[-1] - bins[-2]),
            ]
            
        dbins[k0]['edges'] = bin_edges

    # ----------
    # bin method - TBF -------------------------------------------------------
    # ----------

    if integrate is True:
        
        for k0, v0 in dbins.items():
            if variable_bin:
                
            else:
                dv = np.abs(np.diff(v0['data']))
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

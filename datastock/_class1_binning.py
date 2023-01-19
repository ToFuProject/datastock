# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 20:14:40 2023

@author: dvezinet
"""


import itertools as itt


import numpy as np
import scipy.stats as scpst


# specific
from . import _generic_check


# ############################################################
# ############################################################
#               interpolate spectral
# ############################################################


def binning(
    coll=None,
    key=None,
    ref_key=None,
    bins=None,
):
    """ Return the binned data

    """
    
    # ----------
    # checks
    
    # keys
    key, ref_key, axis, units, units_ref = _binning_check_keys(
        coll=coll,
        key=key,
        ref_key=ref_key,
    )
    
    # bins
    bins, units_bins, db = _binning_check_bins(
        coll=coll,
        key=key,
        ref_key=ref_key,
        bins=bins,
        vect=coll.ddata[ref_key]['data'],
    )
    
    # units
    units = _units(
        key=key,
        units=units,
        units_ref=units_ref,
        units_bins=units_bins,
    )
        
    # ------------
    # bsplines
    
    val = _bin(
        bins=bins,
        db=db,
        vect=coll.ddata[ref_key]['data'],
        data=coll.ddata[key]['data'],
        axis=axis,
    )

    return val, units
    
    
# ####################################
#       check
# ####################################


def _binning_check_keys(
    coll=None,
    key=None,
    ref_key=None,
):
    
    # ---------
    # keys

    # key   
    key = _generic_check._check_var(
        key, 'key',
        types=str,
        allowed=list(coll.ddata.keys()),
    )
    
    # --------------
    # ref_keyor
    
    lrefv = [
        k0 for k0, v0 in coll.ddata.items()
        if v0['monot'] == (True,)
        and v0['ref'][0] in coll.ddata[key]['ref']
    ]
    lref = [
        k0 for k0, v0 in coll.dref.items()
        if k0 in coll.ddata[key]['ref']
        and coll.get_ref_vector(ref=k0)[1]
    ]
    
    if ref_key is None and len(lrefv) == 1:
        ref_key = lrefv[0]
    
    extra_msg = f"\n\nIdentify ref vector using get_ref_vector('{key}')"
    ref_key = _generic_check._check_var(
        ref_key, 'ref_key',
        types=str,
        allowed=lrefv + lref,
        extra_msg=extra_msg,
    )
    
    if ref_key in lref:
        ref_key = coll.get_ref_vector(ref=ref_key)[3]
    
    # --------------
    # axis and units
    
    # axis
    axis = coll.ddata[key]['ref'].index(coll.ddata[ref_key]['ref'][0])
    
    # units
    units = coll.ddata[key]['units']
    units_ref = coll.ddata[ref_key]['units']
    
    return key, ref_key, axis, units, units_ref


def _binning_check_bins(
    coll=None,
    key=None,
    ref_key=None,
    bins=None,
    vect=None,
):
    
    # ---------
    # bins
    
    if isinstance(bins, str):
        lok = [k0 for k0, v0 in coll.ddata.items() if v0['monot'] == (True,)]
        bins = _generic_check._check_var(
            bins, 'bins',
            types=str,
            allowed=lok,
        )

        bins_units = coll.ddata[bins]['units']        
        bins = coll.ddata[bins]['data']
        
    else:
        bins_units = None
    
    bins = _generic_check._check_flat1darray(
        bins, 'bins',
        dtype=float,
        unique=True,
        can_be_None=False,
    )
    
    # -----------------
    # check uniformity
    
    db = np.abs(np.diff(bins))
    if not np.allclose(db[0], db):
        msg = (
            "Arg bins must be a uniform bin edges vector!"
            f"Provided diff(bins) = {db}"
        )
        raise Exception(msg)
    db = db[0]
        
    # ----------
    # bin method
    
    dv = np.abs(np.mean(np.diff(vect)))
    if db < 2*dv:
        msg = (
            f"Uncertain binning for '{key}' using ref vect '{ref_key}':\n"
            f"Binning steps ({db}) are smaller than 2*ref ({2*dv}) vector step"
        )
        raise Exception(msg)
        
    return bins, bins_units, db


def _units(
    key=None,
    units=None,
    units_ref=None,
    units_bins=None,
):
    
    if units in [None, '']:
        return ''
    
    elif units_ref in [None, ''] and units_bins in [None, '']:
        return ''
    
    elif units_ref in [None, '']:
        return units * units_bins
    
    elif units_bins in [None, '']:
        return units * units_ref
    
    elif units_ref == units_bins:
        return units * units_ref
    
    else:
        msg = (
            "Units do not agree between ref vector and bins for '{key}'!\n"
            f"\t- units     : {units}\n"
            f"\t- units_ref : {units_ref}\n"
            f"\t- units_bins: {units_bins}\n"
        )
        raise Exception(msg)


# ####################################
#       bin
# ####################################


def _bin(
    bins=None,
    db=None,
    vect=None,
    data=None,
    axis=None,
):
    
    indin = (vect >= bins[0]) & (vect <= bins[-1])
    vect = vect[indin]
    
    if data.ndim == 1:
        
        data = data[indin]
        
        val = scpst.binned_statistic(
            vect,
            data,
            bins=bins,
            statistic=np.nansum,
        )[0]
        
    else:
        
        # remove out
        sli = tuple([
            indin if ii == axis else slice(None)
            for ii in range(data.ndim)
        ])
        data = data[sli]
        
        # shape
        shape = list(data.shape)
        shape[axis] = int(bins.size - 1)
        shape_other = np.r_[shape[:axis], shape[axis+1:]].astype(int)
        
        # indices
        linds = [range(nn) for nn in shape_other]
        indi = list(range(data.ndim-1))
        indi.insert(axis, None)
        
        
        # initialize val
        val = np.zeros(tuple(shape), dtype=data.dtype)
        
        for ind in itt.product(*linds):
            
            sli = tuple([
                slice(None) if ii == axis else ind[indi[ii]]
                for ii in range(len(shape))
            ])
            
            # bin            
            val[sli] = scpst.binned_statistic(
                vect,
                data[sli],
                bins=bins,
                statistic=np.nansum,
            )[0]
        
    return val * db
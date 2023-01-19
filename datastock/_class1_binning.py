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
    key_ref_vect=None,
    bins=None,
    val_out=None,
):
    """ Return the binned data

    """
    
    # ----------
    # checks
    
    # keys
    (
     key, key_ref_vect,
     axis, units, units_ref,
     ) = _binning_check_keys_options(
        coll=coll,
        key=key,
        key_ref_vect=key_ref_vect,
        val_out=val_out,
    )
    
    # bins
    bins, units_bins, db = _binning_check_bins(
        bins,
        vect=coll.ddata[key_ref_vect]['data'],
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
        vect=coll.ddata[key_ref_vect]['data'],
        data=coll.ddata[key]['data'],
        axis=axis,
    )

    return val, units
    
    
# ####################################
#       check
# ####################################


def _binning_check_keys_options(
    coll=None,
    key=None,
    key_ref_vect=None,
):
    
    # ---------
    # keys

    # key   
    key = _generic_check._check_var_iter(
        key, 'key',
        types=str,
        allowed=list(coll.ddata.keys()),
    )
    
    # --------------
    # key_ref_vector
    
    lrefv = [
        k0 for k0, v0 in coll.ddata.items()
        if v0['monot'] == (True,)
        and v0['ref'][0] in coll.ddata[key]['ref']
    ]
    
    extra_msg = "Identify ref vector using get_ref_vector(key)"
    key_ref_vect = _generic_check._check_var_iter(
        key_ref_vect, 'key_ref_vect',
        types=str,
        allowed=lrefv,
        extra_msg=extra_msg,
    )
    
    # --------------
    # axis and units
    
    # axis
    axis = coll.ddata[key]['ref'].index(coll.ddata[key_ref_vect]['ref'][0])
    
    # units
    units = coll.ddata[key]['units']
    units_ref = coll.ddata[key_ref_vect]['units']
    
    return key, key_ref_vect, axis, units, units_ref


def _binning_check_bins(
    coll=None,
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
    
    db = np.diff(bins)
    if not np.allclose(db[0], db):
        msg = (
            "Arg bins must be a uniform bin edges vector!"
            f"Provided diff(bins) = {db}"
        )
        raise Exception(msg)
        
    # ----------
    # bin method
    
    dv = np.abs(np.diff(vect))
    if dv >= db:
        msg = (
            "Uncertain binning for '{key}' using ref vect '{key_ref_vect}':\n"
            "The binning steps are smaller than the ref vector step"
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
            statistic='sum',
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
        shape[axis] = bins.size - 1
        shape_other = np.r_[shape[:axis], shape[axis+1:]]
        
        # indices
        linds = [range(nn) for nn in shape_other]
        indi = list(range(data.ndim-1))
        indi.insert(axis, None)
        
        
        # initialize val
        val = np.zeros(tuple(shape), dtype=data.dtype)
        
        for ind in itt.product(*linds):
            
            sli = [
                slice(None) if ii == axis else ind[indi[ii]]
                for ii in range(len(shape))
            ]
            
            # bin
            val[sli] = scpst.binned_statistic(
                vect,
                data[sli],
                bins=bins,
                statistic='sum',
            )[0]
        
    return val * db
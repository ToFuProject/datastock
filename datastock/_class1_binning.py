# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 20:14:40 2023

@author: dvezinet
"""


import numpy as np
import scipy.stats as scpst
import astropy.units as asunits


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
     axis, units, units_ref, val_out,
     ) = _binning_check_keys_options(
        coll=coll,
        key=key,
        key_ref_vect=key_ref_vect,
        val_out=val_out,
    )
    
    # bins
    bins, units_bins = _binning_check_bins(
        bins,
        vect=coll.ddata[key_ref_vect]['data'],
    )
    
    # units
    units = _units(units=units, units_ref=units_ref, units_bins=units_bins)
        
    # ------------
    # bsplines
    
    val = _bin(
        bins=bins,
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
    val_out=None,
):
    
    # --------------
    # key_ref_vector
    
    lok = [k0 for k0, v0 in coll.ddata.items() if v0['monot'] == (True,)]
    extra_msg = "Identify ref vector using get_ref_vector(key)"
    key_ref_vect = _generic_check._check_var_iter(
        key_ref_vect, 'key_ref_vect',
        types=str,
        allowed=lok,
        extra_msg=extra_msg,
    )
    
    # ---------
    # keys

    lok = [
        k0 for k0, v0 in coll.ddata.items()
        if coll.get_ref_vector(key=k0, warn=False)[3] == key_ref_vect
    ]

    # key   
    extra_msg = "Identify ref vector using get_ref_vector(key)"
    key = _generic_check._check_var_iter(
        key, 'key',
        types=str,
        allowed=lok,
        extra_msg=extra_msg,
    )
    
    # axis
    axis = coll.ddata[key]['ref'].index(coll.ddata[key_ref_vect]['ref'][0])
    
    # units
    units = coll.ddata[key]['units']
    units_ref = coll.ddata[key_ref_vect]['units']
    
    # --------------
    # others
    
    val_out = _generic_check._check_var(
        val_out, 'val_out',
        default=np.nan,
        allowed=[False, np.nan, 0.],
    )
    
    return key, key_ref_vect, axis, units, units_ref, val_out


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
        
    return bins, bins_units


def _units(
    units=None,
    units_ref=None,
    units_bins=None,
):
    
    
    
    return units


# ####################################
#       bin
# ####################################


def _bin(
    bins=None,
    db=None,
    vect=None,
    data=None,
    axis=None,
    bin_method=None,
    val_out=None,
):
    
    if data.ndim == 1:
        val = scpst.binned_statistic(
            vect,
            data,
            bins=bins,
            statistic='sum',
        )[0] * db
        
        indout = (vect < bins[0]) | (vect > bins[-1])
        val[indout] = val_out
        
    else:
            
        # initialize
        shape = list(data.shape)
        shape[axis] = bins.size - 1
        val = np.full(tuple(shape), val_out)
        
        # in
        indin = (vect >= bins[0]) & (vect <= bins[-1])
        
        for ind in itt.product():
            
            sli = []
            
            # bin
            val[sli_v] = scpst.binned_statistic(
                vect[indin],
                data[sli_d],
                bins=bins,
                statistic='sum',
            )[0] * db
        
    return val
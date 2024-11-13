# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np


# local
from ._class02_BSplines2D import BSplines2D as Previous
from . import _class03_checks as _checks
from . import _class03_binning as _binning


__all__ = ['Bins']


# #############################################################################
# #############################################################################
#
# #############################################################################


class Bins(Previous):

    _which_bins = 'bins'
    _ddef = copy.deepcopy(Previous._ddef)
    _dshow = dict(Previous._dshow)

    _dshow.update({
        _which_bins: [
            'nd',
            'cents',
            'shape',
            'ref',
        ],
    })

    # -----------------
    # bsplines
    # ------------------

    def add_bins(
        self,
        key=None,
        edges=None,
        # custom names
        key_ref=None,
        key_cents=None,
        key_res=None,
        # attributes
        **kwdargs,
    ):
        """ Add bin """

        # --------------
        # check inputs

        key, dref, ddata, dobj = _checks.check(
            coll=self,
            key=key,
            edges=edges,
            # custom names
            key_cents=key_cents,
            key_ref=key_ref,
            # attributes
            **kwdargs,
        )

        # --------------
        # update dict and crop if relevant

        self.update(dobj=dobj, ddata=ddata, dref=dref)

    def remove_bins(
        self,
        key=None,
        propagate=None,
    ):

        _checks.remove_bins(
            coll=self,
            key=key,
            propagate=propagate,
        )

    # -----------------
    # binning tools
    # ------------------

    def binning(
        self,
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
        verb=None,
        returnas=None,
        # storing
        store=None,
        store_keys=None,
    ):
        """ Bin data along ref_key

        Binning is treated here as an integral
        Hence, if:
            - the data has units [ph/eV]
            - the ref_key has units [eV]
            - the binned data has units [ph]

        return a dict with data and units per key

        """

        return _binning.binning(
            coll=self,
            data=data,
            data_units=data_units,
            axis=axis,
            # binning
            bins0=bins0,
            bins1=bins1,
            bin_data0=bin_data0,
            bin_data1=bin_data1,
            bin_units0=bin_units0,
            # kind of binning
            integrate=integrate,
            statistic=statistic,
            # options
            safety_ratio=safety_ratio,
            dref_vector=dref_vector,
            verb=verb,
            returnas=returnas,
            # storing
            store=store,
            store_keys=store_keys,
        )
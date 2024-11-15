# -*- coding: utf-8 -*-


# Built-in
import copy


# local
from ._class02 import DataStock2 as Previous
from . import _class03_checks as _checks
from . import _class03_bin_vs_bs as _bin_vs_bs


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
            'shape_edges',
            'edges',
            'ref_edges',
            'is_linear',
            'is_log',
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
        key_edges=None,
        key_cents=None,
        key_ref_edges=None,
        key_ref_cents=None,
        # additional attributes
        **kwdargs,
    ):
        """ Add bin

        Defined from edges, which can be:
            - np.ndarray or tuple of 2
            - key to existing monotnous array or tuple of 2

        key names are generated automatically
        But can also be specified:
            - for creation
            - or for refering to existing data

        """

        # --------------
        # check inputs

        key, dref, ddata, dobj = _checks.check(
            coll=self,
            key=key,
            edges=edges,
            # custom names
            key_edges=key_edges,
            key_cents=key_cents,
            key_ref_edges=key_ref_edges,
            key_ref_cents=key_ref_cents,
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
            - a list of any of the above if each data has diff. size along axis

        bin_units: str
            only used if integrate = True and bin_data is a np.ndarray

        integrate: bool
            flag indicating whether binning is used for integration
            Implies that:
                Only usable for 1d binning (axis has to be a single index)
                data is multiplied by bin_data0 step prior to binning

        statistic: str
            the statistic kwd feed to scipy.stats.binned_statistic()
            automatically set to 'sum' if integrate = True

        store: bool
            If True, will sotre the result in ddata
            Only possible if all (data, bin_data and bin) are provided as keys
        """

        return _bin_vs_bs.main(
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

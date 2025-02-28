# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np
import astropy.units as asunits


# library-specific
from . import _generic_check
from . import _generic_utils
from . import _class1_check
from . import _class1_show
from ._class0 import *
from . import _class1_compute
from . import _class1_domain
from . import _class1_binning
from . import _class1_interpolate
from . import _class1_uniformize
from . import _class1_color_touch as _color_touch
from . import _export_dataframe
from . import _find_plateau


#############################################
#############################################
#       Abstract Parent class
#############################################
#############################################


class DataStock1(DataStock0):
    """ A generic class for handling data

    Provides methods for:
        - introspection
        - visualization

    """
    # Fixed (class-wise) dictionary of default properties
    _ddef = {
        'params': {
            'dref': {},
            'ddata': {
                'source': {'cls': str, 'def': ''},
                'dim':    {'cls': str, 'def': ''},
                'quant':  {'cls': str, 'def': ''},
                'name':   {'cls': str, 'def': ''},
                'units':  {'cls': (str, asunits.core.UnitBase), 'def': ''},
            },
            'dobj': {},
         },
    }

    # short names
    _dshort = {
        'ref': 'n',
        'data': 'd',
    }

    # _dallowed_params = None
    _data_none = None
    _reserved_keys = None

    _max_ndim = None
    _dshow = {
        'data': [
            'shape',
            'ref',
            'dim',
            'quant',
            'name',
            'units',
            # 'source',
            # 'monot',
        ],
    }

    def __init__(
        self,
        dref=None,
        ddata=None,
        dobj=None,
    ):
        super().__init__()
        self._reset()
        self.update(
            dref=dref,
            ddata=ddata,
            dobj=dobj,
        )

    def _reset(self):
        self._dref = {}
        self._ddata = {}
        self._dobj = {}
        self.__dlinks = {}

    ###########
    # set dictionaries
    ###########

    def update(
        self,
        dobj=None,
        ddata=None,
        dref=None,
        harmonize=None,
    ):
        """ Can be used to set/add data/ref

        Will update existing attribute with new dict
        """
        # Check consistency
        (
            self._dref, self._ddata, self._dobj, self.__dlinks,
        ) = _class1_check._consistency(
            dobj=dobj, dobj0=self._dobj,
            ddata=ddata, ddata0=self._ddata,
            dref=dref, dref0=self._dref,
            reserved_keys=self._reserved_keys,
            ddefparams_data=self._ddef['params'].get('ddata'),
            ddefparams_obj=self._ddef['params'].get('dobj'),
            data_none=self._data_none,
            max_ndim=self._max_ndim,
            harmonize=harmonize,
            dshort=self._dshort,
        )

    # ---------------------
    # Adding ref / quantity one by one
    # ---------------------

    def add_ref(self, key=None, size=None, data=None, harmonize=None, **kwds):
        dref = {key: {'data': data, 'size': size, **kwds}}
        # Check consistency
        self.update(ddata=None, dref=dref, harmonize=harmonize)

    def add_data(self, key=None, data=None, ref=None, harmonize=None, **kwds):
        ddata = {key: {'data': data, 'ref': ref, **kwds}}
        # Check consistency
        self.update(ddata=ddata, dref=None, harmonize=harmonize)

    def add_obj(self, which=None, key=None, harmonize=None, **kwds):
        dobj = {which: {key: kwds}}
        # Check consistency
        self.update(dobj=dobj, dref=None, harmonize=harmonize)

        # show dict
        if which not in self._dshow.keys():
            lk = self.get_lparam(which=which, for_show=True)
            lk = [
                kk for kk in lk
                if 'func' not in kk
                and 'class' not in kk
                and kk not in ['handle']
                and not (which == 'axes' and kk == 'bck')
                and all([
                    not isinstance(v1[kk], dict)
                    for v1 in self._dobj[which].values()
                ])
            ]
            self._dshow[which] = lk

    # ---------------------
    # Removing ref / quantities
    # ---------------------

    def remove_ref(self, key=None, propagate=None):
        """ Remove a ref (or list of refs) and all associated data """
        (
            self._dref, self._ddata, self._dobj, self.__dlinks,
        ) = _class1_check._remove_ref(
            key=key,
            dref0=self._dref, ddata0=self._ddata,
            dobj0=self._dobj,
            propagate=propagate,
            reserved_keys=self._reserved_keys,
            ddefparams_data=self._ddef['params'].get('ddata'),
            ddefparams_obj=self._ddef['params'].get('dobj'),
            data_none=self._data_none,
            max_ndim=self._max_ndim,
        )

    def remove_data(self, key=None, propagate=True):
        """ Remove a data (or list of data) """
        (
            self._dref, self._ddata, self._dobj, self.__dlinks,
        ) = _class1_check._remove_data(
            key=key,
            dref0=self._dref, ddata0=self._ddata,
            dobj0=self._dobj,
            propagate=propagate,
            reserved_keys=self._reserved_keys,
            ddefparams_data=self._ddef['params'].get('ddata'),
            ddefparams_obj=self._ddef['params'].get('dobj'),
            data_none=self._data_none,
            max_ndim=self._max_ndim,
        )

    def remove_obj(self, key=None, which=None, propagate=True):
        """ Remove a data (or list of data) """
        (
            self._dref, self._ddata, self._dobj, self.__dlinks,
        ) = _class1_check._remove_obj(
            key=key,
            which=which,
            propagate=propagate,
            dobj0=self._dobj,
            ddata0=self._ddata,
            dref0=self._dref,
            reserved_keys=self._reserved_keys,
            ddefparams_data=self._ddef['params'].get('ddata'),
            ddefparams_obj=self._ddef['params'].get('dobj'),
            data_none=self._data_none,
            max_ndim=self._max_ndim,
        )

    def remove_all(self, excluded=None):

        # check excluded
        if isinstance(excluded, str):
            excluded = [excluded]

        # remove all obj
        lw = list(self.dobj.keys())
        for ww in lw:
            if (excluded is not None) and ww in excluded:
                continue
            self.remove_obj(
                list(self.dobj[ww].keys()),
                which=ww,
                propagate=True)

        # remove all data
        self.remove_data(list(self.ddata.keys()), propagate=True)

        # remove all refs
        self.remove_ref(list(self.dref.keys()), propagate=True)

    # ---------------------
    # Get / set / add / remove params
    # ---------------------

    def __check_which(self, which=None, return_dict=None):
        """ Check which in ['data'] + list(self._dobj.keys() """
        return _class1_check._check_which(
            dref=self._dref,
            ddata=self._ddata,
            dobj=self._dobj,
            which=which,
            return_dict=return_dict,
        )

    def get_lparam(self, which=None, for_show=None):
        """ Return the list of params for the chosen dict

        which can be:
            - 'ref'
            - 'data'
            - dobj[<which>]
        """
        which, dd = self.__check_which(which, return_dict=True)
        if which in ['ref', 'data']:
            for_show = False
        return _class1_show._get_lparam(dd=dd, for_show=for_show)

    def get_param(
        self,
        param=None,
        key=None,
        ind=None,
        returnas=None,
        which=None,
    ):
        """ Return the array of the chosen parameter (or list of parameters)

        which can be:
            - 'ref'
            - 'data'
            - dobj[<which>]

        param cen be a str or a list of str

        Can be returned as:
            - dict: {param0: {key0: values0, key1: value1...}, ...}
            - np.ndarray: {param0: np.r_[values0, value1...], ...}

        """
        which, dd = self.__check_which(which, return_dict=True)
        return _class1_check._get_param(
            dd=dd, dd_name=which,
            param=param, key=key, ind=ind, returnas=returnas,
        )

    def set_param(
        self,
        param=None,
        value=None,
        ind=None,
        key=None,
        which=None,
        distribute=None,
    ):
        """ Set the value of a parameter

        which can be:
            - 'ref'
            - 'data'
            - dobj[<which>]

        value can be:
            - None
            - a unique value (int, float, bool, str, tuple) common to all keys
            - an iterable of vlues (array, list) => one for each key

        A subset of keys can be chosen (ind, key, fed to self.select()) to set
        only the value of some key

        """
        which, dd = self.__check_which(which, return_dict=True)
        param = _class1_check._set_param(
            dd=dd, dd_name=which,
            param=param, value=value, ind=ind, key=key,
            distribute=distribute,
        )

        # if param refers to an object => update
        if param in self._dobj.keys():
            self.update()

    def add_param(
        self,
        param,
        value=None,
        which=None,
    ):
        """ Add a parameter, optionnally also set its value """
        which, dd = self.__check_which(which, return_dict=True)
        param = _class1_check._add_param(
            dd=dd,
            dd_name=which,
            param=param,
            value=value,
        )

        # if param refers to an object => update
        if param in self._dobj.keys():
            self.update()

    def remove_param(
        self,
        param=None,
        which=None,
    ):
        """ Remove a parameter, none by default, all if param = 'all' """
        which, dd = self.__check_which(which, return_dict=True)
        _class1_check._remove_param(
            dd=dd,
            dd_name=which,
            param=param,
        )

    ###########
    # to / from dict
    ###########

    @classmethod
    def from_dict(cls, din=None, sep=None, obj=None):
        obj = super().from_dict(din=din, sep=sep, obj=obj)
        obj.update()
        return obj

    ###########
    # properties
    ###########

    @property
    def dref(self):
        """ the dict of references """
        return self._dref

    @property
    def ddata(self):
        """ the dict of data """
        return self._ddata

    @property
    def dobj(self):
        """ the dict of obj """
        return self._dobj

    ###########
    # set and propagate indices for refs
    ###########

    def add_indices_per_ref(self, indices=None, ref=None, distribute=None):

        lparam = self.get_lparam(which='ref')
        if 'indices' not in lparam:
            self.add_param('indices', which='ref')

        self.set_param(
            which='ref',
            param='indices',
            key=ref,
            value=np.array(indices).ravel(),
            distribute=distribute,
        )

    def propagate_indices_per_ref(
        self,
        ref=None,
        lref=None,
        ldata=None,
        param=None,
    ):
        """ Propagate the indices set for a ref to all other lref

        Index propagation is done:
            - ldata = list of len() = 1 + len(lref)
                according to arbitrary (monotonous) data for each ref
            - according to a criterion:
                - 'index': set matching indices (default)
                - param: set matching monotonous quantities depending on ref
        """
        _class1_compute.propagate_indices_per_ref(
            ref=ref,
            lref=lref,
            ldata=ldata,
            dref=self._dref,
            ddata=self._ddata,
            param=param,
            lparam_data=self.get_lparam(which='data')
        )

    ###########
    # extract
    ###########

    def extract(
        self,
        keys=None,
        # optional includes
        inc_monot=None,
        inc_vectors=None,
        inc_allrefs=None,
        # output
        coll2=None,
        inplace=None,
        return_keys=None,
    ):
        """ Extract some selected data and return as new instance

        Automatically includes:
            - all desired data keys
            - all relevant ref

        Optionally can also include:
            - inc_monot: monotonous vectors matching any ref
            - inc_vectors: all (1d) vectors matching any ref
            - inc_allrefs: all (nd) array matching any full ref set

        Optionally:
            coll2: DataStock instance to be populated
            return_keys: returns the value of keys

        """

        return _class1_compute._extract_instance(
            self,
            keys=keys,
            # optional includes
            inc_monot=inc_monot,
            inc_vectors=inc_vectors,
            inc_allrefs=inc_allrefs,
            # output
            coll2=coll2,
            inplace=inplace,
            return_keys=return_keys,
        )

    ###########
    # General use methods
    ###########

    # ---------------------
    # export methods
    # ---------------------

    def to_DataFrame(self, which=None, keys=None):
        """ Export a set of uniform data arrays to a pandas DataFrame

        To be done

        """
        return _export_dataframe.to_dataframe(
            coll=self,
            which=which,
            keys=keys,
        )

    def find_plateau(self, keys=None, ref=None):
        """ Typically used for time-traces, identify plateau phases """
        return _find_plateau.find_plateau(
            coll=self,
            keys=keys,
            ref=ref,
        )

    # ---------------------
    # Key selection methods
    # ---------------------

    def select(self, which=None, log=None, returnas=None, **kwdargs):
        """ Return the indices / keys of data matching criteria

        The selection is done comparing the value of all provided parameters
        The result is a boolean indices array, optionally with the keys list
        It can include:
            - log = 'all': only the data matching all criteria
            - log = 'any': the data matching any criterion

        If log = 'raw', a dict of indices arrays is returned, showing the
        details for each criterion

        """
        which, dd = self.__check_which(which, return_dict=True)
        return _class1_check._select(
            dd=dd, dd_name=which,
            log=log, returnas=returnas,
            **kwdargs,
        )

    def _ind_tofrom_key(
        self,
        ind=None,
        key=None,
        returnas=int,
        which=None,
    ):
        """ Return ind from key or key from ind for all data """
        which, dd = self.__check_which(which, return_dict=True)
        return _class1_check._ind_tofrom_key(
            dd=dd, dd_name=which, ind=ind, key=key,
            returnas=returnas,
        )

    def _get_sort_index(self, which=None, param=None):
        """ Return sorting index of self.ddata dict """

        if param is None:
            return

        if param == 'key':
            ind = np.argsort(list(dd.keys()))
        elif isinstance(param, str):
            ind = np.argsort(
                self.get_param(param, which=which, returnas=np.ndarray)[param]
            )
        else:
            msg = "Arg param must be a valid str\n  Provided: {}".format(param)
            raise Exception(msg)
        return ind

    def sortby(self, param=None, order=None, which=None):
        """ sort the self.ddata dict by desired parameter """

        # --------------
        # Check inputs

        # order
        order = _generic_check._check_var(
            order,
            'order',
            types=str,
            default='increasing',
            allowed=['increasing', 'reverse'],
        )

        # which
        which, dd = self.__check_which(which, return_dict=True)

        # --------------
        # sort
        ind = self._get_sort_index(param=param, which=which)
        if ind is None:
            return
        if order == 'reverse':
            ind = ind[::-1]

        lk = list(dd.keys())
        dd = {lk[ii]: dd[lk[ii]] for ii in ind}

        if which == 'data':
            self._ddata = dd
        elif which == 'ref':
            self.dref = dd
        elif which in self._dobj.keys():
            self._dobj[which] = dd

    # ---------------------
    # Getting a common reference vector
    # ---------------------

    def get_ref_vector(
        self,
        # key
        key0=None,
        # which ref / dimension
        key=None,
        ref=None,
        dim=None,
        quant=None,
        name=None,
        units=None,
        # exclude from search
        key_exclude=None,
        ref_exclude=None,
        # nearest-neighbour interpolation input
        values=None,
        indices=None,
        ind_strict=None,
        warn=None,
    ):
        """ Return the monotonous vector associated to a ref of key

        Typical use: get the time vector of a multidimensional key

        >>> import datastock as ds
        >>> nt = 11; t0 = np.linspace(0, 10, nt);
        >>> nx = 21; x = np.linspace(0, 10, nx);
        >>> xt = np.sin(t0)[:, None] * (x-x.mean())[None, :]
        >>> st = ds.DataStock()
        >>> st.add_ref(key='nt', size=nt)
        >>> st.add_ref(key='nx', size=nx)
        >>> st.add_data(key='t0', data=t0)
        >>> st.add_data(key='x', data=x)
        >>> st.add_data(key='xt', data=xt)
        >>> hasref, hasvect, ref, key_vect, dind = st.get_ref_vector(key='xt', ref='nt', values=[2, 3, 3.1, 5])

        In the above example:
            - hasref = True: 'xt' has 'nt' has ref
            - hasvect = True: there is a monotonous vector with ref 'nt'
            - ref = 'nt'
            - key_vect = 't0'
            - dind = {
                'key': [2, 3, 3.1, 5],  # the desired time points
                'ind':  [2, 3, 3, 5],   # the indices of t in t0
                'indu': [2, 3, 5]       # the unique indices of t in t0
                'indr': (3, 4),         # bool array showing, for each indu, matching ind
                'indok': [True, False, ...]
              }

        """

        return _class1_uniformize.get_ref_vector(
            # ressources
            ddata=self._ddata,
            dref=self._dref,
            # inputs
            key0=key0,
            key=key,
            ref=ref,
            dim=dim,
            quant=quant,
            name=name,
            units=units,
            # exclude from search
            key_exclude=key_exclude,
            ref_exclude=ref_exclude,
            # parameters
            values=values,
            indices=indices,
            ind_strict=ind_strict,
            warn=warn,
        )

    def get_ref_vector_common(
        self,
        keys=None,
        # for selecting ref vector
        key=None,
        ref=None,
        dim=None,
        quant=None,
        name=None,
        units=None,
        # strategy for choosing common ref vector
        strategy=None,
        strategy_bounds=None,
        # exclude from search
        key_exclude=None,
        ref_exclude=None,
        # values, indices
        values=None,
        indices=None,
        ind_strict=None,
    ):
        """ Return a unique ref vector and a dict of indices


        Return
        ------
        val:        np.ndarray
            common finest vector
        dout:       dict
            dict of indices, per key, to match val

        """

        return _class1_uniformize.get_ref_vector_common(
            # ressources
            ddata=self._ddata,
            dref=self._dref,
            # inputs
            keys=keys,
            # for selecting ref vector
            key=key,
            ref=ref,
            dim=dim,
            quant=quant,
            name=name,
            units=units,
            # strategy for choosing common ref vector
            strategy=strategy,
            strategy_bounds=strategy_bounds,
            # exclude from search
            key_exclude=key_exclude,
            ref_exclude=ref_exclude,
            # parameters
            values=values,
            indices=indices,
            ind_strict=ind_strict,
        )

    # ---------------------
    # Uniformize
    # ---------------------

    def uniformize(
        self,
        keys=None,
        refs=None,
        param=None,
        lparam=None,
        returnas=None,
    ):

        return _class1_uniformize.uniformize(
            coll=self,
            keys=keys,
            refs=refs,
            param=param,
            lparam=lparam,
            returnas=returnas,
        )

    # ---------------------
    # domain
    # ---------------------

    def get_domain_ref(
        self,
        domain=None,
    ):
        """ Return a dict of index of valid steps based on desired domain
        """

        return _class1_domain.domain_ref(coll=self, domain=domain)

    # ---------------------
    # Binning
    # ---------------------

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

        return _class1_binning.binning(
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

    # ---------------------
    # Interpolation
    # ---------------------

    def interpolate(
        self,
        # interpolation base
        keys=None,
        ref_key=None,
        # interpolation pts
        x0=None,
        x1=None,
        # domain limitations
        domain=None,
        # common ref
        ref_com=None,
        # parameters
        grid=None,
        deg=None,
        deriv=None,
        log_log=None,
        nan0=None,
        # store vs return
        returnas=None,
        return_params=None,
        store=None,
        store_keys=None,
        inplace=None,
    ):
        """ Interpolate keys in desired dimension

        """
        return _class1_interpolate.interpolate(
            coll=self,
            # interpolation base
            keys=keys,
            ref_key=ref_key,
            # interpolation pts
            x0=x0,
            x1=x1,
            # domain limitations
            domain=domain,
            # common ref
            ref_com=ref_com,
            # parameters
            grid=grid,
            deg=deg,
            deriv=deriv,
            log_log=log_log,
            nan0=nan0,
            # store vs return
            returnas=returnas,
            return_params=return_params,
            store=store,
            store_keys=store_keys,
            inplace=inplace,
        )

    # ---------------------
    # color touch array
    # ---------------------

    def get_color_touch(
        self,
        data=None,
        dcolor=None,
        # options
        color_default=None,
        vmin=None,
        vmax=None,
        log=None,
    ):

        return _color_touch.main(
            coll=self,
            data=data,
            dcolor=dcolor,
            # options
            color_default=color_default,
            vmin=vmin,
            vmax=vmax,
            log=log,
        )

    # ---------------------
    # Methods computing correlations
    # ---------------------

    def compute_correlations(
        self,
        data=None,
        ref=None,
        correlations=None,
        verb=None,
        returnas=None,
    ):
        return _class1_compute.correlations(
            data=data,
            ref=ref,
            correlations=correlations,
            ddata=self._ddata,
            dref=self._dref,
            verb=verb,
            returnas=returnas,
        )

    # ---------------------
    # Methods for showing data
    # ---------------------

    def show(
        self,
        show_which=None,
        show=None,
        # pretty print options
        sep=None,
        line=None,
        justify=None,
        table_sep=None,
        # bool options
        verb=True,
        returnas=False,
    ):
        """ Summary description of the object content """
        return _class1_show.main(
            coll=self,
            show_which=show_which,
            show=show,
            # pretty print options
            sep=sep,
            line=line,
            justify=justify,
            table_sep=table_sep,
            # bool options
            verb=verb,
            returnas=returnas,
        )

    def _get_show_obj(self, which=None):
        return _class1_show._show_obj_def

    def show_data(self):
        self.show(show_which=['ref', 'data'])

    def show_obj(self):
        self.show(show_which=('ref', 'data'))

    def show_interactive(self):
        self.show(show_which=['axes', 'mobile', 'interactivity'])

    def show_details(
        self,
        key=None,
        which=None,
        # pretty print options
        sep=None,
        line=None,
        justify=None,
        table_sep=None,
        # bool options
        verb=True,
        returnas=False,
    ):
        """ Summary description of the object content """
        return _class1_show.main_details(
            coll=self,
            which=which,
            key=key,
            # pretty print options
            sep=sep,
            line=line,
            justify=justify,
            table_sep=table_sep,
            # bool options
            verb=verb,
            returnas=returnas,
        )

    def _get_show_details(self, which=None, key=None):
        raise NotImplementedError()

    def __repr__(self):
        try:
            return self.show(returnas=str, verb=False)
        except Exception:
            return self.__class__.__name__

    # ------
    # links

    def show_links(self):

        lcol = [['category', 'depends on']]
        lar = [[[k0, str(v0)] for k0, v0 in self.__dlinks.items()]]
        return _generic_utils.pretty_print(
            headers=lcol,
            content=lar,
            sep=None,
            line=None,
            table_sep=None,
            verb=True,
            returnas=False,
        )


# #############################################################################
# #############################################################################
#            set __all__
# #############################################################################


__all__ = [
    sorted([k0 for k0 in locals() if k0.startswith('DataStock')])[-1]
]
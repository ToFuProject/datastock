# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np


# library-specific
from . import _generic_check
from . import _generic_utils
from . import _class1_check
from ._class0 import *
from . import _class1_compute


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
            'ddata': {
                'units':  (str, 'a.u.'),
                'dim':    (str, 'unknown'),
                'quant':  (str, 'unknown'),
                'name':   (str, 'unknown'),
                'source': (str, 'unknown'),
            },
         },
    }

    # _dallowed_params = None
    _data_none = None
    _reserved_keys = None

    _show_in_summary_core = ['shape', 'ref']
    _show_in_summary = 'all'
    _max_ndim = None

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
        )

    # ---------------------
    # Adding ref / quantity one by one
    # ---------------------

    def add_ref(self, size=None, key=None, data=None, **kwdargs):
        dref = {key: {'data': data, 'size': size, **kwdargs}}
        # Check consistency
        self.update(ddata=None, dref=dref, harmonize=True)

    def add_data(self, data=None, key=None, ref=None, **kwdargs):
        ddata = {key: {'data': data, 'ref': ref, **kwdargs}}
        # Check consistency
        self.update(ddata=ddata, dref=None, harmonize=True)

    def add_obj(self, which=None, key=None, harmonize=None, **kwdargs):
        dobj = {which: {key: kwdargs}}
        # Check consistency
        self.update(dobj=dobj, dref=None, harmonize=harmonize)

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
            ddefparams_data=self._ddef['params']['ddata'],
            ddefparams_obj=self._ddef['params']['dobj'],
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
            ddefparams_data=self._ddef['params']['ddata'],
            ddefparams_obj=self._ddef['params']['dobj'],
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
            dobj0=self._dobj,
            ddata0=self._ddata,
            dref0=self._dref,
            reserved_keys=self._reserved_keys,
            ddefparams_data=self._ddef['params']['ddata'],
            ddefparams_obj=self._ddef['params']['dobj'],
            data_none=self._data_none,
            max_ndim=self._max_ndim,
        )

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

    def get_lparam(self, which=None):
        """ Return the list of params for the chosen dict

        which can be:
            - 'ref'
            - 'data'
            - dobj[<which>]
        """
        which, dd = self.__check_which(which, return_dict=True)
        return list(list(dd.values())[0].keys())

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
        _class1_check._set_param(
            dd=dd, dd_name=which,
            param=param, value=value, ind=ind, key=key,
            distribute=distribute,
        )

    def add_param(
        self,
        param,
        value=None,
        which=None,
    ):
        """ Add a parameter, optionnally also set its value """
        which, dd = self.__check_which(which, return_dict=True)
        _class1_check._add_param(
            dd=dd,
            dd_name=which,
            param=param,
            value=value,
        )

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
    def from_dict(cls, din=None, sep=None):
        obj = super().from_dict(din=din, sep=sep)
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

    def extract(self, keys=None):
        """ Extract some selected data and return as new instance """

        # ----------------
        # check inputs

        if keys is None:
            return
        if isinstance(keys, str):
            keys = [keys]

        keys = _generic_check._check_var_iter(
            keys, 'keys',
            types=list,
            allowed=self._ddata.keys(),
        )

        # -----------------------------
        # Get corresponding list of ref

        lref = set([
            k0 for k0, v0 in self._dref.items()
            if any([ss in keys for ss in v0['ldata']])
        ])

        # -------------------
        # Populate with ref

        coll = self.__class__()

        lpar = [
            pp for pp in self.get_lparam(which='ref')
            if pp not in ['ldata', 'ldata_monot', 'ind', 'data']
        ]
        for k0 in lref:
            coll.add_ref(
                key=k0,
                **copy.deepcopy({pp: self._dref[k0][pp] for pp in lpar}),
            )

        # -------------------
        # Populate with data

        lpar = [
            pp for pp in self.get_lparam(which='data')
            if pp not in ['shape', 'monot']
        ]
        for k0 in keys:
            coll.add_data(
                key=k0,
                **copy.deepcopy({pp: self._ddata[k0][pp] for pp in lpar}),
            )

        return coll

    ###########
    # General use methods
    ###########

    def to_DataFrame(self, which=None):
        which, dd = self.__check_which(which, return_dict=True)
        if which is None:
            return
        import pandas as pd
        return pd.DataFrame(dd)

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
            log=log, returnas=returnas, **kwdargs,
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
        show_core=None,
        sep='  ',
        line='-',
        just='l',
        table_sep=None,
        verb=True,
        returnas=False,
    ):
        """ Summary description of the object content """

        # ------------
        # check inputs

        if show_which is None:
            show_which = ['ref', 'data', 'obj']

        lcol, lar = [], []

        # -----------------------
        # Build for dref

        if 'ref' in show_which and len(self._dref) > 0:
            lcol.append(['ref key', 'size', 'nb. data', 'nb. data monot.'])
            lar.append([
                [
                    k0,
                    str(self._dref[k0]['size']),
                    str(len(self._dref[k0]['ldata'])),
                    str(len(self._dref[k0]['ldata_monot'])),
                ]
                for k0 in self._dref.keys()
            ])

            lp = self.get_lparam(which='ref')
            if 'indices' in lp:
                lcol[0].append('indices')
                for ii, (k0, v0) in enumerate(self._dref.items()):
                    if self._dref[k0]['indices'] is None:
                        lar[0][ii].append(str(v0['indices']))
                    else:
                        lar[0][ii].append(str(list(v0['indices'])))

            if 'group' in lp:
                lcol[0].append('group')
                for ii, (k0, v0) in enumerate(self._dref.items()):
                    lar[0][ii].append(str(self._dref[k0]['group']))

            if 'inc' in lp:
                lcol[0].append('increment')
                for ii, (k0, v0) in enumerate(self._dref.items()):
                    lar[0][ii].append(str(self._dref[k0]['inc']))

        # -----------------------
        # Build for ddata

        if 'data' in show_which and len(self._ddata) > 0:

            if show_core is None:
                show_core = self._show_in_summary_core
            if isinstance(show_core, str):
                show_core = [show_core]

            lp = self.get_lparam(which='data')
            lkcore = ['shape', 'ref']
            assert all([ss in lp + lkcore for ss in show_core])
            col2 = ['data key'] + show_core

            if show is None:
                show = self._show_in_summary
            if show == 'all':
                col2 += [pp for pp in lp if pp not in col2]
            else:
                if isinstance(show, str):
                    show = [show]
                assert all([ss in lp for ss in show])
                col2 += [pp for pp in show if pp not in col2]
            col2 = [cc for cc in col2 if cc != 'data']

            ar2 = []
            for k0 in self._ddata.keys():
                lu = [k0] + [str(self._ddata[k0].get(cc)) for cc in col2[1:]]
                ar2.append(lu)

            lcol.append(col2)
            lar.append(ar2)

        # -----------------------
        # Build for dobj

        anyobj = (
            len(self._dobj) > 0
            and any([
                ss in show_which
                for ss in ['obj'] + list(self._dobj.keys())
            ])
        )
        if anyobj:
            for k0, v0 in self._dobj.items():
                if 'obj' in show_which or k0 in show_which:
                    lk = self.get_lparam(which=k0)
                    lk = [
                        kk for kk in lk
                        if 'func' not in kk
                        and 'class' not in kk
                        and kk not in ['handle']
                        and not (k0 == 'axes' and kk == 'bck')
                        and all([
                            not isinstance(v1[kk], dict)
                            for v1 in v0.values()
                        ])
                    ]
                    lcol.append([k0] + [pp for pp in lk])
                    lar.append([
                        [k1] + [str(v1[kk]) for kk in lk]
                        for k1, v1 in v0.items()
                    ])

        return _generic_utils.pretty_print(
            headers=lcol,
            content=lar,
            sep=sep,
            line=line,
            table_sep=table_sep,
            verb=verb,
            returnas=returnas,
        )

    def show_all(self):
        self.show(show_which=None)

    def show_data(self):
        self.show(show_which=['ref', 'data'])

    def show_interactive(self):
        self.show(show_which=['axes', 'mobile', 'interactivity'])

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

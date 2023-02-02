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
from ._class0 import *
from . import _class1_compute
from . import _class1_domain
from . import _class1_binning
from . import _class1_interpolate
from . import _class1_uniformize
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

    _show_in_summary_core = ['shape', 'ref']
    _show_in_summary = 'all'
    _max_ndim = None
    _dshow = {}

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

    def add_ref(self, size=None, key=None, data=None, harmonize=None, **kwds):
        dref = {key: {'data': data, 'size': size, **kwds}}
        # Check consistency
        self.update(ddata=None, dref=dref, harmonize=harmonize)

    def add_data(self, data=None, key=None, ref=None, harmonize=None, **kwds):
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
            dobj0=self._dobj,
            ddata0=self._ddata,
            dref0=self._dref,
            reserved_keys=self._reserved_keys,
            ddefparams_data=self._ddef['params'].get('ddata'),
            ddefparams_obj=self._ddef['params'].get('dobj'),
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
        return _class1_check._get_lparam(dd=dd, for_show=for_show)

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
        key=None,
        # which ref / dimension
        ref=None,
        dim=None,
        quant=None,
        name=None,
        units=None,
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
            key=key,
            ref=ref,
            dim=dim,
            quant=quant,
            name=name,
            units=units,
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
        ref=None,
        dim=None,
        quant=None,
        name=None,
        units=None,
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
            ref=ref,
            dim=dim,
            quant=quant,
            name=name,
            units=units,
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
        keys=None,
        ref_key=None,
        bins=None,
    ):
        """ return binned data and units along dimension indicated by refkey"""

        return _class1_binning.binning(
            coll=self,
            keys=keys,
            ref_key=ref_key,
            bins=bins,
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
        return_params=None,
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
            return_params=return_params,
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
        elif isinstance(show_which, tuple):
            if 'obj' in show_which:
                show_which = [
                    k0 for k0 in ['ref', 'data'] if k0 not in show_which
                ]
            else:
                show_which = [
                    k0 for k0 in ['ref', 'data'] + list(self._dobj.keys())
                    if k0 not in show_which
                ]

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
                    lk = _class1_check._show_get_fields(
                        which=k0,
                        dobj=self._dobj,
                        lparam=self.get_lparam(which=k0, for_show=True),
                        dshow=self._dshow,
                    )
                    lcol.append([k0] + [pp.split('.')[-1] for pp in lk])
                    lar.append([
                        [k1] + _class1_check._show_extract(dobj=v1, lk=lk)
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

    def show_data(self):
        self.show(show_which=['ref', 'data'])

    def show_obj(self):
        self.show(show_which=('ref', 'data'))

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

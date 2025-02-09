

import copy


import numpy as np


# library-specific
from . import _generic_utils
from . import _generic_check
from . import _saveload


class DataStock0(object):

    def __init__(self):
        self.__object = object()

    def to_dict(
        self,
        flatten=None,
        sep=None,
        asarray=None,
        excluded=None,
        # copy vs ref
        copy=None,
        # dtypes
        returnas=None,
    ):
        """ Return a flat dict view of the object's attributes

        Useful for:
            * displaying all attributes
            * saving to file
            * exchanging data with other libraries

        Parameters
        ----------
        sep :       str
            Separator char used for flattening the dict
            The output dict is flat (i.e.: no nested dict)
            Keys are created from the keys of nested dict, separated by sep

        Return
        ------
        returnas:      str
            - 'types': a dict with only the types
            - 'values': a dict with the values
            - 'both': 2 seperate dicts, one with types, one with values
            - 'blended': a dict with both types and values (save compatible)

        """

        return _generic_utils.to_dict(
            self,
            flatten=flatten,
            sep=sep,
            asarray=asarray,
            excluded=excluded,
            # copy vs ref
            copy=copy,
            # dtypes
            returnas=returnas,
        )

    @classmethod
    def from_dict(cls, din=None, isflat=None, sep=None, obj=None):
        """ Populate the instances attributes using an input dict

        The input dict must be properly formatted
        In practice it should be the return output of a similar class to_dict()

        Parameters
        ----------
        din :    dict
            The properly formatted ditionnary from which to read the attributes
        sep :   str
            The separator that was used to format fd keys (cf. self.to_dict())
        """

        if isflat is True:
            din = _generic_utils.reshape_dict(din, sep=sep)

        # ---------------------
        # Instanciate and populate

        if obj is None:
            obj = cls()

        for k0 in din.keys():
            if k0 == '_ddef':
                if 'dobj' not in din[k0]['params'].keys():
                    din[k0]['params']['dobj'] = {}
                if 'dref' not in din[k0]['params'].keys():
                    din[k0]['params']['dref'] = {}
            setattr(obj, k0, din[k0])

        return obj

    def copy(self, excluded=None, sep=None):
        """ Return another instance of the object, with the same attributes

        If deep=True, all attributes themselves are also copies
        """
        return self.__class__.from_dict(
            din=self.to_dict(
                flatten=False,
                excluded=excluded,
                returnas='values',
                copy=True,
            )
        )

    def get_nbytes(self):
        """ Compute and return the object size in bytes (i.e.: octets)

        A flat dict containing all the objects attributes is first created
        The size of each attribute is then estimated with np.asarray().nbytes

        Returns
        -------
        total :     int
            The total object estimated size, in bytes
        dsize :     dict
            A dictionnary giving the size of each attribute
        """
        dd = self.to_dict(flatten=True, copy=False, returnas='values')
        dsize = dd.fromkeys(dd.keys(), 0)
        total = 0
        for k0, v0 in dd.items():
            try:
                dsize[k0] = np.asarray(v0).nbytes
            except Exception as err:
                dsize[k0] = str(err)
            total += dsize[k0]
        return total, dsize


    #############################
    #  operator overloading
    #############################


    def __eq__(self, obj, excluded=None, returnas=None, verb=None):
        return _generic_utils.compare_obj(
            obj0=self,
            obj1=obj,
            excluded=excluded,
            returnas=returnas,
            verb=verb,
        )

    def __neq__(self, obj, excluded=None, returnas=None, verb=None):
        return not self.__eq__(obj, excluded=excluded, returnas=returnas, verb=verb)

    def __hash__(self, *args, **kargs):
        return self.__object.__hash__(*args, **kargs)

    #############################
    #  saving
    #############################

    def save(
        self,
        pfe=None,
        path=None,
        name=None,
        sep=None,
        overwrite=None,
        return_pfe=False,
        verb=True,
    ):

        lsep = [';', '&', '?', '#', ',', '~', '.', '-', '_']
        if sep is None:
            for ss in lsep:
                c0 = (
                    any([ss in k0 for k0 in self.ddata.keys()])
                    or any([ss in k0 for k0 in self.dref.keys()])
                    or any([
                        any([ss in k0 for k0 in self.dobj[k0].keys()])
                        for k0 in self._dobj.keys()
                    ])
                )
                if not c0:
                    sep = ss
                    break

        # call parent method
        return _saveload.save(
            dflat=self.to_dict(
                flatten=True,
                sep=sep,
                asarray=True,
                returnas='blended',
            ),
            pfe=pfe,
            sep=sep,
            path=path,
            name=name,
            overwrite=overwrite,
            clsname=self.__class__.__name__,
            return_pfe=return_pfe,
            verb=verb,
        )


# #############################################################################
# #############################################################################
#            set __all__
# #############################################################################


__all__ = [
    sorted([k0 for k0 in locals() if k0.startswith('DataStock')])[-1]
]
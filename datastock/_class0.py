

import copy


import numpy as np


# library-specific
from . import _generic_utils
from . import _generic_check
from . import _saveload


class DataStock0(object):

    def __init__(self):
        self.__object = object()

    def to_dict(self, flatten=None, sep=None):
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
        dout :      dict
            dict containing all the objects attributes (optionally flattened)

        """

        # ------------
        # check inputs

        flatten = _generic_check._check_var(
            flatten, 'flatten',
            default=False,
            types=bool,
        )

        # ---------------------------
        # Get list of dict attributes

        dout = copy.deepcopy({
            k0: getattr(self, k0)
            for k0 in dir(self)
            if isinstance(getattr(self, k0), dict)
            and k0 != '__dict__'
            and not (
                hasattr(self.__class__, k0)
                and isinstance(getattr(self.__class__, k0), property)
            )
        })

        # -------------------
        # optional flattening

        if flatten is True:
            dout = _generic_utils.flatten_dict(dout, parent_key=None, sep=sep)

        return dout

    @classmethod
    def from_dict(cls, din=None, reshape=None, sep=None):
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

        if reshape is True:
            din = _generic_utils.reshape_dict(din, sep=sep)

        # ---------------------
        # Instanciate and populate

        obj = cls()
        for k0 in din.keys():
            setattr(obj, k0, din[k0])

        return obj

    def copy(self):
        """ Return another instance of the object, with the same attributes

        If deep=True, all attributes themselves are also copies
        """
        return self.__class__.from_dict(din=copy.deepcopy(self.to_dict()))

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
        dd = self.to_dict(flatten=True)
        dsize = dd.fromkeys(dd.keys(), 0)
        total = 0
        for k0, v0 in dd.items():
            try:
                dsize[k0] = np.asarray(v0).nbytes
                total += dsize[k0]
            except Exception as err:
                dsize[k0] = str(err)
        return total, dsize


    #############################
    #  operator overloading
    #############################

    def __eq__(self, obj, returnas=None, verb=None):
        return _generic_utils.compare_obj(
            obj0=self,
            obj1=obj,
            returnas=returnas,
            verb=verb,
        )

    def __neq__(self, obj, returnas=None, verb=None):
        return not self.__eq__(obj, returnas=returnas, verb=verb)

    def __hash__(self, *args, **kargs):
        return self.__object.__hash__(*args, **kargs)

    #############################
    #  saving
    #############################

    def save(
        self,
        path=None,
        name=None,
        sep=None,
        verb=True,
        return_pfe=False,
    ):

        # call parent method
        return _saveload.save(
            dflat=self.to_dict(flatten=True, sep=sep),
            path=path,
            name=name,
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



import itertools as itt
from copy import deepcopy


import numpy as np
import scipy.sparse as scpsp
import astropy.units as asunits
from functools import reduce  # forward compatibility for Python 3
import operator


from . import _generic_check


# ###############################################################
# ###############################################################
#                   Pretty printing
# ###############################################################


def _pretty_print_check(
    headers=None,
    content=None,
    sep=None,
    line=None,
    justify=None,
    table_sep=None,
    verb=None,
    returnas=None,
):

    # -------------
    # headers
    # -------------

    c0 = (
        isinstance(headers, list)
        and all([
            isinstance(h0, list)
            and (
                all([isinstance(h1, str) for h1 in h0])
                or all([
                    isinstance(h1, list)
                    and all([isinstance(h2, str) for h2 in h1])
                    for h1 in h0
                ])
            )
            for h0 in headers
        ])
    )
    if not c0:
        msg = (
            "Arg headers must be a list of list of str / list of str\n"
            "Expected form, one of:\n"
            "\t- [[h00, h01, h02], [h10, h11, h12], ...]\n"
            "\t- [[[h00l0, h01l0, h02l0], [h00l1, h01l1, h02l1]], "
            " [[h10l0, h11l0, h12l0], [h10l1, h11l1, h12l1]], ...]\n"
            "The second form is used in case of multi-line headers\n\n"
            f"Provided:\n{headers}"
        )
        raise Exception(msg)

    if len(headers) > 0 and len(headers[0]) > 0:
        if isinstance(headers[0][0], str):
            for ii, h0 in enumerate(headers):
                headers[ii] = [headers[ii]]
    nhead = [len(h0[0]) for h0 in headers]

    # -------------
    # content
    # -------------

    c0 = (
        isinstance(content, list)
        and all([
            isinstance(c0, list)
            and all([
                isinstance(c1, list)
                and all([isinstance(c2, str) for c2 in c1])
                for c1 in c0
            ])
            for c0 in content
        ])
        and all([
            all([len(c1) == nhead[ii] for c1 in c0])
            for ii, c0 in enumerate(content)
        ])
    )
    if not c0:
        msg = (
            "Arg content must be a list of list of list of str\n"
            "Expected form, one of:\n"
            "\t- [[[c00l0, c01l0, c02l0], [c00l1, c01l1, c02l1]], ...\n"
            f"\t- Each with len() = {nhead}\n"
            f"Provided:\n{content}"
        )
        raise Exception(msg)

    # -------------
    # options
    # -------------

    # sep
    sep = _generic_check._check_var(
        sep, 'sep',
        default='  ',
        types=str,
    )

    # line
    line = _generic_check._check_var(
        line, 'line',
        default='-',
        types=str,
    )

    # justify
    justify = _generic_check._check_var(
        justify, 'justify',
        default='left',
        allowed=['left', 'right'],
    )

    # table_sep
    table_sep = _generic_check._check_var(
        table_sep, 'table_sep',
        default='\n\n',
        types=str,
    )

    # returnas
    returnas = _generic_check._check_var(
        returnas, 'returnas',
        default=False,
        allowed=[False, str, np.ndarray],
    )

    # verb
    verb = _generic_check._check_var(
        verb, 'verb',
        default=returnas is False,
        types=bool,
    )

    return headers, content, sep, line, justify, table_sep, verb, returnas


def _pretty_print_chararray(
    head=None,
    content=None,
    sep=None,
    line=None,
    justify=None,
):

    head = np.array(head, dtype='U')
    content = np.array(content, dtype='U')

    if head.ndim == 1:
        head = head.reshape((1, -1))
    if content.ndim == 1:
        content = content.reshape((1, -1))

    assert head.shape[1] == content.shape[1]
    ncol = head.shape[1]

    # Get length for justifying
    nmax = np.max(
        [
            np.char.str_len(head).max(axis=0),
            np.char.str_len(content).max(axis=0),
        ],
        axis=0,
    )

    # justify
    fjust = 'ljust' if justify == 'left' else 'rjust'
    return '\n'.join(
        [
            sep.join([
                getattr(h1, fjust)(nmax[ii])
                for ii, h1 in enumerate(h0)
            ])
            for h0 in head
        ]
        + [sep.join([line*nmax[ii] for ii in range(ncol)])]
        + [
            sep.join([
                getattr(c1, fjust)(nmax[ii])
                for ii, c1 in enumerate(c0)
            ])
            for c0 in content
        ]
    )


def pretty_print(
    headers=None,
    content=None,
    sep=None,
    line=None,
    justify=None,
    table_sep=None,
    verb=None,
    returnas=None,
):
    """ Summary description of the object content as a np.array of str """

    # --------------
    #  check inputs

    (
        headers, content, sep, line, justify, table_sep, verb, returnas,
    ) = _pretty_print_check(
        headers=headers,
        content=content,
        sep=sep,
        line=line,
        justify=justify,
        table_sep=table_sep,
        verb=verb,
        returnas=returnas,
    )

    # ---------
    # format

    lmsg = [
        _pretty_print_chararray(
            head=head,
            content=content[ii],
            sep=sep,
            line=line,
            justify=justify,
        )
        for ii, head in enumerate(headers)
    ]

    if verb is True or returnas is str:
        msg = table_sep.join(lmsg)
        if verb:
            print(msg)

    # -------
    # return

    if returnas is str:
        return msg


# ###############################################################
# ###############################################################
#                   Compare dict
# ###############################################################


def _compare_dict_verb_return(dout, returnas, verb):

    if len(dout) > 0:

        # msg ?
        if verb or returnas is str:
            lstr = [f"{k0}: {v0}" for k0, v0 in dout.items()]
            msg = (
                "The following differences have been found:\n"
                + "\n".join(lstr)
            )
            if verb:
                print(msg)

        # return
        if returnas is str:
            return msg
        elif returnas is dict:
            return dout
        else:
            return False

    else:
        return True


def compare_dict(
    d0=None,
    d1=None,
    dname=None,
    returnas=None,
    verb=None,
):

    # ------------
    # check inputs

    # dname
    dname = _generic_check._check_var(
        dname, 'dname',
        default='',
        types=str,
    )

    # returnas
    returnas = _generic_check._check_var(
        returnas, 'returnas',
        default=bool,
        allowed=[bool, str, dict],
    )

    # verb
    verb = _generic_check._check_var(
        verb, 'verb',
        default=True,
        types=bool,
    )

    # ------------
    # check inputs

    dout = {}
    if dname == '':
        kroot = 'root'
    else:
        kroot = dname

    # Class
    if not (isinstance(d0, dict) and isinstance(d1, dict)):
        dout[kroot] = f'different classes: {type(d0)} vs {type(d1)}'
        return _compare_dict_verb_return(dout, returnas, verb)

    # Keys
    lk0 = sorted(list(d0.keys()))
    lk1 = sorted(list(d1.keys()))
    if lk0 != lk1:
        lk00 = [kk for kk in lk0 if kk not in lk1]
        lk11 = [kk for kk in lk1 if kk not in lk0]
        dout[kroot] = f'different keys: {lk00} vs {lk11}'
        return _compare_dict_verb_return(dout, returnas, verb)

    # values
    dkeys = {}
    for k0 in lk0:
        if dname == '':
            key = k0
        else:
            key = f'{dname}.{k0}'

        # class
        if d0[k0].__class__ != d1[k0].__class__:
            dkeys[key] = f'!= class ({type(d0[k0])} vs {type(d1[k0])})'
            continue

        elif d0[k0] is None:
            pass

        # scalars (int, float, bool, str)
        elif np.isscalar(d0[k0]):
            if d0[k0] != d1[k0]:
                dkeys[key] = f'!= values ({d0[k0]} vs {d1[k0]})'
                continue

        # numpy arrays
        elif isinstance(d0[k0], np.ndarray):
            msg = _compare_arrays(
                dname=dname,
                k0=k0,
                d0=d0,
                d1=d1,
            )
            if msg is not None:
                dkeys[key] = msg

        # sparse arrays
        elif scpsp.issparse(d0[k0]):
            if d0[k0].shape != d1[k0].shape:
                dkeys[key] = f'!= shapes ({d0[k0].shape} vs {d1[k0].shape})'
            elif not np.allclose(d0[k0].data, d1[k0].data, equal_nan=True):
                dkeys[key] = "not allclose"

        # lists and tuples
        elif isinstance(d0[k0], (list, tuple)):
            msg = _compare_list_tuple(
                dname=dname,
                k0=k0,
                d0=d0,
                d1=d1,
                key=key,
            )
            if msg is not None:
                dkeys[key] = msg

        # functions
        elif callable(d0[k0]):
            if d0[k0] != d1[k0]:
                dkeys[key] = "!= callable"

        # dict
        elif isinstance(d0[k0], dict):
            dd = compare_dict(
                d0=d0[k0],
                d1=d1[k0],
                dname=key,
                returnas=dict,
                verb=False,
            )
            if isinstance(dd, dict):
                dkeys.update(dd)

        # units (astropy)
        elif isinstance(d0[k0], asunits.core.UnitBase):
            if d0[k0] != d1[k0]:
                dkeys[key] = f'!= astropy units ({d0[k0]} vs {d1[k0]})'

        # not implemented cases
        else:
            msg = (
                f"Don't know how to handle d0['{k0}']:\n"
                f"\t- type: {type(d0[k0])}\n"
                f"\t- value: {d0[k0]}\n"
            )
            raise NotImplementedError(msg)

    # ------
    # return

    return _compare_dict_verb_return(dkeys, returnas, verb)


def _compare_arrays(
    dname=None,
    k0=None,
    d0=None,
    d1=None,
):

    msg = None

    # shape
    if d0[k0].shape != d1[k0].shape:
        msg = f'!= shapes ({d0[k0].shape} vs {d1[k0].shape})'

    # dtype
    elif not d0[k0].dtype == d1[k0].dtype:
        msg = f"!= dtypes ({d0[k0].dtype} vs {d1[k0].dtype})"

    # special case: array of str
    if 'str' in d0[k0].dtype.name:
        d0flat = d0[k0].ravel().tolist()
        d1flat = d1[k0].ravel().tolist()
        c0 = all([ss == d1flat[ii] for ii, ss in enumerate(d0flat)])
    else:
        try:
            c0 = np.allclose(d0[k0], d1[k0], equal_nan=True)
        except Exception as err:
            msg = (
                f"Failed to compare 2 arrays from '{dname}':\n"
                f"\t- d0['{k0}'] = {d0[k0]}\n"
                f"\t- d1['{k0}'] = {d1[k0]}\n"
            )
            raise Exception(msg) from err

        if not c0:
            msg = "not allclose"

    return msg


def _compare_list_tuple(
    dname=None,
    k0=None,
    d0=None,
    d1=None,
    key=None,
):

    msg = None

    # length
    if len(d0[k0]) != len(d1[k0]):
        msg = f'!= length ({len(d0[k0])} vs {len(d1[k0])})'

    # content type
    ltyp0 = [_simple_type(type(ss)) for ss in d0[k0]]
    ltyp1 = [_simple_type(type(ss)) for ss in d1[k0]]
    if any([t0 != t1 for t0, t1 in zip(ltyp0, ltyp1)]):
        msg = f"!= content type ({ltyp0} vs {ltyp1})"

    # content
    for ii in range(len(d0[k0])):

        if isinstance(d0[k0][ii], np.ndarray):
            c0 = np.allclose(d0[k0][ii], d1[k0][ii])

        else:

            try:
                c0 = d0[k0][ii] == d1[k0][ii]
            except Exception as err:
                msg = (
                    f"Don't know how to handle {key}:\n"
                    f"\t- type: {type(d0[k0])}\n"
                    f"\t- value: {d0[k0]}\n"
                )
                raise NotImplementedError(msg) from err

        if not c0:
            msg = f"!= content at position {ii} ({d0[k0][ii]} vs {d1[k0][ii]})"
            break

    return msg


def _simple_type(typ):
    return ''.join([
        ss for ss in typ.__name__.replace('_', '')
        if not ss.isdigit()
    ])


def compare_obj(
    obj0=None,
    obj1=None,
    excluded=None,
    returnas=None,
    verb=None,
):
    """ Compare the content of 2 instances """

    # -----------
    # Check class

    if obj0.__class__ != obj1.__class__:
        msg = (
            f"classes: {obj0.__class__.__name__} vs {obj1.__class__.__name__}"
        )
        raise Exception(msg)

    # -----------
    # Check

    return compare_dict(
        d0=obj0.to_dict(excluded=excluded, returnas='values', copy=False),
        d1=obj1.to_dict(excluded=excluded, returnas='values', copy=False),
        dname=None,
        returnas=returnas,
        verb=verb,
    )


# ###############################################################
# ###############################################################
#               Flatten / reshape dict
# ###############################################################


def to_dict(
    coll=None,
    flatten=None,
    sep=None,
    excluded=None,
    # copy vs ref
    asarray=None,
    copy=None,
    # dtypes
    returnas=None,
):

    # ------------
    # check inputs

    flatten = _generic_check._check_var(
        flatten, 'flatten',
        default=True,
        types=bool,
    )

    returnas = _generic_check._check_var(
        returnas, 'returnas',
        default='types',
        types=str,
        allowed=['types', 'values', 'both', 'blended'],
    )

    # ----------------------
    # get flat key/type tree

    dtypes, sep = flatten_dict_keys(
        din=coll,
        parent_key=None,
        sep=sep,
        excluded=excluded,
    )

    if returnas == 'types':
        if flatten is False:
            return reshape_dict(dtypes, sep=sep)
        else:
            return dtypes

    # ---------------------------
    # Get list of dict attributes

    dout = dict_from_dtypes(
        coll,
        dtypes=dtypes,
        flatten=flatten,
        sep=sep,
        asarray=asarray,
        copy=copy,
    )

    # ---------
    # return

    if returnas == 'blended':
        return _blend_dicts(
            dbase=dout,
            dextra=dtypes,
            extra_key='__type',
            flatten=flatten,
        )
    if returnas == 'both':
        return dtypes, dout
    else:
        return dout


def _flatten_dict_check(
    din=None,
    parent_key=None,
    sep=None,
    excluded=None,
):
    # ------------
    # check inputs

    # sep
    if sep is not None:
        sep = _generic_check._check_var(
            sep, 'sep',
            default='.',
            types=str,
        )

    # parent_key
    if parent_key is not None:
        parent_key = _generic_check._check_var(
            parent_key, 'parent_key',
            default=('',),
            types=(str, tuple),
        )

    # excluded
    if excluded is not None:
        if isinstance(excluded, str):
            excluded = ((excluded,),)

        if not isinstance(excluded, (list, tuple)):
            msg = "Arg excluded must be a tuple of tuples of str!"
            raise Exception(msg)

        if any([isinstance(ss, (str, list)) for ss in excluded]):
            excluded = tuple([
                tuple(ss) if isinstance(ss, list)
                else (ss if isinstance(ss, tuple) else (ss,))
                for ss in excluded
            ])

        c0 = (
            isinstance(excluded, tuple)
            and all([isinstance(tt, tuple) for tt in excluded])
            and all([all([isinstance(ss, str) for ss in tt]) for tt in excluded])
        )
        if not c0:
            msg = "Arg excluded must be a tuple of tuples of str!"
            raise Exception(msg)

    return parent_key, sep, excluded


def flatten_dict_keys(
    din=None,
    parent_key=None,
    sep=None,
    excluded=None,
):
    """ Return a flattened version of the input dict keys"""

    # ------------
    # check inputs
    # ------------

    parent_key, sep, excluded = _flatten_dict_check(
        din=din,
        parent_key=parent_key,
        sep=sep,
        excluded=excluded,
    )

    # ------------
    # top level
    # ------------

    if isinstance(din, dict):

        dkeys = {}
        for k0, v0 in din.items():

            # key
            if parent_key is None:
                key = (k0,)
            else:
                key = tuple([k1 for k1 in parent_key] + [k0])

            # value
            if isinstance(v0, dict):
                dkeys.update(
                    flatten_dict_keys(
                        v0,
                        key,
                        sep=None,
                        excluded=excluded,
                    )[0]
                )

            else:

                # get class
                if excluded is None or key not in excluded:
                    dkeys[key] = v0.__class__.__name__

    else:
        dkeys = {}
        lk0 = [
            k0 for k0 in dir(din)
            if k0 != '__dict__'
            and '__dlinks' not in k0
            and not (
                hasattr(din.__class__, k0)
                and isinstance(getattr(din.__class__, k0), property)
            )
            and isinstance(getattr(din, k0), dict)
        ]
        for k0 in lk0:
            dkeys.update(flatten_dict_keys(
                getattr(din, k0),
                parent_key=(k0,),
                sep=None,
                excluded=excluded,
            )[0])

    # ---------------------
    # format keys using sep
    # ---------------------

    if sep is not None:

        # --------------------
        # safety check vs sep

        # dict of non-conform keys
        dkout = {
            k0: v0 for k0, v0 in dkeys.items()
            if any([sep in k1 for k1 in k0])
        }

        # error msg
        if len(dkout) > 0:
            lstr = [f"\t- {k0}: {v0}" for k0, v0 in dkout.items()]
            msg = (
                f"The following keys already have the desired sep '{sep}':\n"
                + '\n'.join(lstr)
            )
            raise Exception(msg)

        # ----------
        # formatting

        dkeys = {sep.join(k0): v0 for k0, v0 in dkeys.items()}

    return dkeys, sep


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


def dict_from_dtypes(
    coll=None,
    dtypes=None,
    flatten=None,
    sep=None,
    asarray=None,
    copy=None,
):
    """ Assumes dtypes is flat """

    # -------------------------
    # check inputs

    # asarray
    asarray = _generic_check._check_var(
        asarray, 'asarray',
        default=False,
        types=bool,
    )

    copy = _generic_check._check_var(
        copy, 'copy',
        default=False,
        types=bool,
    )

    # -------------------------
    # get flat / unflat version

    # keys
    if sep is None:
        lkeys = sorted(dtypes.keys())
    else:
        lkeys = sorted([k0.split(sep) for k0 in dtypes.keys()])

    # -----------
    # build

    # initialize dout
    if flatten is True:
        dout = {}
    else:
        dout = dict(reshape_dict(dtypes, sep=sep))

    # loop on all keys from (flat) dtypes
    for k0 in lkeys:

        # get value from coll
        for ii, k1 in enumerate(k0):
            if ii == 0:
                out = getattr(coll, k1)
            else:
                out = out[k1]

        # asarray
        if asarray is True:
            out = np.asarray(out)

        # set value in dout
        if flatten is True:
            if sep is None:
                dout[k0] = out
            else:
                dout[sep.join(k0)] = out
        else:
            setInDict(dout, k0, out)

    # ---------------
    # prepare output

    if copy is True:
        return deepcopy(dout)
    else:
        return dout


def _blend_dicts(
    dbase=None,
    dextra=None,
    extra_key=None,
    flatten=None,
):

    # -------------
    # check inputs

    extra_key = _generic_check._check_var(
        extra_key, 'extra_key',
        default='__type',
        types=str,
    )

    assert isinstance(dbase, dict) and isinstance(dextra, dict)

    # ------------
    # blend

    if flatten is True:

        for k0, v0 in dextra.items():
            dbase[f"{k0}{extra_key}"] = v0

    else:
        msg = "Blended dict not implement for flatten = False"
        raise NotImplementedError(msg)

    return dbase


def _reshape_dict(k0, v0, dinit={}, sep=None):
    """ Populate dinit """

    assert isinstance(dinit, dict), dinit

    if sep is None:
        lk = k0
    else:
        lk = k0.split(sep)

    k0 = lk[0]

    if len(lk) == 2:
        if k0 not in dinit.keys():
            dinit[k0] = {}
        assert isinstance(dinit[k0], dict), (k0, dinit[k0])
        dinit[k0].update({lk[1]: v0})

    elif len(lk) > 2:
        if k0 not in dinit.keys():
            dinit[k0] = {}
        assert isinstance(dinit[k0], dict), (k0, dinit[k0])

        knew = lk[1:] if sep is None else sep.join(lk[1:])
        _reshape_dict(knew, v0, dinit=dinit[k0], sep=sep)

    else:
        assert k0 not in dinit.keys()
        dinit[k0] = v0


def reshape_dict(din, sep=None):
    """ Return a reshaped version of the input dict, according to sep """

    # ------------
    # check inputs

    if sep is not None:
        sep = _generic_check._check_var(
            sep, 'sep',
            default='.',
            types=str,
        )

    # ------------------------
    # Get all individual keys

    dout = {}
    for k0, v0 in din.items():
        _reshape_dict(k0, v0, dinit=dout, sep=sep)

    return dout


# ########################################################################
# ########################################################################
#            Find all indices of a subsequence in sequence
# ########################################################################


def KnuthMorrisPratt(text, pattern):

    """ Yields all starting positions of copies of the pattern in the sequence

    Calling conventions are similar to string.find, but its arguments can be
    lists or iterators, not just strings, it returns all matches, not just
    the first one, and it does not need the whole text in memory at once.
    Whenever it yields, it will have read the text exactly up to and including
    the match that caused the yield.
    """

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos-shift]:
            shift += shifts[pos-shift]
        shifts[pos+1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == len(pattern) or \
              matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos


import numpy as np
import scipy.sparse as scpsp


from . import _generic_check


# #################################################################
# #################################################################
#                   Pretty printing
# #################################################################


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

    # headers
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

    # content
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

    # sep
    sep = _generic_check._check_var(
        sep, 'sep',
        default=' ',
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


# #################################################################
# #################################################################
#                   Compare dict
# #################################################################


def _compare_dict_verb_return(dout, returnas, verb):

    if len(dout) > 0:
        if verb:
            lstr = [f"{k0}: {v0}" for k0, v0 in dout.items()]
            msg = (
                "The following differences have been found:\n"
                + "\n".join(lstr)
            )
            print(msg)
        if returnas:
            return dout
        else:
            return False

    else:
        return True


def compare_dict(d0=None, d1=None, dname=None, returnas=None, verb=None):

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
        default=False,
        types=bool,
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
        dout[kroot] = f'different keys: {lk0} vs {lk1}'
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
            if d0[k0].shape != d1[k0].shape:
                dkeys[key] = f'!= shapes ({d0[k0].shape} vs {d1[k0].shape})'
            elif not np.allclose(d0[k0], d1[k0], equal_nan=True):
                dkeys[key] = "not allclose"

        # sparse arrays
        elif scpsp.issparse(d0[k0]):
            if d0[k0].shape != d1[k0].shape:
                dkeys[key] = f'!= shapes ({d0[k0].shape} vs {d1[k0].shape})'
            elif not np.allclose(d0[k0].data, d1[k0].data, equal_nan=True):
                dkeys[key] = "not allclose"

        # lists and tuples
        elif isinstance(d0[k0], (list, tuple)):
            try:
                if not d0[k0] == d1[k0]:
                    dkeys[key] = "!= list/tuple values"
            except Exception as err:
                msg = (
                    f"Don't know how to handle {key}:\n"
                    f"\t- type: {type(d0[k0])}\n"
                    f"\t- value: {d0[k0]}\n"
                )
                raise NotImplementedError(msg)

        # functions
        elif callable(d0[k0]):
            if d0[k0] == d1[k0]:
                dkeys[key] = "!= callable"

        # dict
        elif isinstance(d0[k0], dict):
            dd = compare_dict(
                d0=d0[k0],
                d1=d1[k0],
                dname=key,
                returnas=True,
                verb=False,
            )
            if isinstance(dd, dict):
                dkeys.update(dd)

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


def compare_obj(obj0=None, obj1=None, returnas=None, verb=None):
    """ Compare thje content of 2 instances """

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
        d0=obj0.to_dict(),
        d1=obj1.to_dict(),
        dname=None,
        returnas=returnas,
        verb=verb,
    )


# #################################################################
# #################################################################
#               Flatten / reshape dict                 
# #################################################################


def flatten_dict(din=None, parent_key=None, sep=None):
    """ Return a flattened version of the input dict """

    # ------------
    # check inputs

    sep = _generic_check._check_var(
        sep, 'sep',
        default='.',
        types=str,
    )

    if parent_key is not None:
        parent_key = _generic_check._check_var(
            parent_key, 'parent_key',
            default='.',
            types=str,
        )

    # --------
    # flatten

    items = []
    for k0, v0 in din.items():

        # key
        if parent_key is None:
            key = k0
        else:
            key = f'{parent_key}{sep}{k0}'

        # value
        if isinstance(v0, dict):
            items.extend(flatten_dict(v0, key, sep=sep).items())
        else:
            items.append((key, v0))

    return dict(items)


def _reshape_dict(k0, v0, dinit={}, sep=None):
    """ Populate dinit """

    lk = k0.split(sep)
    k0 = k0 if len(lk) == 1 else lk[0]

    if len(lk) == 2:
        if k0 not in dinit.keys():
            dinit[k0] = {}
        assert isinstance(dinit[k0], dict)
        dinit[k0].update({lk[1]: v0})

    elif len(lk) > 2:
        if k0 not in dinit.keys():
            dinit[k0] = {}
        _reshape_dict(sep.join(lk[1:]), v0, dinit=dinit[k0], sep=sep)

    else:
        assert k0 not in dinit.keys()
        dinit[k0] = v0


def reshape_dict(din, sep=None):
    """ Return a reshaped version of the input dict, according to sep """

    # ------------
    # check inputs

    sep = _generic_check._check_var(
        sep, 'sep',
        default='.',
        types=str,
    )

    # ------------
    # Get all individual keys

    dout = {}
    for k0, v0 in din.items():
        _reshape_dict(k0, v0, dinit=dout, sep=sep)
    return dout

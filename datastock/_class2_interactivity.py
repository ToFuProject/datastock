

import itertools as itt


import numpy as np
import scipy.sparse as scpsparse


from . import _generic_check
from . import _generic_utils
from . import _class1_compute


_INCREMENTS = [1, 10]
_DKEYS = {
    'control': {'val': False, 'action': 'generic'},
    'ctrl': {'val': False, 'action': 'generic'},
    'shift': {'val': False, 'action': 'generic'},
    'alt': {'val': False, 'action': 'generic'},
    'left': {'val': False, 'action': 'move'},
    'right': {'val': False, 'action': 'move'},
    'up': {'val': False, 'action': 'move'},
    'down': {'val': False, 'action': 'move'},
}


_DCOMMANDS = {
    'shift + left clic': 'add a slice at selected location',
    'left clic': 'move currently selected slice to selected location',
    'ctrl + left clic': 'remove all visible slices',
    '0, 1, 2...': 'select slice number 0, 1, 2...',
    'f1, f2, f3...': 'select group of slice number 1, 2, 3...',
    'left/right/up/down arrows': 'increment current slice index by 1',
    'alt + left/right/up/down arrows': 'increment current slice index by 10',
}


# #############################################################################
# #############################################################################
#            show commands
# #############################################################################


def show_commands(
    verb=None,
    returnas=None,
):

    # ------------
    # check inputs

    verb = _generic_check._check_var(
        verb, 'verb',
        default=True,
        types=bool,
    )

    returnas = _generic_check._check_var(
        returnas, 'returnas',
        default=False,
        allowed=[False, str, dict],
    )

    # ------------
    # get dcommands

    dcom = dict(_DCOMMANDS)

    # --------------
    # col and ar

    lcol = [['command', 'description']]
    lar = [[
        [
            k0,
            v0,
        ]
        for k0, v0 in dcom.items()
    ]]

    # -------
    # return

    out = _generic_utils.pretty_print(
        headers=lcol,
        content=lar,
        verb=verb,
        returnas=False if returnas is dict else returnas,
    )
    if returnas is dict:
        return dcom
    else:
        return out



# #############################################################################
# #############################################################################
#            setup interactivity
# #############################################################################


def _setup_dgroup(
    dgroup=None,
    dobj0=None,
    dref0=None,
):
    """ Check dgroup, if None return current group

    newgroup = True if dgroup is provided
    """

    newgroup = True
    if dgroup is None and dobj0.get('group') is not None:
        dgroup = dobj0['group']
        newgroup = False

    if not isinstance(dgroup, dict):
        msg = 'Arg dgroup must be a dict!'
        raise Exception(msg)

    dc = {
        k0: (
            isinstance(k0, str),
            isinstance(v0, dict),
            isinstance(v0.get('ref'), list),
            isinstance(v0.get('data'), list),
            len(v0['ref']) == len(v0['data']),
            all([ss in dref0.keys() for ss in v0['ref']])
        )
        for k0, v0 in dgroup.items()
    }
    dc = {k0: v0 for k0, v0 in dc.items() if not all(v0)}
    if len(dc) > 0:
        lstr = [f'\t- {k0}: {v0}' for k0, v0 in dc.items()]
        msg = "Non-valid form for arg dgroup:\n" + "\n".join(lstr)
        raise Exception(msg)

    for k0, v0 in dgroup.items():
        if v0.get('nmax') is None:
            dgroup[k0]['nmax'] = 0
        dgroup[k0]['nmaxcur'] = 0
        dgroup[k0]['indcur'] = 0

    return dgroup, newgroup


def _setup_dinc(dinc=None, lparam_ref=None, dref0=None):

    newinc = True
    if dinc is None:
        if 'inc' in lparam_ref:
            dinc = {k0: v0['inc'] for k0, v0 in dref0.items()}
            newinc = False
        else:
            dinc = {k0: _INCREMENTS for k0 in dref0.keys()}

    elif isinstance(dinc, list) and len(dinc) == 2:
        dinc = {k0: dinc for k0 in dref0.keys()}

    elif isinstance(dinc, dict):
        c0 = all([
            ss in dref0.keys()
            and isinstance(vv, list)
            and len(vv) == 2
            for ss, vv in dinc.items()
        ])
        if not c0:
            msg = (
                "Arg dinc must be a dict of type {ref0: [inc0, inc1]}\n"
                f"\t- Provided: {dinc}"
            )
            raise Exception(msg)

        for k0 in dref0.keys():
            if k0 not in dinc.keys():
                dinc[k0] = _INCREMENTS
    else:
        msg = (
            "Arg dinc must be a dict of type {ref0: [inc0, inc1]}\n"
            f"\t- Provided: {dinc}"
        )
        raise Exception(msg)
    return dinc, newinc


def _setup_drefgroup(dref0=None, dgroup=None):

    drefgroup = dict.fromkeys(dref0.keys())
    for k0, v0 in dref0.items():
        lg = [k1 for k1, v1 in dgroup.items() if k0 in v1['ref']]
        if len(lg) > 1:
            msg = f"Ref {k0} has no/several groups!\n\t- found: {lg}"
            raise Exception(msg)
        elif len(lg) == 0:
            lg = [None]
        drefgroup[k0] = lg[0]

    return drefgroup


def _setup_mobile(
    dmobile=None,
    dref=None,
    ddata=None,
):
    """ update dmobile with group, group_vis and func """

    for k0, v0 in dmobile.items():

        # group
        groups = [
            [dref[r1]['group'] for r1 in r0]
            for r0 in v0['refs']
        ]
        dmobile[k0]['group'] = tuple(set(itt.chain.from_iterable(groups)))

        # group _vis
        if dmobile[k0]['group_vis'] is None:
            dmobile[k0]['group_vis'] = dmobile[k0]['group']

        if isinstance(dmobile[k0]['group_vis'], str):
           dmobile[k0]['group_vis'] = (dmobile[k0]['group_vis'],)
        c0 = (
            isinstance(dmobile[k0]['group_vis'], tuple)
            and all([
                isinstance(ss, str)
                and ss in dmobile[k0]['group']
                for ss in dmobile[k0]['group_vis']
            ])
        )
        if not c0:
            msg = (
                f"dmobile['{k0}']['group_vis'] must be:\n"
                f"\t- a tuple of groups in dmobile['{k0}']['group']\n"
                "\t- specifies which groups determine visibility\n"
                f"If None: set to dmobile['{k0}']['group']"
                f" = {dmobile[k0]['group']}\n"
                f"Provided: {dmobile[k0]['group_vis']}"
            )
            raise Exception(msg)

        # functions for updating
        nocc = len(set(dmobile[k0]['dtype']))

        dmobile[k0]['func_set_data'] = [
           get_fupdate(
               handle=v0['handle'],
               dtype=dmobile[k0]['dtype'][ii],
               norm=None,
               bstr=v0.get('bstr'),
           )
           for ii in range(nocc)
        ]

        # functions for slicing
        dmobile[k0]['func_slice'] = [
            _class1_compute._get_slice(
                laxis=dmobile[k0]['axis'][ii],
                ndim=(
                    1 if dmobile[k0]['data'][ii] == 'index'
                    else ddata[dmobile[k0]['data'][ii]]['data'].ndim
                )
            )
            for ii in range(nocc)
        ]


def _setup_keys(dkeys=None, dgroup=None):
    """ return dkeys """

    if dkeys is None:
        dkeys = dict(_DKEYS)

    # add key for switching groups
    dkeys.update({
        v0.get('key', f'f{ii+1}'): {
            'group': k0,
            'val': False,
            'action': 'group',
        }
        for ii, (k0, v0) in enumerate(dgroup.items())
    })

    # add keys for switching indices within groups
    nMax = np.max([v0['nmax'] for v0 in dgroup.values()])
    dkeys.update({
        str(ii): {'ind': ii, 'val': False, 'action': 'indices'}
        for ii in range(0, nMax)
    })
    return dkeys


# #############################################################################
# #############################################################################
#           data of mobile based on indices
# #############################################################################


def _set_dbck(
    lax=None,
    daxes=None,
    dcanvas=None,
    dmobile=None,
    event=None,
):
    """ Update background of relevant axes (ex: in case of resizing) """

    # first allow resizing to happen
    lcan = set([daxes[k0]['canvas'] for k0 in lax])

    # Make all invisible
    for k0 in lax:
        for k1 in daxes[k0]['mobile']:
            dmobile[k1]['handle'].set_visible(False)

    # Draw and reset bck
    lcan = set([daxes[k0]['canvas'] for k0 in lax])
    for k0 in lcan:
        dcanvas[k0]['handle'].draw()

    # set bck (= bbox copy)
    for k0 in lax:
        #ax.draw(self.can.renderer)
        daxes[k0]['bck'] = dcanvas[
            daxes[k0]['canvas']
        ]['handle'].copy_from_bbox(daxes[k0]['handle'].bbox)

    # Redraw
    for k0 in lax:
        for k1 in daxes[k0]['mobile']:
            dmobile[k1]['handle'].set_visible(dmobile[k1]['visible'])
            #ax.draw(self.can.renderer)

    for k0 in lcan:
        dcanvas[k0]['handle'].draw()


# #############################################################################
# #############################################################################
#           Update number of visible indices
# #############################################################################


def _get_nn_ii_group(
    nmax=None,
    nmaxcur=None,
    indcur=None,
    ctrl=None,
    shift=None,
    group=None,
):
    """"""

    if shift and nmaxcur == nmax:
        msg = f"Max nb. of plots reached for group '{group}': {nmax}"
        print(msg)
        return False

    if ctrl:
        nn = 0
        ii = 0
    elif shift:
        nn = int(nmaxcur) + 1
        ii = nn - 1
    else:
        nn = int(nmaxcur)
        ii = int(indcur)
    return nn, ii


def _update_indices_nb(group=None, dgroup=None, ctrl=None, shift=None):
    """"""
    out = _get_nn_ii_group(
        nmax=dgroup[group]['nmax'],
        nmaxcur=dgroup[group]['nmaxcur'],
        indcur=dgroup[group]['indcur'],
        ctrl=ctrl,
        shift=shift,
        group=group,
    )
    if out is False:
        return False
    else:
        dgroup[group]['nmaxcur'] = out[0]
        dgroup[group]['indcur'] = out[1]


# #############################################################################
# #############################################################################
#            mouseclic utility
# #############################################################################


def _get_ix_for_refx_only_1or2d(
    cur_data=None,
    cur_ref=None,
    eventdata=None,
    # resources
    ddata=None,
    dref=None,
    dgroup=None,
):
    """
    Return the index coreesponding to:

        - event data (mouseclic)
        - desired cur_data
        - desired cur_ref

    Handles 1d or 2d arrays (useful for profile1d)

    """

    monot = None
    c0 = (
        cur_data == 'index'
        or ddata[cur_data]['data'].dtype.type == np.str_
    )
    if c0:
        cd = 'index'
    else:
        monot = ddata[cur_data]['monot'] == (True,)
        cd = ddata[cur_data]['data']

    # prepare input for >= 2d data
    if (isinstance(cd, str) and cd == 'index') or cd.ndim == 1:
        laxis, linds = None, None

    elif cd.ndim == 2:
        laxis = [
            (
                ii,
                dref[rr]['indices'][dgroup[dref[cur_ref]['group']]['indcur']],
            )
            for ii, rr in enumerate(ddata[cur_data]['ref'])
            if rr != cur_ref
        ]
        laxis, linds = zip(*laxis)
    else:
        raise NotImplementedError()

    # get index of datax corresponding to clicked point
    return _class1_compute._get_index_from_data(
        data=cd,
        data_pick=np.r_[eventdata],
        monot=monot,
        # for >= 2d data
        laxis=laxis,
        linds=linds,
    )[0]


# #############################################################################
# #############################################################################
#           data of mobile based on indices
# #############################################################################


def get_fupdate(handle=None, dtype=None, norm=None, bstr=None):

    # Note: set_xdata() and set_ydata() do not accept scalar values
    #       Deprecation warning since matplotlib 3.7
    # see https://github.com/matplotlib/matplotlib/pull/22329
    # see https://github.com/matplotlib/matplotlib/issues/28927

    if dtype == 'xdata':
        def func(val, handle=handle):
            handle.set_xdata(np.atleast_1d(val))
    elif dtype == 'ydata':
        def func(val, handle=handle):
            handle.set_ydata(np.atleast_1d(val))
    elif dtype in ['data']:   # Also works for imshow
        def func(val, handle=handle):
            handle.set_data(val)
    elif dtype in ['data.T']:   # Also works for imshow
        def func(val, handle=handle):
            handle.set_data(val.T)
    elif dtype in ['xy']:   # works for filled polygons
        def func(val, handle=handle):
            handle.set_xy(val)
    elif dtype in ['alpha']:   # Also works for imshow
        def func(val, handle=handle, norm=norm):
            handle.set_alpha(norm(val))
    elif dtype == 'txt':
        func = lambda val, handle=handle, bstr=bstr: handle.set_text(bstr.format(val))
    elif dtype == 'x':
        def func(val, handle=handle):
            handle.set_x(val)
    elif dtype == 'y':
        def func(val, handle=handle):
            handle.set_y(val)
    elif dtype == 'position':
        def func(val, handle=handle):
            handle.set_position(val)
    else:
        msg = f'Unknown mobile dtype: {dtype}'
        raise Exception(msg)
    return func


def _update_mobile(k0=None, dmobile=None, dref=None, ddata=None):
    """ Update mobile objects data """

    # All ref do not necessarily have the same nb of indices
    iref = [
        [
            dref[r1]['indices'][
                min(dmobile[k0]['ind'], len(dref[r1]['indices']) - 1)
            ]
            for r1 in r0
        ]
        for r0 in dmobile[k0]['refs']
    ]

    nocc = len(set(dmobile[k0]['dtype']))

    for ii in range(nocc):
        c0 = (
            dmobile[k0]['data'][ii] == 'index'
            or ddata[dmobile[k0]['data'][ii]]['data'].dtype.type == np.str_
        )
        if c0:
            dmobile[k0]['func_set_data'][ii](*iref[ii])

        elif scpsparse.issparse(ddata[dmobile[k0]['data'][ii]]['data']):
            if ddata[dmobile[k0]['data'][ii]]['data'].ndim <= 2:
                dmobile[k0]['func_set_data'][ii](
                    ddata[dmobile[k0]['data'][ii]]['data'][
                        dmobile[k0]['func_slice'][ii](*iref[ii])
                    ].todense()
                )
            else:
                msg = "Sparse data of dim > 2: not handled yet"
                raise Exception(msg)

        else:
            dmobile[k0]['func_set_data'][ii](
                ddata[dmobile[k0]['data'][ii]]['data'][
                    dmobile[k0]['func_slice'][ii](*iref[ii])
                ]
            )


    # if nocc == 1:
        # c0 = (
            # dmobile[k0]['data'][0] == 'index'
            # or ddata[dmobile[k0]['data'][0]]['data'].dtype.type == np.str_
        # )
        # if c0:
            # dmobile[k0]['func_set_data'][0](*iref)

        # elif scpsparse.issparse(ddata[dmobile[k0]['data'][0]]['data']):
            # if ddata[dmobile[k0]['data'][0]]['data'].ndim <= 2:
                # dmobile[k0]['func_set_data'][0](
                    # ddata[dmobile[k0]['data'][0]]['data'][
                        # dmobile[k0]['func_slice'][0](*iref)
                    # ].todense()
                # )
            # else:
                # msg = "Sparse data of dim > 2: not handled yet"
                # raise Exception(msg)

        # else:
            # dmobile[k0]['func_set_data'][0](
                # ddata[dmobile[k0]['data'][0]]['data'][
                    # dmobile[k0]['func_slice'][0](*iref)
                # ]
            # )

    # else:
        # for ii in range(nocc):
            # c0 = (
                # dmobile[k0]['data'][0] == 'index'
                # or ddata[dmobile[k0]['data'][0]]['data'].dtype.type == np.str_
            # )
            # if c0:
                # dmobile[k0]['func_set_data'][ii](iref[ii])

            # elif scpsparse.issparse(ddata[dmobile[k0]['data'][ii]]['data']):
                # if ddata[dmobile[k0]['data'][ii]]['data'].ndim <= 2:
                    # dmobile[k0]['func_set_data'][ii](
                        # ddata[dmobile[k0]['data'][ii]]['data'][
                            # dmobile[k0]['func_slice'][ii](iref[ii])
                        # ].todense()
                    # )
                # else:
                    # msg = "Sparse data of dim > 2: not handled yet"
                    # raise Exception(msg)

            # else:
                # dmobile[k0]['func_set_data'][ii](
                    # ddata[dmobile[k0]['data'][ii]]['data'][
                        # dmobile[k0]['func_slice'][ii](iref[ii])
                    # ]
                # )

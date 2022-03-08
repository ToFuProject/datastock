


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
#           data of mobile based on indices
# #############################################################################


def get_fupdate(handle=None, dtype=None, norm=None, bstr=None):
    if dtype == 'xdata':
        func = lambda val, handle=handle: handle.set_xdata(val)
    elif dtype == 'ydata':
        func = lambda val, handle=handle: handle.set_ydata(val)
    elif dtype in ['data']:   # Also works for imshow
        func = lambda val, handle=handle: handle.set_data(val)
    elif dtype in ['alpha']:   # Also works for imshow
        func = lambda val, handle=handle, norm=norm: handle.set_alpha(norm(val))
    elif dtype == 'txt':
        func = lambda val, handle=handle, bstr=bstr: handle.set_text(bstr.format(val))
    return func


def _get_slice(laxis=None, ndim=None):

    nax = len(laxis)
    assert nax in range(1, ndim + 1)

    if ndim == nax:
        def fslice(*args):
            return args

    else:
        def fslice(*args, laxis=laxis):
            ind = [slice(None) for ii in range(ndim)]
            for ii, aa in enumerate(args):
                ind[laxis[ii]] = aa
            return tuple(ind)

    return fslice


def get_slice(nocc=None, laxis=None, lndim=None):

    if nocc == 1:
        return [_get_slice(laxis=laxis, ndim=lndim[0])]

    elif nocc == 2:
        return [
            _get_slice(laxis=[laxis[0]], ndim=lndim[0]),
            _get_slice(laxis=[laxis[1]], ndim=lndim[1]),
        ]


def _update_mobile(k0=None, dmobile=None, dref=None, ddata=None):
    """ Update mobile objects data """

    func = dmobile[k0]['func']
    kref = dmobile[k0]['ref']
    kdata = dmobile[k0]['data']

    # All ref do not necessarily have the same nb of indices
    iref = [
        dref[rr]['indices'][
            min(dmobile[k0]['ind'], len(dref[rr]['indices']) - 1)
        ]
        for rr in dmobile[k0]['ref']
    ]
    nocc = len(set(dmobile[k0]['dtype']))
    if nocc == 1 and dmobile[k0]['data'][0] == 'index':
        dmobile[k0]['func_set_data'][0](*iref)

    elif nocc == 1:
        dmobile[k0]['func_set_data'][0](
            ddata[dmobile[k0]['data'][0]]['data'][
                dmobile[k0]['func_slice'][0](*iref)
            ]
        )

    else:
        for ii in range(nocc):
            dmobile[k0]['func_set_data'][ii](
                ddata[dmobile[k0]['data'][ii]]['data'][
                    dmobile[k0]['func_slice'][ii](iref[ii])
                ]
            )

# coding utf-8


# Common
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


# library-specific
from . import _generic_check
from . import _plot_text


# #############################################################
# #############################################################
#                       Main
# #############################################################


def main(
    # parameters
    coll=None,
    key=None,
    lab=None,
    dkeys=None,
    ind=None,
    dvminmax=None,
    dscale=None,
    cmap=None,
    aspect=None,
    nmax=None,
    color_dict=None,
    lkeys=None,
    bstr_dict=None,
    rotation=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    label=None,
    # unused
    **kwdargs,
):

    # --------------
    #  Prepare data
    # --------------

    data = coll.ddata[key]['data']
    if hasattr(data, 'nnz'):
        data = data.toarray()
    assert data.ndim == len(coll.ddata[key]['ref']) == 1
    n0, = data.shape

    for k0, v0 in dkeys.items():
        if v0['ref'] is not None:
            dkeys[k0]

    # --------------
    # prepare figure
    # --------------

    if dax is None:
        dax = _create_axes(
            fs=fs,
            dmargin=dmargin,
        )

    dax = _generic_check._check_dax(dax=dax, main='matrix')

    # ---------------
    # plot fixed part
    # ---------------

    axtype = 'matrix'
    lkax = [kk for kk, vv in dax.items() if axtype in vv['type']]
    for kax in lkax:
        ax = dax[kax]['handle']

        ax.plot(
            coll.ddata[dkeys['X']['data']]['data'],
            data,
            color='k',
            marker='.',
            ms=6,
        )

        # plt.colorbar(im, ax=ax, **dcolorbar)
        if dleg is not False:
            ax.legend(**dleg)

    # ----------------
    # define and set dgroup
    # ---------------

    dgroup = {
        'X': {
            'ref': [dkeys['X']['ref']],
            'data': ['index'],
            'nmax': nmax,
        },
    }

    # ----------------
    # plot mobile part
    # ---------------

    axtype = 'matrix'
    lkax = [kk for kk, vv in dax.items() if axtype in vv['type']]
    for kax in lkax:
        ax = dax[kax]['handle']

        # ind0, ind1
        for ii in range(nmax):
            lv = ax.axvline(ind[0], c=color_dict['X'][ii], lw=1., ls='-')

            # update coll
            kv = f'{key}_v{ii:02.0f}'
            coll.add_mobile(
                key=kv,
                handle=lv,
                refs=dkeys['X']['ref'],
                data=dkeys['X']['data'],
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[dkeys['X']['ref']], datax=[dkeys['X']['data']])

    # ---------
    # add text
    # ---------------

    kax = 'text'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            key=key,
            ax=ax,
            ref=dkeys['X']['ref'],
            group='X',
            ind=ind[0],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    # ---------------
    # labels
    # ---------------

    if label:
        _label_axes(
            coll=coll,
            data_lab=lab,
            dax=dax,
            key=key,
            dkeys=dkeys,
            dvminmax=dvminmax,
            rotation=rotation,
        )

    return coll, dax, dgroup


# #############################################################
# #############################################################
#                       Create axes
# #############################################################


def _create_axes(
    fs=None,
    dmargin=None,
):

    # ---------------
    # check / prepare
    # ---------------

    # figure size
    if fs is None:
        fs = (12, 8)

    # dict of margins
    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.95,
            'bottom': 0.10, 'top': 0.90,
            'hspace': 0.15, 'wspace': 0.2,
        }

    # -----------------
    # create
    # ---------------

    # figure
    fig = plt.figure(figsize=fs)

    # axes grid
    gs = gridspec.GridSpec(ncols=4, nrows=1, **dmargin)

    # axes for data
    ax0 = fig.add_subplot(gs[0, :3], aspect='auto')

    # axes for text
    ax1 = fig.add_subplot(gs[0, 3], frameon=False)

    # --------------
    # assemble dax
    # --------------

    dax = {
        'matrix': {'handle': ax0},
        'text': {'handle': ax1},
    }

    return dax


# #############################################################
# #############################################################
#                   Label axes
# #############################################################


def _label_axes(
    coll=None,
    data_lab=None,
    dax=None,
    tit=None,
    key=None,
    dkeys=None,
    dvminmax=None,
    inverty=None,
    rotation=None,
):

    # ------
    # fig
    # ------

    fig = list(dax.values())[0]['handle'].figure
    fig.suptitle(tit, size=14, fontweight='bold')

    # -------------------
    # axes for data
    # -------------------

    axtype = 'matrix'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xlabel(dkeys['X']['lab'], size=12, fontweight='bold')
        ax.set_ylabel(data_lab, size=12, fontweight='bold')

        if np.isfinite(dvminmax['X']['min']):
            ax.set_xlim(left=dvminmax['X']['min'])
        if np.isfinite(dvminmax['X']['max']):
            ax.set_xlim(right=dvminmax['X']['max'])

        if np.isfinite(dvminmax['data']['min']):
            ax.set_ylim(bottom=dvminmax['data']['min'])
        if np.isfinite(dvminmax['data']['max']):
            ax.set_ylim(top=dvminmax['data']['max'])

        # x text ticks
        if dkeys['X']['str'] is not False:
            ax.set_xticks(coll.ddata[dkeys['X']['data']]['data'])
            ax.set_xticklabels(
                dkeys['X']['str'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )

    # -------------
    # axes for text
    # -------------

    axtype = 'text'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xticks([])
        ax.set_yticks([])

    return dax
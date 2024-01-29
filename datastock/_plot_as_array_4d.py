# coding utf-8


# Common
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


# library-specific
from . import _generic_check
from . import _class1_compute
from . import _plot_text


# #############################################################
# #############################################################
#                       Main
# #############################################################


def main(
    # parameters
    coll=None,
    key=None,
    dkeys=None,
    dscale=None,
    dvminmax=None,
    ind=None,
    cmap=None,
    aspect=None,
    nmax=None,
    color_dict=None,
    lkeys=None,
    bstr_dict=None,
    rotation=None,
    inverty=None,
    bck=None,
    interp=None,
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
    ndim = data.ndim
    assert ndim == len(coll.ddata[key]['ref']) == 4
    n0, n1, n2, n3 = data.shape

    # -----------------
    #  prepare slicing
    # -----------------

    # here slice X => slice in dim Y and vice-versa
    sliZ2 = _class1_compute._get_slice(
        laxis=[dkeys['keyZ']['axis'], dkeys['keyU']['axis']],
        ndim=4,
    )

    # check if transpose is necessary
    if dkeys['keyX']['axis'] < dkeys['keyY']['axis']:
        datatype = 'data.T'
        dataplot = data[sliZ2(ind[2], ind[3])].T
    else:
        datatype = 'data'
        dataplot = data[sliZ2(ind[2], ind[3])]

    # ----------------------
    #  labels and data
    # ----------------------

    extent = (
        coll.ddata[dkeys['keyX']['data']]['data'][0] - dkeys['keyX']['d2'],
        coll.ddata[dkeys['keyX']['data']]['data'][-1] + dkeys['keyX']['d2'],
        coll.ddata[dkeys['keyY']['data']]['data'][0] - dkeys['keyY']['d2'],
        coll.ddata[dkeys['keyY']['data']]['data'][-1] + dkeys['keyY']['d2'],
    )

    # --------------
    # plot - prepare
    # --------------

    if dax is None:
        dax = _create_axes(
            fs=fs,
            dmargin=dmargin,
        )

    dax = _generic_check._check_dax(dax=dax, main='matrix')

    # ----------------------------------
    # plot fixed parts (traces envelops)
    # ----------------------------------

    for ss in ['Z', 'U']:

        k0 = f"key{ss}"
        axis = dkeys[k0]['axis']

        axtype = f'traces{ss}'
        lax = [k1 for k1, v1 in dax.items() if axtype in v1['type']]
        if len(lax) == 1:
            kax = lax[0]
            ax = dax[kax]['handle']
            dat = coll.ddata[dkeys[k0]['data']]['data']

            if bck == 'lines':
                shap = list(data.shape)
                shap[axis] = 1
                nan = np.full(shap, np.nan)
                bckl = np.concatenate((data, nan), axis=axis)
                bckl = np.swapaxes(bckl, axis, -1).ravel()
                dat = np.tile(np.r_[dat, np.nan], int(np.prod(shap)))
                ax.plot(
                    dat,
                    bckl,
                    c=(0.8, 0.8, 0.8),
                    ls='-',
                    lw=1.,
                    marker='None',
                )
            else:
                tax = tuple([
                    v1['axis'] for k1, v1 in dkeys.items() if k1 != k0
                ])
                bckenv = [
                    np.nanmin(data, axis=tax),
                    np.nanmax(data, axis=tax),
                ]
                ax.fill_between(
                    dat,
                    bckenv[0],
                    bckenv[1],
                    facecolor=(0.8, 0.8, 0.8, 0.8),
                    edgecolor='None',
                )

    # ----------------
    # define and set dgroup
    # ----------------

    dgroup = {
        'X': {
            'ref': [dkeys['keyX']['ref']],
            'data': ['index'],
            'nmax': nmax,
        },
        'Y': {
            'ref': [dkeys['keyY']['ref']],
            'data': ['index'],
            'nmax': nmax,
        },
        'Z': {
            'ref': [dkeys['keyZ']['ref']],
            'data': ['index'],
            'nmax': 1,
        },
        'U': {
            'ref': [dkeys['keyU']['ref']],
            'data': ['index'],
            'nmax': 1,
        },
    }

    # -----------------
    # plot mobile parts
    # -----------------

    # matrix
    axtype = 'matrix'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        # image
        im = ax.imshow(
            dataplot,
            extent=extent,
            interpolation=interp,
            origin='lower',
            aspect=aspect,
            cmap=cmap,
            vmin=dvminmax['data']['min'],
            vmax=dvminmax['data']['max'],
        )

        km = f'{key}_im'
        coll.add_mobile(
            key=km,
            handle=im,
            refs=((dkeys['keyZ']['ref'], dkeys['keyU']['ref']),),
            data=key,
            dtype=datatype,
            axes=kax,
            ind=0,
        )

        if inverty is True:
            ax.invert_yaxis()

        # ind0, ind1
        for ii in range(nmax):

            lh = ax.axhline(
                coll.ddata[dkeys['keyY']['data']]['data'][ind[1]],
                c=color_dict['X'][ii],
                lw=1.,
                ls='-',
            )

            lv = ax.axvline(
                coll.ddata[dkeys['keyX']['data']]['data'][ind[0]],
                c=color_dict['Y'][ii],
                lw=1.,
                ls='-',
            )

            mi, = ax.plot(
                coll.ddata[dkeys['keyX']['data']]['data'][ind[0]],
                coll.ddata[dkeys['keyY']['data']]['data'][ind[1]],
                marker='s',
                ms=6,
                markeredgecolor=color_dict['X'][ii],
                markerfacecolor='None',
            )

            # update coll
            kh = f'{key}_h{ii:02.0f}'
            kv = f'{key}_v{ii:02.0f}'
            coll.add_mobile(
                key=kh,
                handle=lh,
                refs=dkeys['keyY']['ref'],
                data=dkeys['keyY']['key'],
                dtype='ydata',
                axes=kax,
                ind=ii,
            )
            coll.add_mobile(
                key=kv,
                handle=lv,
                refs=dkeys['keyX']['ref'],
                data=dkeys['keyX']['key'],
                dtype='xdata',
                axes=kax,
                ind=ii,
            )
            km = f'{key}_m{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=mi,
                refs=[dkeys['keyX']['ref'], dkeys['keyY']['ref']],
                data=[dkeys['keyX']['key'], dkeys['keyY']['key']],
                dtype=['xdata', 'ydata'],
                axes=kax,
                ind=ii,
            )

        dax[kax].update(
            refx=[dkeys['keyX']['ref']],
            refy=[dkeys['keyY']['ref']],
            datax=[dkeys['keyX']['key']],
            datay=[dkeys['keyY']['key']],
        )

    # --------------
    # slices
    # --------------

    lslices = [('X', 'horizontal'), ('Y', 'vertical')]
    for i0, (ss, axtype) in enumerate(lslices):
        lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
        if len(lax) == 1:
            kax = lax[0]
            ax = dax[kax]['handle']
            k0 = f'key{ss}'
            sli = dkeys[k0]['sli']
            iind = i0
            args = [ind[jj] for jj in range(ndim) if jj != iind]
            refs = tuple([
                dkeys[f"key{k1}"]['ref'] for k1 in ['X', 'Y', 'Z', 'U']
                if k1 != ss
            ])
            dat = coll.ddata[dkeys[k0]['data']]['data']

            # ---- DEBUG ----
            print()
            print(ss, k0)
            print(iind)
            print(dkeys[k0])
            # ---------------

            for ii in range(nmax):
                if ss == 'Y':
                    l0, = ax.plot(
                        data[sli(*args)],
                        dat,
                        ls='-',
                        marker='.',
                        lw=1.,
                        color=color_dict[ss][ii],
                        label=f'ind0 = {ind[iind]}',
                    )
                    xydata = 'xdata'
                    km = f'{key}_vprof{ii:02.0f}'
                else:
                    l0, = ax.plot(
                        dat,
                        data[sli(*args)],
                        ls='-',
                        marker='.',
                        lw=1.,
                        color=color_dict[ss][ii],
                        label=f'ind0 = {ind[iind]}',
                    )
                    xydata = 'ydata'
                    km = f'{key}_vhor{ii:02.0f}'

                coll.add_mobile(
                    key=km,
                    handle=l0,
                    refs=(refs,),
                    data=[key],
                    dtype=[xydata],
                    group_vis=lslices[1-i0][0],  # 'X' <-> 'Y'
                    axes=kax,
                    ind=ii,
                )

                #
                axline = ax.axhline if ss == 'Y' else ax.axvline
                l0 = axline(
                    dat[ind[iind]],
                    c=color_dict[lslices[1-i0][0]][ii],  # 'X' <-> 'Y'
                )

                if ss == 'Y':
                    xydata = 'ydata'
                    km = f'{key}_lh-v{ii:02.0f}'
                else:
                    xydata = 'xdata'
                    km = f'{key}_lv-h{ii:02.0f}'
                coll.add_mobile(
                    key=km,
                    handle=l0,
                    refs=(dkeys[k0]['ref'],),
                    data=dkeys[k0]['key'],
                    dtype=xydata,
                    group_vis=ss,
                    axes=kax,
                    ind=ii,
                )

            if ss == 'Y':
                dax[kax].update(
                    refy=[dkeys[k0]['ref']],
                    datay=[dkeys[k0]['key']],
                )
            else:
                dax[kax].update(
                    refx=[dkeys[k0]['ref']],
                    datax=[dkeys[k0]['key']],
                )

    # -----------------
    # traces Z & U
    # -----------------

    for i0, ss in enumerate(['Z', 'U']):
        k0 = f"key{ss}"
        axtype = f'traces{ss}'
        lax = [k1 for k1, v1 in dax.items() if axtype in v1['type']]
        if len(lax) == 1:

            kax = lax[0]
            ax = dax[kax]['handle']
            dat = coll.ddata[dkeys[k0]['data']]['data']
            sli = dkeys[k0]['sli']
            iind = i0 + 2
            args = [ind[jj] for jj in range(ndim) if jj != iind]
            refs = tuple([
                dkeys[f"key{k1}"]['ref'] for k1 in ['X', 'Y', 'Z', 'U']
                if k1 != ss
            ])

            # individual time traces
            for ii in range(nmax):
                l1, = ax.plot(
                    dat,
                    data[sli(*args)],
                    ls='-',
                    marker='None',
                    color=color_dict[ss][ii],
                )

                km = f'{key}_trace{ss}{ii:02.0f}'
                coll.add_mobile(
                    key=km,
                    handle=l1,
                    refs=(refs,),
                    data=[key],
                    dtype=['ydata'],
                    group_vis=('X', 'Y'),  # 'X' <-> 'Y'
                    axes=kax,
                    ind=ii,
                )

            # vlines for single index selection
            l0 = ax.axvline(
                dat[ind[iind]],
                c='k',
            )
            km = f'{key}_lv_{ss}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(dkeys[k0]['ref'],),
                data=dkeys[k0]['key'],
                dtype='xdata',
                axes=kax,
                ind=0,
            )

            dax[kax].update(refx=[dkeys[k0]['ref']], datax=[dkeys[k0]['key']])

    # ---------
    # add text
    # ---------

    for ii, ss in enumerate(['X', 'Y', 'Z', 'U']):
        k0 = f"key{ss}"
        axtype = f'text{ss}'
        lax = [k1 for k1, v1 in dax.items() if axtype in v1['type']]
        if len(lax) == 1:
            kax = lax[0]
            ax = dax[kax]['handle']

            _plot_text.plot_text(
                coll=coll,
                kax=kax,
                key=key,
                ax=ax,
                ref=dkeys[k0]['ref'],
                group=ss,
                ind=ind[ii],
                lkeys=lkeys,
                nmax=nmax,
                color_dict=color_dict,
                bstr_dict=bstr_dict,
            )

    # -------------------
    # labeling and limits
    # -------------------

    if label:
        _label_axes(
            coll=coll,
            dax=dax,
            key=key,
            dkeys=dkeys,
            dvminmax=dvminmax,
            inverty=inverty,
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

    if fs is None:
        fs = (17, 9)

    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.95,
            'bottom': 0.06, 'top': 0.90,
            'hspace': 0.5, 'wspace': 0.4,
        }

    # ---------------
    # create
    # ---------------

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(ncols=7, nrows=6, **dmargin)

    # axes for image
    ax0 = fig.add_subplot(gs[:4, 2:4], aspect='auto')

    # axes for vertical profile
    ax1 = fig.add_subplot(gs[:4, 4], sharey=ax0)

    # axes for horizontal profile
    ax2 = fig.add_subplot(gs[4:, 2:4], sharex=ax0)

    # axes for tracesZ
    ax3 = fig.add_subplot(gs[:3, :2])

    # axes for tracesU
    ax4 = fig.add_subplot(gs[3:, :2])

    # axes for text
    ax5 = fig.add_subplot(gs[:3, 5], frameon=False)
    ax6 = fig.add_subplot(gs[3:, 5], frameon=False)
    ax7 = fig.add_subplot(gs[:3, 6], frameon=False)
    ax8 = fig.add_subplot(gs[3:, 6], frameon=False)

    # --------------
    # assemble dax
    # --------------

    # dax
    dax = {
        # data
        'matrix': {'handle': ax0},
        'vertical': {'handle': ax1},
        'horizontal': {'handle': ax2},
        'tracesZ': {'handle': ax3},
        'tracesU': {'handle': ax4},
        # text
        'textX': {'handle': ax5},
        'textY': {'handle': ax6},
        'textZ': {'handle': ax7},
        'textU': {'handle': ax8},
    }
    return dax


# #############################################################
# #############################################################
#                   Label axes
# #############################################################


def _label_axes(
    coll=None,
    dax=None,
    key=None,
    dkeys=None,
    dvminmax=None,
    inverty=None,
    rotation=None,
):

    # ------------
    # labels: fig
    # ------------

    fig = list(dax.values())[0]['handle'].figure
    fig.suptitle(key, size=14, fontweight='bold')

    # ---------------
    # labels: image
    # ---------------

    axtype = 'matrix'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax.xaxis.set_label_position('top')

        # x text ticks
        k0 = 'keyX'
        if dkeys[k0]['str'] is not False:
            ax.set_xticks(coll.ddata[dkeys[k0]['data']]['data'])
            ax.set_xticklabels(
                dkeys[k0]['str'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
        else:
            ax.set_xlabel(dkeys[k0]['lab'])

        # y text ticks
        k0 = 'keyY'
        if dkeys[k0]['str'] is not False:
            ax.set_yticks(coll.ddata[dkeys[k0]['data']]['data'])
            ax.set_yticklabels(
                dkeys[k0]['str'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
        else:
            ax.set_ylabel(dkeys[k0]['lab'])
        dax[kax]['inverty'] = inverty

    # --------------------------------
    # labels: horizontal and vertical
    # --------------------------------

    # axes for vertical profile
    axtype = 'vertical'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        ss = 'Y'
        k0 = f"key{ss}"
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xlabel('data')
        ax.set_ylabel(dkeys[k0]['lab'])
        ax.tick_params(
            axis="y",
            left=False, right=True,
            labelleft=False, labelright=True,
        )
        ax.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax.yaxis.set_label_position('right')
        ax.xaxis.set_label_position('top')

        if np.isfinite(dvminmax[ss]['min']):
            ax.set_ylim(bottom=dvminmax[ss]['min'])
        if np.isfinite(dvminmax[ss]['max']):
            ax.set_ylim(top=dvminmax[ss]['max'])

        if np.isfinite(dvminmax['data']['min']):
            ax.set_xlim(left=dvminmax['data']['min'])
        if np.isfinite(dvminmax['data']['max']):
            ax.set_xlim(right=dvminmax['data']['max'])

        # y text ticks
        if dkeys[k0]['str'] is not False:
            ax.set_yticks(coll.ddata[dkeys[k0]['data']]['data'])
            ax.set_yticklabels(
                dkeys[k0]['str'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
        dax[kax]['inverty'] = inverty

    # axes for horizontal profile
    axtype = 'horizontal'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        ss = 'X'
        k0 = f"key{ss}"
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_ylabel('data')
        ax.set_xlabel(dkeys[k0]['lab'])

        if np.isfinite(dvminmax[ss]['min']):
            ax.set_xlim(left=dvminmax[ss]['min'])
        if np.isfinite(dvminmax[ss]['max']):
            ax.set_xlim(right=dvminmax[ss]['max'])

        if np.isfinite(dvminmax['data']['min']):
            ax.set_ylim(bottom=dvminmax['data']['min'])
        if np.isfinite(dvminmax['data']['max']):
            ax.set_ylim(top=dvminmax['data']['max'])

        # x text ticks
        if dkeys[k0]['str'] is not False:
            ax.set_yticks(coll.ddata[dkeys[k0]['data']]['data'])
            ax.set_xticklabels(
                dkeys[k0]['str'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )

    # --------------
    # labels: traces
    # --------------

    for ss in ['Z', 'U']:
        axtype = f'traces{ss}'
        lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
        if len(lax) == 1:
            k0 = f"key{ss}"
            kax = lax[0]
            ax = dax[kax]['handle']
            ax.set_ylabel('data')
            ax.set_xlabel(dkeys[k0]['lab'])

            if np.isfinite(dvminmax[ss]['min']):
                ax.set_xlim(left=dvminmax[ss]['min'])
            if np.isfinite(dvminmax[ss]['max']):
                ax.set_xlim(right=dvminmax[ss]['max'])

            # z text ticks
            if dkeys[k0]['str'] is not False:
                ax.set_yticks(coll.ddata[dkeys[k0]['data']]['data'])
                ax.set_yticklabels(
                    dkeys[k0]['str'],
                    rotation=rotation,
                    horizontalalignment='right',
                    verticalalignment='top',
                )

    # -------------
    # labels: text
    # -------------

    for ss in ['X', 'Y', 'Z', 'U']:
        axtype = f'text{ss}'
        lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
        if len(lax) == 1:
            kax = lax[0]
            ax = dax[kax]['handle']
            ax.set_xticks([])
            ax.set_yticks([])

    return dax

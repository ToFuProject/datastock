# coding utf-8


# Common
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


# library-specific
from . import _generic_check
from . import _class1_compute
from . import _generic_utils_plot as _uplot
from . import _plot_text


# #############################################################
# #############################################################
#                       Main
# #############################################################


def main(
    # parameters
    coll=None,
    key=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    refX=None,
    refY=None,
    refZ=None,
    islogX=None,
    islogY=None,
    islogZ=None,
    ind=None,
    vmin=None,
    vmax=None,
    cmap=None,
    ymin=None,
    ymax=None,
    aspect=None,
    nmax=None,
    color_dict=None,
    lkeys=None,
    bstr_dict=None,
    rotation=None,
    inverty=None,
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
    refs = coll.ddata[key]['ref']
    if hasattr(data, 'nnz'):
        data = data.toarray()
    assert data.ndim == len(coll.ddata[key]['ref']) == 2
    n0, n1 = data.shape

    # check if transpose is necessary
    if refs.index(refX) == 0:
        dataplot = data.T
        nx, ny = n0, n1
        axisX, axisY = 0, 1
    else:
        dataplot = data
        nx, ny = n1, n0
        axisX, axisY = 1, 0

    # -----------------
    #  prepare slicing
    # -----------------

    # here slice X => slice in dim Y and vice-versa
    sliX = _class1_compute._get_slice(laxis=[1-axisX], ndim=2)
    sliY = _class1_compute._get_slice(laxis=[1-axisY], ndim=2)

    # ----------------------
    #  labels and data
    # ----------------------

    keyX, xstr, dataX, dX2, labX = _uplot._get_str_datadlab(
        keyX=keyX, nx=nx, islogX=islogX, coll=coll,
    )
    keyY, ystr, dataY, dY2, labY = _uplot._get_str_datadlab(
        keyX=keyY, nx=ny, islogX=islogY, coll=coll,
    )

    extent = (
        dataX[0] - dX2, dataX[-1] + dX2,
        dataY[0] - dY2, dataY[-1] + dY2,
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

    if label:
        _label_axes(
            coll=coll,
            dax=dax,
            key=key,
            labX=labX,
            labY=labY,
            ymin=ymin,
            ymax=ymax,
            xstr=xstr,
            ystr=ystr,
            keyX=keyX,
            keyY=keyY,
            dataX=dataX,
            dataY=dataY,
            inverty=inverty,
            rotation=rotation,
        )

    # ---------------
    # plot fixed part
    # ---------------

    axtype = 'matrix'
    kax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(kax) == 1:
        kax = kax[0]
        ax = dax[kax]['handle']

        ax.imshow(
            dataplot,
            extent=extent,
            interpolation=interp,
            origin='lower',
            aspect=aspect,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if inverty is True:
            ax.invert_yaxis()

    # ----------------
    # define and set dgroup
    # ----------------

    dgroup = {
        'X': {
            'ref': [refX],
            'data': ['index'],
            'nmax': nmax,
        },
        'Y': {
            'ref': [refY],
            'data': ['index'],
            'nmax': nmax,
        },
    }

    # ----------------
    # plot mobile part
    # ----------------

    axtype = 'matrix'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        # ind0, ind1
        axtype = 'vertical'
        lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
        if len(lax) == 1:
            for ii in range(nmax):
                lh = ax.axhline(
                    dataY[ind[1]], c=color_dict['X'][ii], lw=1., ls='-',
                )

                # update coll
                kh = f'{key}_h{ii:02.0f}'
                coll.add_mobile(
                    key=kh,
                    handle=lh,
                    refs=refY,
                    data=keyY,
                    dtype='ydata',
                    axes=kax,
                    ind=ii,
                )

            # for ax clic
            ax_refx = [refX]
            ax_datax = [keyX]
        else:
            # for ax clic
            ax_refx = None
            ax_datax = None

        # ind0
        axtype = 'horizontal'
        lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
        if len(lax) == 1:
            for ii in range(nmax):
                lv = ax.axvline(
                    dataX[ind[0]], c=color_dict['Y'][ii], lw=1., ls='-',
                )

                # update coll
                kv = f'{key}_v{ii:02.0f}'
                coll.add_mobile(
                    key=kv,
                    handle=lv,
                    refs=refX,
                    data=keyX,
                    dtype='xdata',
                    axes=kax,
                    ind=ii,
                )

            # for ax clic
            ax_refy = [refY]
            ax_datay = [keyY]
        else:
            # for ax clic
            ax_refy = None
            ax_datay = None

        dax[kax].update(
            refx=ax_refx,
            datax=ax_datax,
            refy=ax_refy,
            datay=ax_datay,
        )

    axtype = 'vertical'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l0, = ax.plot(
                data[sliY(ind[0])],
                dataY,
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['Y'][ii],
                label=f'ind0 = {ind[0]}',
            )

            km = f'{key}_vprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(refX,),
                data=key,
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

            l0 = ax.axhline(
                dataY[ind[1]],
                c=color_dict['X'][ii],
            )
            km = f'{key}_lh-v{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(refY,),
                data=keyY,
                dtype='ydata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refy=[refY], datay=[keyY])

    axtype = 'horizontal'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l1, = ax.plot(
                dataX,
                data[sliX(ind[1])],
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['X'][ii],
                label=f'ind1 = {ind[1]}',
            )

            km = f'{key}_hprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l1,
                refs=(refY,),
                data=[key],
                dtype='ydata',
                axes=kax,
                ind=ii,
            )

            l0 = ax.axvline(
                dataX[ind[0]],
                c=color_dict['Y'][ii],
            )
            km = f'{key}_lv-h{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(refX,),
                data=keyX,
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[refX], datax=[keyX])

    # ---------
    # add text
    # ---------

    axtype = 'text0'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            key=key,
            ax=ax,
            ref=refX,
            group='X',
            ind=ind[0],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    axtype = 'text1'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            key=key,
            ax=ax,
            ref=refY,
            group='Y',
            ind=ind[1],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
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
        fs = (14, 8)

    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.95,
            'bottom': 0.06, 'top': 0.90,
            'hspace': 0.2, 'wspace': 0.3,
        }

    # -----------------
    # create
    # ---------------

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(ncols=4, nrows=6, **dmargin)

    # axes for image
    ax0 = fig.add_subplot(gs[:4, :2], aspect='auto')

    # axes for vertical profile
    ax1 = fig.add_subplot(gs[:4, 2], sharey=ax0)

    # axes for horizontal profile
    ax2 = fig.add_subplot(gs[4:, :2], sharex=ax0)

    # axes for text
    ax3 = fig.add_subplot(gs[:3, 3], frameon=False)
    ax4 = fig.add_subplot(gs[3:, 3], frameon=False)

    # --------------
    # assemble dax
    # --------------

    dax = {
        # data
        'matrix': {'handle': ax0},
        'vertical': {'handle': ax1},
        'horizontal': {'handle': ax2},
        # text
        'text0': {'handle': ax3},
        'text1': {'handle': ax4},
    }

    return dax


# #############################################################
# #############################################################
#                   Label axes
# #############################################################


def _label_axes(
    coll=None,
    dax=None,
    tit=None,
    key=None,
    labX=None,
    labY=None,
    ymin=None,
    ymax=None,
    xstr=None,
    ystr=None,
    keyX=None,
    keyY=None,
    dataX=None,
    dataY=None,
    inverty=None,
    rotation=None,
):

    # ------
    # fig
    # ------

    fig = list(dax.values())[0]['handle'].figure
    fig.suptitle(tit, size=14, fontweight='bold')

    # ---------------
    # axes for image
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
        if xstr:
            ax.set_xticks(dataX)
            ax.set_xticklabels(
                coll.ddata[keyX]['data'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
        else:
            ax.set_xlabel(labX)

        # y text ticks
        if ystr:
            ax.set_yticks(dataY)
            ax.set_yticklabels(
                coll.ddata[keyY]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax.set_ylabel(labY)

        dax[kax]['inverty'] = inverty

    # --------------------------
    # axes for vertical profile
    # -------------------------

    axtype = 'vertical'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xlabel('data')
        ax.set_ylabel(labY)
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

        if np.isfinite(ymin):
            ax.set_xlim(left=ymin)
        if np.isfinite(ymax):
            ax.set_xlim(right=ymax)

        # y text ticks
        if ystr:
            ax.set_yticks(dataY)
            ax.set_yticklabels(
                coll.ddata[keyY]['data'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
        else:
            ax.set_ylabel(labY)

        dax[kax]['inverty'] = inverty

    # -----------------------------
    # axes for horizontal profile
    # -----------------------------

    axtype = 'horizontal'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_ylabel('data')
        ax.set_xlabel(labX)

        if np.isfinite(ymin):
            ax.set_ylim(bottom=ymin)
        if np.isfinite(ymax):
            ax.set_ylim(top=ymax)

        # x text ticks
        if xstr:
            ax.set_xticks(dataX)
            ax.set_xticklabels(
                coll.ddata[keyX]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax.set_xlabel(labX)

    # -----------------
    # axes for text
    # -----------------

    axtype = 'textX'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xticks([])
        ax.set_yticks([])

    axtype = 'textY'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xticks([])
        ax.set_yticks([])

    return dax

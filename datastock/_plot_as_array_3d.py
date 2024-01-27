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
    refs = coll.ddata[key]['ref']
    if hasattr(data, 'nnz'):
        data = data.toarray()
    assert data.ndim == len(coll.ddata[key]['ref']) == 3
    n0, n1, n2 = data.shape

    # check if transpose is necessary
    [axX, axY, axZ] = [refs.index(rr) for rr in [refX, refY, refZ]]
    [nx, ny, nz] = [data.shape[aa] for aa in [axX, axY, axZ]]

    # -----------------
    #  prepare slicing
    # -----------------

    # here slice X => slice in dim Y and vice-versa
    sliX = _class1_compute._get_slice(laxis=[axY, axZ], ndim=3)
    sliY = _class1_compute._get_slice(laxis=[axX, axZ], ndim=3)
    sliZ = _class1_compute._get_slice(laxis=[axX, axY], ndim=3)
    sliZ2 = _class1_compute._get_slice(laxis=[axZ], ndim=3)

    if axX < axY:
        datatype = 'data.T'
        dataplot = data[sliZ2(ind[2])].T
    else:
        datatype = 'data'
        dataplot = data[sliZ2(ind[2])]

    # ----------------------
    #  labels and data
    # ----------------------

    keyX, xstr, dataX, dX2, labX = _uplot._get_str_datadlab(
        keyX=keyX, nx=nx, islogX=islogX, coll=coll,
    )
    keyY, ystr, dataY, dY2, labY = _uplot._get_str_datadlab(
        keyX=keyY, nx=ny, islogX=islogY, coll=coll,
    )

    keyZ, zstr, dataZ, dZ2, labZ = _uplot._get_str_datadlab(
        keyX=keyZ, nx=nz, islogX=islogZ, coll=coll,
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
            labZ=labZ,
            ymin=ymin,
            ymax=ymax,
            xstr=xstr,
            ystr=ystr,
            zstr=zstr,
            keyX=keyX,
            keyY=keyY,
            keyZ=keyZ,
            dataX=dataX,
            dataY=dataY,
            dataZ=dataZ,
            inverty=inverty,
            rotation=rotation,
        )

    # ---------------
    # plot fixed part
    # ---------------

    axtype = 'tracesZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        if bck == 'lines':
            shap = list(data.shape)
            shap[axZ] = 1
            bckl = np.concatenate((data, np.full(shap, np.nan)), axis=axZ)
            bckl = np.swapaxes(bckl, axZ, -1).ravel()
            zdat = np.tile(np.r_[dataZ, np.nan], nx*ny)
            ax.plot(
                zdat,
                bckl,
                c=(0.8, 0.8, 0.8),
                ls='-',
                lw=1.,
                marker='None',
            )
        else:
            bckenv = [
                np.nanmin(data, axis=(axX, axY)),
                np.nanmax(data, axis=(axX, axY)),
            ]
            zdat = dataZ
            ax.fill_between(
                zdat,
                bckenv[0],
                bckenv[1],
                facecolor=(0.8, 0.8, 0.8, 0.8),
                edgecolor='None',
            )

    # ---------------------
    # define and set dgroup
    # ---------------------

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
        'Z': {
            'ref': [refZ],
            'data': ['index'],
            'nmax': 1,
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

        # image
        im = ax.imshow(
            dataplot,
            extent=extent,
            interpolation=interp,
            origin='lower',
            aspect=aspect,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        km = f'{key}_im'
        coll.add_mobile(
            key=km,
            handle=im,
            refs=refZ,
            data=key,
            dtype=datatype,
            axes=kax,
            ind=0,
            harmonize=False,
        )

        if inverty is True:
            ax.invert_yaxis()

        # ind0, ind1
        for ii in range(nmax):
            lh = ax.axhline(
                dataY[ind[1]], c=color_dict['X'][ii], lw=1., ls='-',
            )
            lv = ax.axvline(
                dataX[ind[0]], c=color_dict['Y'][ii], lw=1., ls='-',
            )
            mi, = ax.plot(
                dataX[ind[0]],
                dataY[ind[1]],
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
                refs=refY,
                data=keyY,
                dtype='ydata',
                axes=kax,
                ind=ii,
                harmonize=False,
            )
            coll.add_mobile(
                key=kv,
                handle=lv,
                refs=refX,
                data=keyX,
                dtype='xdata',
                axes=kax,
                ind=ii,
                harmonize=False,
            )
            km = f'{key}_m{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=mi,
                refs=[refX, refY],
                data=[keyX, keyY],
                dtype=['xdata', 'ydata'],
                axes=kax,
                ind=ii,
                harmonize=False,
            )

        dax[kax].update(
            refx=[refX],
            refy=[refY],
            datax=[keyX],
            datay=[keyY],
        )

    axtype = 'vertical'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l0, = ax.plot(
                data[sliY(ind[0], ind[2])],
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
                refs=((refX, refZ),),
                data=[key],
                dtype=['xdata'],
                group_vis='X',
                axes=kax,
                ind=ii,
                harmonize=False,
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
                group_vis='Y',
                axes=kax,
                ind=ii,
                harmonize=False,
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
                data[sliX(ind[1], ind[2])],
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['X'][ii],
            )

            km = f'{key}_hprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l1,
                refs=((refY, refZ),),
                data=[key],
                dtype=['ydata'],
                group_vis='Y',
                axes=kax,
                ind=ii,
                harmonize=False,
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
                group_vis='X',
                axes=kax,
                ind=ii,
                harmonize=False,
            )

        dax[kax].update(refx=[refX], datax=[keyX])

    # traces
    axtype = 'tracesZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l1, = ax.plot(
                dataZ,
                data[sliZ(ind[0], ind[1])],
                ls='-',
                marker='None',
                color=color_dict['X'][ii],
            )

            km = f'{key}_traceZ{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l1,
                refs=((refX, refY),),
                data=[key],
                dtype=['ydata'],
                axes=kax,
                ind=ii,
                harmonize=False,
            )

        l0 = ax.axvline(
            dataZ[ind[2]],
            c='k',
        )
        km = f'{key}_lv-z'
        coll.add_mobile(
            key=km,
            handle=l0,
            refs=(refZ,),
            data=keyZ,
            dtype='xdata',
            axes=kax,
            ind=0,
            harmonize=False,
        )

        dax[kax].update(refx=[refZ], datax=[keyZ])

    # ---------
    # add text
    # ---------

    axtype = 'textX'
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

    axtype = 'textY'
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

    axtype = 'textZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            key=key,
            ax=ax,
            ref=refZ,
            group='Z',
            ind=ind[2],
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
        fs = (15, 9)

    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.95,
            'bottom': 0.06, 'top': 0.90,
            'hspace': 0.2, 'wspace': 0.3,
        }

    # ---------------
    # create
    # ---------------

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(ncols=6, nrows=6, **dmargin)

    # axes for image
    ax0 = fig.add_subplot(gs[:4, 2:4], aspect='auto')

    # axes for vertical profile
    ax1 = fig.add_subplot(gs[:4, 4], sharey=ax0)

    # axes for horizontal profile
    ax2 = fig.add_subplot(gs[4:, 2:4], sharex=ax0)

    # axes for traces
    ax3 = fig.add_subplot(gs[:3, :2])

    # axes for text
    ax4 = fig.add_subplot(gs[:3, 5], frameon=False)
    ax5 = fig.add_subplot(gs[3:, 5], frameon=False)
    ax6 = fig.add_subplot(gs[4:, :2], frameon=False)

    # --------------
    # assemble dax
    # --------------

    dax = {
        # data
        'matrix': {'handle': ax0},
        'vertical': {'handle': ax1},
        'horizontal': {'handle': ax2},
        'tracesZ': {'handle': ax3},
        # text
        'textX': {'handle': ax4},
        'textY': {'handle': ax5},
        'textZ': {'handle': ax6},
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
    labZ=None,
    ymin=None,
    ymax=None,
    xstr=None,
    ystr=None,
    zstr=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    dataX=None,
    dataY=None,
    dataZ=None,
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

    # ------------------
    # axes for traces
    # ------------------

    axtype = 'tracesZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_ylabel('data')
        ax.set_xlabel(labZ)

        if np.isfinite(ymin):
            ax.set_ylim(bottom=ymin)
        if np.isfinite(ymax):
            ax.set_ylim(top=ymax)

        # z text ticks
        if zstr:
            ax.set_yticks(dataZ)
            ax.set_yticklabels(
                coll.ddata[keyZ]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax.set_ylabel(labZ)

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

    axtype = 'textZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xticks([])
        ax.set_yticks([])

    return dax

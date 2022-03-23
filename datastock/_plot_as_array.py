# coding utf-8


# Built-in
import itertools as itt
import warnings


# Common
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.transforms as transforms
import matplotlib.lines as mlines
import matplotlib.colors as mcolors


# library-specific
from . import _generic_check
from . import _plot_text
from . import _class2_interactivity


__all__ = ['plot_as_array']


__github = 'https://github.com/ToFuProject/datacollection/issues'
_WINTIT = f'report issues at {__github}'


_CONNECT = True
_BCKCOLOR = 'w'

_LCOLOR_DICT = [
    [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
    ],
    ['r', 'g', 'b'],
    ['m', 'y', 'c'],
]


# #############################################################################
# #############################################################################
#                       generic entry point
# #############################################################################


def plot_as_array(
    # parameters
    coll=None,
    key=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    ind=None,
    vmin=None,
    vmax=None,
    cmap=None,
    ymin=None,
    ymax=None,
    aspect=None,
    nmax=None,
    color_dict=None,
    dinc=None,
    lkeys=None,
    bstr_dict=None,
    rotation=None,
    inverty=None,
    bck=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    connect=None,
    inplace=None,
):


    # ------------
    #  check inputs

    # check key, inplace flag and extract sub-collection
    key, inplace, coll2 = _generic_check._check_inplace(
        coll=coll,
        keys=[key],
        inplace=inplace,
    )
    key = key[0]
    ndim = coll2._ddata[key]['data'].ndim

    # --------------
    # check input

    (
        key,
        keyX, refX,
        keyY, refY,
        keyZ, refZ,
        ind,
        cmap, vmin, vmax,
        ymin, ymax,
        aspect, nmax,
        color_dict,
        rotation,
        inverty,
        bck,
        dcolorbar, dleg, connect,
    ) = _plot_as_array_check(
        ndim=ndim,
        coll=coll2,
        key=key,
        keyX=keyX,
        keyY=keyY,
        keyZ=keyZ,
        ind=ind,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ymin=ymin,
        ymax=ymax,
        aspect=aspect,
        nmax=nmax,
        color_dict=color_dict,
        rotation=rotation,
        inverty=inverty,
        bck=bck,
        # figure
        dcolorbar=dcolorbar,
        dleg=dleg,
        connect=connect,
    )

    # -------------------------
    #  call appropriate routine

    if ndim == 1:
        coll2, dax, dgroup = plot_as_array_1d(
            # parameters
            coll=coll2,
            key=key,
            keyX=keyX,
            refX=refX,
            ind=ind,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ymin=ymin,
            ymax=ymax,
            aspect=aspect,
            nmax=nmax,
            color_dict=color_dict,
            lkeys=lkeys,
            bstr_dict=bstr_dict,
            rotation=rotation,
            # figure-specific
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
        )

    elif ndim == 2:
        coll2, dax, dgroup = plot_as_array_2d(
            # parameters
            coll=coll2,
            key=key,
            keyX=keyX,
            keyY=keyY,
            refX=refX,
            refY=refY,
            ind=ind,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ymin=ymin,
            ymax=ymax,
            aspect=aspect,
            nmax=nmax,
            color_dict=color_dict,
            lkeys=lkeys,
            bstr_dict=bstr_dict,
            rotation=rotation,
            inverty=inverty,
            # figure-specific
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
        )

    elif ndim == 3:
        coll, dax, dgroup = plot_as_array_3d(
            # parameters
            coll=coll2,
            key=key,
            keyX=keyX,
            keyY=keyY,
            keyZ=keyZ,
            refX=refX,
            refY=refY,
            refZ=refZ,
            ind=ind,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ymin=ymin,
            ymax=ymax,
            aspect=aspect,
            nmax=nmax,
            color_dict=color_dict,
            lkeys=lkeys,
            bstr_dict=bstr_dict,
            rotation=rotation,
            inverty=inverty,
            bck=bck,
            # figure-specific
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
        )

    # --------------------------
    # add axes and interactivity

    # add axes
    for ii, kax in enumerate(dax.keys()):
        harmonize = ii == len(dax.keys()) - 1
        coll2.add_axes(key=kax, harmonize=harmonize, **dax[kax])

    # connect
    if connect is True:
        coll2.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)
        coll2.disconnect_old()
        coll2.connect()

    return coll2

# #############################################################################
# #############################################################################
#                       check
# #############################################################################


def _check_keyXYZ(coll=None, refs=None, keyX=None, ndim=None, dimlim=None):
    """   """

    refX = None
    if ndim >= dimlim:
        if keyX is not None:
            if keyX in coll._ddata.keys():
                lok = [
                    k0 for k0, v0 in coll._ddata.items()
                    if len( v0['ref']) == 1
                    and v0['ref'][0] in refs
                    and (
                        v0['data'].dtype.type == np.str_
                        or (
                            v0['monot'] == (True,)
                            and np.allclose(
                                np.diff(v0['data']),
                                v0['data'][1] - v0['data'][0],
                                equal_nan=False,
                            )
                        )
                    )
                ]
                keyX = _generic_check._check_var(
                    keyX, 'keyX',
                    allowed=lok,
                )
                refX = coll._ddata[keyX]['ref'][0]

            elif keyX in refs:
                keyX, refX = 'index', keyX

            else:
                msg = f"Arg keyX refers to unknow data:\n\t- Provided: {keyX}"
                raise Exception(msg)
        else:
            keyX, refX = 'index', refs[dimlim - 1]

    return keyX, refX


def _plot_as_array_check(
    ndim=None,
    coll=None,
    key=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    ind=None,
    cmap=None,
    vmin=None,
    vmax=None,
    ymin=None,
    ymax=None,
    aspect=None,
    nmax=None,
    color_dict=None,
    rotation=None,
    inverty=None,
    bck=None,
    # figure
    dcolorbar=None,
    dleg=None,
    data=None,
    connect=None,
):


    # groups
    if ndim == 1:
        groups = ['X']
    elif ndim == 2:
        groups = ['X', 'Y']
    elif ndim == 3:
        groups = ['X', 'Y', 'Z']
    else:
        msg = "ndim must be in [1, 2, 3]"
        raise Exception(msg)

    # keyX, keyY, keyZ
    refs = coll._ddata[key]['ref']
    keyX, refX = _check_keyXYZ(
        coll=coll, refs=refs, keyX=keyX, ndim=ndim, dimlim=1,
    )
    keyY, refY = _check_keyXYZ(
        coll=coll, refs=refs, keyX=keyY, ndim=ndim, dimlim=2,
    )
    keyZ, refZ = _check_keyXYZ(
        coll=coll, refs=refs, keyX=keyZ, ndim=ndim, dimlim=3,
    )

    # unciitiy of refX vs refY
    if ndim == 2 and refX == refY:
        msg = (
            "Arg keyX and keyY have the same references!\n"
            f"\t- keyX, refX: {keyX}, {refX}\n"
            f"\t- keyY, refY: {keyY}, {refY}\n"
        )
        raise Exception(msg)

    if ndim == 3 and len(set([refX, refY, refZ])) < 3:
        msg = (
            "Arg keyX, keyY, keyZ have the same references!\n"
            f"\t- keyX, refX: {keyX}, {refX}\n"
            f"\t- keyY, refY: {keyY}, {refY}\n"
            f"\t- keyZ, refZ: {keyZ}, {refZ}\n"
        )
        raise Exception(msg)

    # ind
    ind = _generic_check._check_var(
        ind, 'ind',
        default=[0 for ii in range(ndim)],
        types=(list, tuple, np.ndarray),
    )
    c0 = (
        len(ind) == ndim
        and all([
            np.isscalar(ii) and isinstance(ii, (int, np.integer))
            for ii in ind
        ])
    )
    if not c0:
        msg = (
            "Arg ind must be an iterable of 2 integer indices!\n"
            f"Provided: {ind}"
        )
        raise Exception(msg)

    # cmap
    if cmap is None or vmin is None or vmax is None:
        if isinstance(coll.ddata[key]['data'], np.ndarray):
            nanmax = np.nanmax(coll.ddata[key]['data'])
            nanmin = np.nanmin(coll.ddata[key]['data'])
        else:
            nanmax = coll.ddata[key]['data'].max()
            nanmin = coll.ddata[key]['data'].min()
        diverging = nanmin * nanmax <= 0

    if cmap is None:
        if diverging:
            cmap = 'seismic'
        else:
            cmap = 'viridis'

    # vmin, vmax
    if vmin is None:
        if diverging:
            vmin = -max(abs(nanmin), nanmax)
        else:
            vmin = nanmin
        if vmax is None:
            if diverging:
                vmax = max(abs(nanmin), nanmax)
            else:
                vmax = nanmax

    # vmin, vmax
    if ymin is None:
        ymin = vmin
    if ymax is None:
        ymax = vmax

    # aspect
    aspect = _generic_check._check_var(
        aspect, 'aspect',
        default='equal',
        types=str,
        allowed=['auto', 'equal'],
    )

    # nmax
    nmax = _generic_check._check_var(
        nmax, 'nmax',
        default=3,
        types=int,
    )

    # color_dict
    cdef = {
        k0: _LCOLOR_DICT[0] for ii, k0 in enumerate(groups)
    }
    color_dict = _generic_check._check_var(
        color_dict, 'color_dict',
        default=cdef,
        types=dict,
    )
    dout = {
        k0: str(v0)
        for k0, v0 in color_dict.items()
        if not (
            isinstance(k0, str)
            and k0 in groups
            and isinstance(v0, list)
            and all([mcolors.is_color_like(v1) for v1 in v0])
        )
    }
    if len(dout) > 0:
        lstr = [f"{k0}: {v0}" for k0, v0 in dout.items()]
        msg = (
            "The following entries of color_dict are invalid"
        )

    # rotation
    rotation = _generic_check._check_var(
        rotation, 'rotation',
        default=45,
        types=(int, float),
    )

    # inverty
    inverty = _generic_check._check_var(
        inverty, 'inverty',
        default=keyY == 'index',
        types=bool,
    )

    # bck
    if coll.ddata[key]['data'].size > 10000:
        bckdef = 'envelop'
    else:
        bckdef = 'lines'
    bck = _generic_check._check_var(
        bck, 'bck',
        default=bckdef,
        allowed=['lines', 'envelop', False],
    )

    # dcolorbar
    defdcolorbar = {
        # 'location': 'right',
        'fraction': 0.15,
        'orientation': 'vertical',
    }
    dcolorbar = _generic_check._check_var(
        dcolorbar, 'dcolorbar',
        default=defdcolorbar,
        types=dict,
    )

    # dleg
    defdleg = {
        'bbox_to_anchor': (1.1, 1.),
        'loc': 'upper left',
        'frameon': True,
    }
    dleg = _generic_check._check_var(
        dleg, 'dleg',
        default=defdleg,
        types=(bool, dict),
    )

    # connect
    connect = _generic_check._check_var(
        connect, 'connect',
        default=_CONNECT,
        types=bool,
    )

    return (
        key,
        keyX, refX,
        keyY, refY,
        keyZ, refZ,
        ind,
        cmap, vmin, vmax,
        ymin, ymax,
        aspect, nmax,
        color_dict,
        rotation,
        inverty,
        bck,
        dcolorbar, dleg, connect,
    )


def _get_str_datadlab(keyX=None, nx=None, coll=None):

    xstr = keyX != 'index' and coll.ddata[keyX]['data'].dtype.type == np.str_
    if keyX == 'index':
        dataX = np.arange(0, nx)
        labX = keyX
        dX2 = 0.5
    elif xstr:
        dataX = np.arange(0, nx)
        labX = ''
        dX2 = 0.5
    else:
        dataX = coll.ddata[keyX]['data']
        dX2 = np.nanmean(np.diff(dataX))
        labX = f"{keyX} ({coll._ddata[keyX]['units']})"

    return xstr, dataX, dX2, labX


# #############################################################################
# #############################################################################
#                       plot_as_array: 1d
# #############################################################################


def plot_as_array_1d(
    # parameters
    coll=None,
    key=None,
    keyX=None,
    refX=None,
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
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
):

    # --------------
    #  Prepare data

    data = coll.ddata[key]['data']
    if hasattr(data, 'nnz'):
        data = data.toarray()
    assert data.ndim == len(coll.ddata[key]['ref']) == 1
    n0, = data.shape

    xstr, dataX, dX2, labX = _get_str_datadlab(keyX=keyX, nx=n0, coll=coll)
    ref = coll._ddata[key]['ref'][0]
    units = coll._ddata[key]['units']
    lab0 = f'ind ({ref})'
    lab1 = f'{key} ({units})'

    # --------------
    # plot - prepare

    if dax is None:

        if fs is None:
            fs = (12, 8)

        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.95,
                'bottom': 0.10, 'top': 0.90,
                'hspace': 0.15, 'wspace': 0.2,
            }

        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(ncols=4, nrows=1, **dmargin)

        ax0 = fig.add_subplot(gs[0, :3], aspect='auto')
        ax0.set_ylabel(lab1)
        ax0.set_title(key, size=14, fontweight='bold')
        if xstr:
            ax0.set_xticks(dataX)
            ax0.set_xticklabels(
                coll.ddata[keyX]['data'],
                rotation=rotation,
            )
        else:
            ax0.set_xlabel(lab0)

        ax1 = fig.add_subplot(gs[0, 3], frameon=False)
        ax1.set_xticks([])
        ax1.set_yticks([])

        dax = {
            'misc': {'handle': ax0, 'type': 'misc'},
            'text': {'handle': ax1, 'type': 'text'},
        }

    dax = _generic_check._check_dax(dax=dax, main='misc')

    # ---------------
    # plot fixed part

    axtype = 'misc'
    lkax = [kk for kk, vv in dax.items() if vv['type'] == axtype]
    for kax in lkax:
        ax = dax[kax]['handle']

        ax.plot(
            dataX,
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

    dgroup = {
        'X': {
            'ref': [ref],
            'data': ['index'],
            'nmax': nmax,
        },
    }

    # ----------------
    # plot mobile part

    axtype = 'misc'
    lkax = [kk for kk, vv in dax.items() if vv['type'] == axtype]
    for kax in lkax:
        ax = dax[kax]['handle']

        # ind0, ind1
        for ii in range(nmax):
            lv = ax.axvline(ind[0], c=color_dict['X'][ii], lw=1., ls='-')

            # update coll
            kv = f'v{ii:02.0f}'
            coll.add_mobile(
                key=kv,
                handle=lv,
                ref=ref,
                data='index',
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[ref])

    # ---------
    # add text

    kax = 'text'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            ax=ax,
            ref=ref,
            group='X',
            ind=ind[0],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    return coll, dax, dgroup


# #############################################################################
# #############################################################################
#                       plot_as_array: 2d
# #############################################################################


def plot_as_array_2d(
    # parameters
    coll=None,
    key=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    refX=None,
    refY=None,
    refZ=None,
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
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    interactive=None,
):

    # --------------
    #  Prepare data

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

    # here slice X => slice in dim Y and vice-versa
    sliX = _class2_interactivity._get_slice(laxis=[1-axisX], ndim=2)
    sliY = _class2_interactivity._get_slice(laxis=[1-axisY], ndim=2)

    # ----------------------
    #  labels and data

    xstr, dataX, dX2, labX = _get_str_datadlab(keyX=keyX, nx=nx, coll=coll)
    ystr, dataY, dY2, labY = _get_str_datadlab(keyX=keyY, nx=ny, coll=coll)

    extent = (
        dataX[0] - dX2, dataX[-1] + dX2,
        dataY[0] - dY2, dataY[-1] + dY2,
    )

    # --------------
    # plot - prepare

    if dax is None:

        if fs is None:
            fs = (14, 8)

        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.95,
                'bottom': 0.06, 'top': 0.90,
                'hspace': 0.2, 'wspace': 0.3,
            }

        fig = plt.figure(figsize=fs)
        fig.suptitle(key, size=14, fontweight='bold')
        gs = gridspec.GridSpec(ncols=4, nrows=6, **dmargin)

        # axes for image
        ax0 = fig.add_subplot(gs[:4, :2], aspect='auto')
        ax0.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax0.xaxis.set_label_position('top')

        # axes for vertical profile
        ax1 = fig.add_subplot(gs[:4, 2], sharey=ax0)
        ax1.set_xlabel('data')
        ax1.set_ylabel(labY)
        ax1.tick_params(
            axis="y",
            left=False, right=True,
            labelleft=False, labelright=True,
        )
        ax1.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax1.yaxis.set_label_position('right')
        ax1.xaxis.set_label_position('top')

        # axes for horizontal profile
        ax2 = fig.add_subplot(gs[4:, :2], sharex=ax0)
        ax2.set_ylabel('data')
        ax2.set_xlabel(labX)

        ax1.set_xlim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)

        # axes for text
        ax3 = fig.add_subplot(gs[:3, 3], frameon=False)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4 = fig.add_subplot(gs[3:, 3], frameon=False)
        ax4.set_xticks([])
        ax4.set_yticks([])

        if xstr:
            ax0.set_xticks(dataX)
            ax0.set_xticklabels(
                coll.ddata[keyX]['data'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
            ax2.set_xticks(dataX)
            ax2.set_xticklabels(
                coll.ddata[keyX]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax0.set_xlabel(labX)
            ax2.set_xlabel(labX)

        if ystr:
            ax0.set_yticks(dataY)
            ax0.set_yticklabels(
                coll.ddata[keyY]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
            ax1.set_yticks(dataY)
            ax1.set_yticklabels(
                coll.ddata[keyY]['data'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
        else:
            ax0.set_ylabel(labY)
            ax1.set_ylabel(labY)

        dax = {
            # data
            'matrix': {'handle': ax0, 'type': 'matrix', 'inverty': inverty},
            'vertical': {'handle': ax1, 'type': 'misc', 'inverty': inverty},
            'horizontal': {'handle': ax2, 'type': 'misc'},
            # text
            'text0': {'handle': ax3, 'type': 'text'},
            'text1': {'handle': ax4, 'type': 'text'},
        }

    dax = _generic_check._check_dax(dax=dax, main='matrix')

    # ---------------
    # plot fixed part

    axtype = 'matrix'
    lkax = [kk for kk, vv in dax.items() if vv['type'] == axtype]
    for kax in lkax:
        ax = dax[kax]['handle']

        im = ax.imshow(
            dataplot,
            extent=extent,
            interpolation='nearest',
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

    kax = 'matrix'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # ind0, ind1
        if dax.get('vertical') is not None:
            for ii in range(nmax):
                lh = ax.axhline(
                    dataY[ind[1]], c=color_dict['X'][ii], lw=1., ls='-',
                )

                # update coll
                kh = f'h{ii:02.0f}'
                coll.add_mobile(
                    key=kh,
                    handle=lh,
                    ref=refY,
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
        if dax.get('horizontal') is not None:
            for ii in range(nmax):
                lv = ax.axvline(
                    dataX[ind[0]], c=color_dict['Y'][ii], lw=1., ls='-',
                )

                # update coll
                kv = f'v{ii:02.0f}'
                coll.add_mobile(
                    key=kv,
                    handle=lv,
                    ref=refX,
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
            refx=ax_refx, datax=ax_datax, refy=ax_refy, datay=ax_datay,
        )

    kax = 'vertical'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l0, = ax.plot(
                data[sliY(ind[1])],
                dataY,
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['Y'][ii],
                label=f'ind0 = {ind[0]}',
            )

            km = f'vprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                ref=(refX,),
                data=key,
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

            l0 = ax.axhline(
                ind[1],
                c=color_dict['X'][ii],
            )
            km = f'lh-v{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                ref=(refY,),
                data=keyY,
                dtype='ydata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refy=[refY], datay=keyY)

    kax = 'horizontal'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l1, = ax.plot(
                dataX,
                data[sliX(ind[0])],
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['X'][ii],
                label=f'ind1 = {ind[1]}',
            )

            km = f'hprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l1,
                ref=(refY,),
                data=[key],
                dtype='ydata',
                axes=kax,
                ind=ii,
            )

            l0 = ax.axvline(
                ind[0],
                c=color_dict['Y'][ii],
            )
            km = f'lv-h{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                ref=(refX,),
                data=keyX,
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[refX], datax=keyX)

    # ---------
    # add text

    kax = 'text0'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            ax=ax,
            ref=refY,
            group='X',
            ind=ind[0],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    kax = 'text1'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            ax=ax,
            ref=refX,
            group='Y',
            ind=ind[1],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    return coll, dax, dgroup


# #############################################################################
# #############################################################################
#                       plot_as_array: 3d
# #############################################################################


def plot_as_array_3d(
    # parameters
    coll=None,
    key=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    refX=None,
    refY=None,
    refZ=None,
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
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
):

    # --------------
    #  Prepare data

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

    # here slice X => slice in dim Y and vice-versa
    sliX = _class2_interactivity._get_slice(laxis=[axY, axZ], ndim=3)
    sliY = _class2_interactivity._get_slice(laxis=[axX, axZ], ndim=3)
    sliZ = _class2_interactivity._get_slice(laxis=[axX, axY], ndim=3)
    sliZ2 = _class2_interactivity._get_slice(laxis=[axZ], ndim=3)

    if axX < axY:
        datatype = 'data.T'
        dataplot = data[sliZ2(ind[2])].T
    else:
        datatype = 'data'
        dataplot = data[sliZ2(ind[2])]

    # ----------------------
    #  labels and data

    xstr, dataX, dX2, labX = _get_str_datadlab(keyX=keyX, nx=nx, coll=coll)
    ystr, dataY, dY2, labY = _get_str_datadlab(keyX=keyY, nx=ny, coll=coll)
    zstr, dataZ, dZ2, labZ = _get_str_datadlab(keyX=keyZ, nx=nz, coll=coll)

    extent = (
        dataX[0] - dX2, dataX[-1] + dX2,
        dataY[0] - dY2, dataY[-1] + dY2,
    )

    # --------------
    # plot - prepare

    if dax is None:

        if fs is None:
            fs = (15, 9)

        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.95,
                'bottom': 0.06, 'top': 0.90,
                'hspace': 0.2, 'wspace': 0.3,
            }

        fig = plt.figure(figsize=fs)
        fig.suptitle(key, size=14, fontweight='bold')
        gs = gridspec.GridSpec(ncols=6, nrows=6, **dmargin)

        # axes for image
        ax0 = fig.add_subplot(gs[:4, 2:4], aspect='auto')
        ax0.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax0.xaxis.set_label_position('top')

        # axes for vertical profile
        ax1 = fig.add_subplot(gs[:4, 4], sharey=ax0)
        ax1.set_xlabel('data')
        ax1.set_ylabel(labY)
        ax1.tick_params(
            axis="y",
            left=False, right=True,
            labelleft=False, labelright=True,
        )
        ax1.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax1.yaxis.set_label_position('right')
        ax1.xaxis.set_label_position('top')

        # axes for horizontal profile
        ax2 = fig.add_subplot(gs[4:, 2:4], sharex=ax0)
        ax2.set_ylabel('data')
        ax2.set_xlabel(labX)

        ax1.set_xlim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)

        # axes for traces
        ax3 = fig.add_subplot(gs[:3, :2])
        ax3.set_ylabel('data')
        ax3.set_xlabel(labZ)

        ax1.set_xlim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)
        ax3.set_ylim(ymin, ymax)

        # axes for text
        ax4 = fig.add_subplot(gs[:3, 5], frameon=False)
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax5 = fig.add_subplot(gs[3:, 5], frameon=False)
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax6 = fig.add_subplot(gs[4:, :2], frameon=False)
        ax6.set_xticks([])
        ax6.set_yticks([])

        if xstr:
            ax0.set_xticks(dataX)
            ax0.set_xticklabels(
                coll.ddata[keyX]['data'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
            ax2.set_xticks(dataX)
            ax2.set_xticklabels(
                coll.ddata[keyX]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax0.set_xlabel(labX)
            ax2.set_xlabel(labX)

        if ystr:
            ax0.set_yticks(dataY)
            ax0.set_yticklabels(
                coll.ddata[keyY]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
            ax1.set_yticks(dataY)
            ax1.set_yticklabels(
                coll.ddata[keyY]['data'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
        else:
            ax0.set_ylabel(labY)
            ax1.set_ylabel(labY)

        if zstr:
            ax3.set_yticks(dataZ)
            ax3.set_yticklabels(
                coll.ddata[keyZ]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax3.set_ylabel(labZ)

        dax = {
            # data
            'matrix': {'handle': ax0, 'type': 'matrix', 'inverty': inverty},
            'vertical': {'handle': ax1, 'type': 'misc', 'inverty': inverty},
            'horizontal': {'handle': ax2, 'type': 'misc'},
            'traces': {'handle': ax3, 'type': 'misc'},
            # text
            'textX': {'handle': ax4, 'type': 'text'},
            'textY': {'handle': ax5, 'type': 'text'},
            'textZ': {'handle': ax6, 'type': 'text'},
        }

    dax = _generic_check._check_dax(dax=dax, main='matrix')

    # ---------------
    # plot fixed part


    kax = 'traces'
    if dax.get(kax) and bck is not False:
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
                np.nanmin(
                    np.nanmin(data, axis=max(axX, axY)), axis=min(axX, axY)
                ),
                np.nanmax(
                    np.nanmax(data, axis=max(axX, axY)), axis=min(axX, axY),
                )
            ]
            zdat = dataZ
            ax.fill_between(
                zdat,
                bckenv[0],
                bckenv[1],
                facecolor=(0.8, 0.8, 0.8, 0.8),
                edgecolor='None',
            )

    # ----------------
    # define and set dgroup

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

    kax = 'matrix'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # image
        im = ax.imshow(
            dataplot,
            extent=extent,
            interpolation='nearest',
            origin='lower',
            aspect=aspect,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        km = f'im'
        coll.add_mobile(
            key=km,
            handle=im,
            ref=refZ,
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
            kh = f'h{ii:02.0f}'
            kv = f'v{ii:02.0f}'
            coll.add_mobile(
                key=kh,
                handle=lh,
                ref=refY,
                data=keyY,
                dtype='ydata',
                axes=kax,
                ind=ii,
            )
            coll.add_mobile(
                key=kv,
                handle=lv,
                ref=refX,
                data=keyX,
                dtype='xdata',
                axes=kax,
                ind=ii,
            )
            km = f'm{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=mi,
                ref=[refX, refY],
                data=[keyX, keyY],
                dtype=['xdata', 'ydata'],
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[refX], refy=[refY], datax=keyX, datay=keyY)

    kax = 'vertical'
    if dax.get(kax) is not None:
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

            km = f'vprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                ref=(refX, refZ),
                data=[key, key],
                dtype=['xdata', 'xdata'],
                group_vis='X',
                axes=kax,
                ind=ii,
            )

            l0 = ax.axhline(
                dataY[ind[1]],
                c=color_dict['X'][ii],
            )
            km = f'lh-v{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                ref=(refY,),
                data=keyY,
                dtype='ydata',
                group_vis='Y',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refy=[refY], datay=keyY)

    kax = 'horizontal'
    if dax.get(kax) is not None:
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

            km = f'hprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l1,
                ref=(refY, refZ),
                data=[key, key],
                dtype=['ydata', 'ydata'],
                group_vis='Y',
                axes=kax,
                ind=ii,
            )

            l0 = ax.axvline(
                dataX[ind[0]],
                c=color_dict['Y'][ii],
            )
            km = f'lv-h{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                ref=(refX,),
                data=keyX,
                dtype='xdata',
                group_vis='X',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[refX], datax=keyX)

    kax = 'traces'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l1, = ax.plot(
                dataZ,
                data[sliZ(ind[0], ind[1])],
                ls='-',
                marker='None',
                color=color_dict['X'][ii],
            )

            km = f'trace{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l1,
                ref=(refX, refY),
                data=[key, key],
                dtype=['ydata', 'ydata'],
                axes=kax,
                ind=ii,
            )

        l0 = ax.axvline(
            dataZ[ind[2]],
            c='k',
        )
        km = f'lv-z'
        coll.add_mobile(
            key=km,
            handle=l0,
            ref=(refZ,),
            data=keyZ,
            dtype='xdata',
            axes=kax,
            ind=0,
        )

        dax[kax].update(refx=[refZ])

    # ---------
    # add text

    kax = 'textX'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            ax=ax,
            ref=refX,
            group='X',
            ind=ind[0],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    kax = 'textY'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            ax=ax,
            ref=refY,
            group='Y',
            ind=ind[1],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    kax = 'textZ'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
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

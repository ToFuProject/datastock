# coding utf-8


# Common
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors


# library-specific
from . import _generic_check
from . import _plot_text
from . import _class1_compute
from ._plot_as_array import _check_keyXYZ
from ._generic_utils_plot import _get_str_datadlab


__all__ = ['plot_as_profile1d']


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


def plot_as_profile1d(
    # parameters
    coll=None,
    key=None,
    key_time=None,
    keyX=None,
    dkeys=None,
    ind=None,
    dscale=None,
    dvminmax=None,
    vmin=None,
    vmax=None,
    ymin=None,
    ymax=None,
    cmap=None,
    aspect=None,
    nmax=None,
    color_dict=None,
    dinc=None,
    lkeys=None,
    bstr_dict=None,
    rotation=None,
    inverty=None,
    bck=None,
    show_commands=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    connect=None,
    inplace=None,
    # unused
    **kwdargs,
):


    # ------------
    #  check inputs

    key = _generic_check._check_var(
        key, 'key',
        types=str,
        allowed=list(coll.ddata.keys()),
    )

    # check key, inplace flag and extract sub-collection
    lk = [kk for kk in [key, key_time, keyX] if kk is not None]
    coll2, key = coll.extract(
        lk,
        inc_monot=False,
        inc_vectors=False,
        inc_allrefs=False,
        return_keys=True,
    )
    key = [kk for kk in key if kk not in [key_time, keyX]][0]
    ndim = coll2._ddata[key]['data'].ndim

    # --------------
    # check input

    (
        key,
        key_time, ref_time, islogtime,
        keyX, refX, refX0,
        ind,
        cmap, vmin, vmax,
        ymin, ymax,
        aspect, nmax,
        color_dict,
        rotation,
        inverty,
        bck,
        dcolorbar, dleg, connect,
    ) = _plot_as_profile1d_check(
        ndim=ndim,
        coll=coll2,
        key=key,
        key_time=key_time,
        keyX=keyX,
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

    # --------------------------
    # call plotting routine

    coll2, dax, dgroup = _plot_as_profile1d(
        # parameters
        coll=coll2,
        key=key,
        key_time=key_time,
        keyX=keyX,
        ref_time=ref_time,
        refX=refX,
        refX0=refX0,
        islogtime=islogtime,
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

        coll2.show_commands(verb=show_commands)
        return coll2
    else:
        return coll2, dgroup


# #############################################################################
# #############################################################################
#                       check
# #############################################################################


def _check_keyX(coll=None, refs=None, ref_time=None, keyX=None):

    # keyX
    if keyX in coll.ddata.keys():
        lkok = [
            k0 for k0, v0 in coll.ddata.items()
            if tuple([kk for kk in refs if kk in v0['ref']]) == v0['ref']
            and len(v0['ref']) in [1, 2]
        ]
        keyX = _generic_check._check_var(
            keyX, 'keyX',
            allowed=lkok,
        )

        # refX, refX0
        refX = coll.ddata[keyX]['ref']
        if refX == refs:
            refX0 = refs[1 - refs.index(ref_time)]
        elif len(refX) == 1 and refX[0] in refs:
            refX0 = refX[0]
        else:
            msg = (
                f"Arg keyX {keyX} must be a data with:\n"
                f"\t- ref = {refs}\n"
                f"\t- or ref = {refs[1 - refs.index(ref_time)]}\n"
                f"Provided: {keyX} with ref = {refX}"
            )
            raise Exception(msg)

    elif keyX in refs:
        assert keyX != ref_time, keyX
        keyX, refX = 'index', keyX
        refX0 = refX

    else:
        msg = f"Unrecongnized keyX: {keyX}"
        raise Exception(msg)

    # final check
    if ref_time == refX:
        msg = (
            "Arg key_time and keyX have the same references!\n"
            f"\t- ref_time: {ref_time}\n"
            f"\t- keyX, refX: {keyX}, {refX}\n"
        )
        raise Exception(msg)

    return keyX, refX, refX0


def _plot_as_profile1d_check(
    ndim=None,
    coll=None,
    key=None,
    key_time=None,
    keyX=None,
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
    if ndim == 2:
        groups = ['time', 'X']
    else:
        msg = f"ndim must be in [2]\n\t- Provided: {ndim}"
        raise Exception(msg)

    # key_time, keyX
    refs = coll._ddata[key]['ref']
    key_time, ref_time, islogtime = _check_keyXYZ(
        coll=coll, refs=refs, keyX=key_time, ndim=ndim, dim_min=1,
        uniform=False,
    )
    keyX, refX, refX0 = _check_keyX(
        coll=coll, refs=refs, keyX=keyX, ref_time=ref_time,
    )

    ndimt = len(coll.ddata[key_time]['ref'])
    if key_time != 'index' and ndimt != 1:
        msg = (
            "Arg key_time must refer to a 1d data vector!\n"
            f"\t- Provided: {key_time} with dim = {ndimt}"
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
            if isinstance(nanmin, np.bool_):
                vmin = 0
            else:
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

    # bck
    if coll.ddata[key]['data'].size > 10000:
        bckdef = 'envelop'
    else:
        bckdef = 'lines'
    if bck is True:
        bck = bckdef

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
        key_time, ref_time, islogtime,
        keyX, refX, refX0,
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


def _get_bck(
    bck=None,
    y=None,
    x=None,
    axisx=None,
):
    nrep = y.shape[1-axisx]
    if bck == 'lines':

        sh = (1, nrep) if axisx == 0 else (nrep, 1)
        if x.ndim == 1:
            bckx = np.tile(np.append(x, [np.nan]), nrep)
        else:
            assert x.shape == y.shape
            if axisx == 0:
                bckx = np.append(x, np.zeros(sh), axis=0).T.ravel()
            else:
                bckx = np.append(y, np.zeros(sh), axis=1).ravel()
        if axisx == 0:
            bcky = np.append(y, np.zeros(sh), axis=0).T.ravel()
        else:
            bcky = np.append(y, np.zeros(sh), axis=1).ravel()
    elif bck == 'envelop' and x.ndim == 1:
        bckx = x
        bcky = [
            np.nanmin(y, axis=1-axisx),
            np.nanmax(y, axis=1-axisx),
        ]
    else:
        bckx, bcky = None, None

    return bckx, bcky


def _get_sliceXt(laxis=None, ndim=None):

    nax = len(laxis)
    assert nax in range(1, ndim + 1)

    if ndim == 1:
        def fslice(*args):
            return slice(None)

    else:
        def fslice(*args, laxis=laxis):
            ind = [slice(None) for ii in range(ndim)]
            for ii, aa in enumerate(args):
                ind[laxis[ii]] = aa
            return tuple(ind)

    return fslice


# #############################################################################
# #############################################################################
#                       plot_as_profile1d
# #############################################################################


def _plot_as_profile1d(
    # parameters
    coll=None,
    key=None,
    key_time=None,
    keyX=None,
    ref_time=None,
    refX=None,
    refX0=None,
    islogtime=None,
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
    bck=None,
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
    if refs.index(ref_time) == 0:
        nt, nx = n0, n1
        axist, axisX = 0, 1
        dataplot = data.ravel()
    else:
        nt, nx = n1, n0
        axist, axisX = 1, 0
        dataplot = data.ravel()

    # ----------------------
    #  labels and data

    key_time, tstr, dt2, labt = _get_str_datadlab(
        keyX=key_time, nx=nt, islogX=islogtime, coll=coll,
    )
    datat = coll.ddata[key_time]['data']

    # keyX can be 2d !!!
    keyX, xstr, _, labX = _get_str_datadlab(
        keyX=keyX, nx=nx, islogX=None, coll=coll,
    )
    dataX = coll.ddata[keyX]['data']

    # -----------------
    #  prepare slicing

    # here slice X => slice in dim Y and vice-versa
    slit = _class1_compute._get_slice(laxis=[1-axist], ndim=2)
    sliX = _class1_compute._get_slice(laxis=[1-axisX], ndim=2)
    sliXt = _get_sliceXt(laxis=[axist], ndim=dataX.ndim)

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
        gs = gridspec.GridSpec(ncols=5, nrows=2, **dmargin)

        # axes for scatter
        ax0 = fig.add_subplot(gs[0, :2], aspect='auto')

        # axes for time traces
        ax1 = fig.add_subplot(gs[1, :2], sharex=ax0)
        ax1.set_ylabel('data')
        ax1.set_xlabel(labt)
        ax1.set_ylim(ymin, ymax)

        # axes for profiles
        ax2 = fig.add_subplot(gs[:, 2:-1], sharey=ax1)
        ax2.set_ylabel('data')
        ax2.set_xlabel(labX)

        # axes for text
        ax3 = fig.add_subplot(gs[0, -1], frameon=False)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4 = fig.add_subplot(gs[1, -1], frameon=False)
        ax4.set_xticks([])
        ax4.set_yticks([])

        if xstr:
            ax0.set_xticks(datat)
            ax0.set_xticklabels(
                coll.ddata[key_time]['data'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
            ax1.set_xticks(datat)
            ax1.set_xticklabels(
                coll.ddata[key_time]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax0.set_xlabel(labt)
            ax1.set_xlabel(labt)

        dax = {
            # data
            'scatter': {'handle': ax0},
            'traces': {'handle': ax1},
            'profiles': {'handle': ax2},
            # text
            'text0': {'handle': ax3, 'type': 'text'},
            'text1': {'handle': ax4, 'type': 'text'},
        }

    dax = _generic_check._check_dax(dax=dax, main='scatter')

    # ---------------
    # plot fixed part

    kax = 'scatter'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        if axist == 0:
            datatscat = np.repeat(datat, nx).ravel()
            if dataX.ndim == 1:
                dataXscat = np.tile(dataX, nt).ravel()
        else:
            datatscat = np.tile(datat, nx).ravel()
            if dataX.ndim == 1:
                dataXscat = np.repeat(dataX, nt).ravel()
        if dataX.ndim == 2:
            dataXscat = dataX.ravel()

        im = ax.scatter(
            datatscat,
            dataXscat,
            s=6,
            c=data.ravel(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker='s',
            edgecolors='None',
        )

    kax = 'traces'
    if dax.get(kax) is not None and bck is not False:
        ax = dax[kax]['handle']

        bckt, bckdata = _get_bck(
            y=data,
            x=datat,
            bck=bck,
            axisx=axist,
        )
        if bck == 'lines':
            ax.plot(
                bckt,
                bckdata,
                c=(0.8, 0.8, 0.8, 0.8),
            )
        else:
            ax.fill_between(
                bckt,
                bckdata[0],
                bckdata[1],
                facecolor=(0.8, 0.8, 0.8, 0.8),
                edgecolor='None',
            )

    kax = 'profiles'
    if dax.get(kax) is not None and bck is not False:
        ax = dax[kax]['handle']

        bckt, bckdata = _get_bck(
            y=data,
            x=dataX,
            bck=bck,
            axisx=axisX,
        )
        if bck == 'lines':
            ax.plot(
                bckt,
                bckdata,
                c=(0.8, 0.8, 0.8, 0.8),
            )
        elif dataX.ndim == 1:
            ax.fill_between(
                bckt,
                bckdata[0],
                bckdata[1],
                facecolor=(0.8, 0.8, 0.8, 0.8),
                edgecolor='None',
            )

    # ----------------
    # define and set dgroup

    dgroup = {
        'time': {
            'ref': [ref_time],
            'data': ['index'],
            'nmax': nmax,
        },
        'X': {
            'ref': [refX0],
            'data': ['index'],
            'nmax': nmax,
        },
    }

    # ----------------
    # plot mobile part

    kax = 'scatter'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # ind0, ind1
        if dax.get('traces') is not None:
            for ii in range(nmax):
                lh = ax.axvline(
                    datat[ind[0]], c=color_dict['time'][ii], lw=1., ls='-',
                )

                # update coll
                kh = f't{ii:02.0f}'
                coll.add_mobile(
                    key=kh,
                    handle=lh,
                    refs=ref_time,
                    data=key_time,
                    dtype='xdata',
                    axes=kax,
                    ind=ii,
                )

            # for ax clic
            ax_refx = [ref_time]
            ax_datax = [key_time]
        else:
            # for ax clic
            ax_refx = None
            ax_datax = None

        # ind0
        # if dax.get('profiles') is not None:
            # for ii in range(nmax):
                # lv = ax.axhline(
                    # dataX[ind[0]], c=color_dict['Y'][ii], lw=1., ls='-',
                # )

                # # update coll
                # kv = f'v{ii:02.0f}'
                # coll.add_mobile(
                    # key=kv,
                    # handle=lv,
                    # refs=refX,
                    # data=keyX,
                    # dtype='xdata',
                    # axes=kax,
                    # ind=ii,
                # )

            # # for ax clic
            # ax_refy = [refY]
            # ax_datay = [keyY]
        # else:
            # # for ax clic
            # ax_refy = None
            # ax_datay = None

        dax[kax].update(
            refx=ax_refx, datax=ax_datax, # refy=ax_refy, datay=ax_datay,
        )

    kax = 'profiles'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l0, = ax.plot(
                dataX[sliXt(ind[0])],
                data[sliX(ind[0])],
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['time'][ii],
                label=f'ind0 = {ind[0]}',
            )

            km = f'prof{ii:02.0f}'
            if dataX.ndim == 1:
                coll.add_mobile(
                    key=km,
                    handle=l0,
                    refs=(ref_time,),
                    data=key,
                    dtype='ydata',
                    axes=kax,
                    ind=ii,
                )
            else:
                coll.add_mobile(
                    key=km,
                    handle=l0,
                    refs=(ref_time, ref_time),
                    data=(keyX, key),
                    dtype=['xdata', 'ydata'],
                    group_vis='time',
                    axes=kax,
                    ind=ii,
                )

            if dataX.ndim == 1:
                dtemp = dataX[ind[0]]
            else:
                dtemp = dataX[ind[0], ind[1]]
            l0 = ax.axvline(
                dtemp,
                c=color_dict['X'][ii],
            )
            km = f'lv-v{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(refX,),
                data=[keyX],
                dtype=['xdata'],
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[refX0], datax=keyX)

    kax = 'traces'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l1, = ax.plot(
                datat,
                data[slit(ind[1])],
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['X'][ii],
                label=f'ind1 = {ind[1]}',
            )

            km = f'trace{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l1,
                refs=(refX0,),
                data=[key],
                dtype='ydata',
                axes=kax,
                ind=ii,
            )

            l0 = ax.axvline(
                datat[ind[0]],
                c=color_dict['time'][ii],
            )
            km = f'lv-h{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(ref_time,),
                data=key_time,
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[ref_time], datax=key_time)

    # ---------
    # add text

    kax = 'text0'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            ax=ax,
            ref=ref_time,
            group='time',
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
            ref=refX0,
            group='X',
            ind=ind[1],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    return coll, dax, dgroup
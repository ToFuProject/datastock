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


__all__ = ['plot_as_mobile_lines']


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


def plot_as_mobile_lines(
    # parameters
    coll=None,
    keyX=None,
    keyY=None,
    key_time=None,
    key_chan=None,
    bck_color=None,
    bck=None,
    aspect=None,
    ind=None,
    nmax=None,
    color_dict=None,
    dinc=None,
    lkeys=None,
    bstr_dict=None,
    rotation=None,
    show_commands=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dleg=None,
    connect=None,
    inplace=None,
):

    # ------------
    #  check inputs

    # check key, inplace flag and extract sub-collection
    lk = [kk for kk in [keyX, keyY, key_time, key_chan] if kk is not None]
    coll2, key = coll.extract(
        lk,
        inc_monot=False,
        inc_vectors=False,
        inc_allrefs=False,
        return_keys=True,
    )
    keyX = [kk for kk in key if kk not in [keyY, key_time, key_chan]][0]
    keyY = [kk for kk in key if kk not in [keyX, key_time, key_chan]][0]
    ndim = coll2._ddata[keyX]['data'].ndim

    # --------------
    # check input

    (
        keyX, keyY, refs,
        keyt, reft, islogt,
        keych, refch, islogch,
        bck, bck_color, ind,
        aspect, nmax,
        color_dict,
        rotation,
        connect,
    ) = _plot_as_mobile_lines_check(
        ndim=ndim,
        coll=coll2,
        keyX=keyX,
        keyY=keyY,
        key_time=key_time,
        key_chan=key_chan,
        bck=bck,
        bck_color=bck_color,
        ind=ind,
        aspect=aspect,
        nmax=nmax,
        color_dict=color_dict,
        rotation=rotation,
        # figure
        dleg=dleg,
        connect=connect,
    )

    # --------------------------
    # call plotting routine

    if ndim == 2:
        coll2, dax, dgroup = _plot_as_mobile_lines2d(
            # parameters
            coll=coll2,
            keyX=keyX,
            keyY=keyY,
            keych=keych,
            refs=refs,
            refch=refch,
            islogch=islogch,
            # parameters
            bck=bck,
            bck_color=bck_color,
            ind=ind,
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
        )
    else:
        coll2, dax, dgroup = _plot_as_mobile_lines3d(
            # parameters
            coll=coll2,
            keyX=keyX,
            keyY=keyY,
            keyt=keyt,
            keych=keych,
            refs=refs,
            reft=reft,
            refch=refch,
            islogt=islogt,
            islogch=islogch,
            # parameters
            bck=bck,
            bck_color=bck_color,
            ind=ind,
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


def _check_notchar(key=None, keyname=None, coll=None):
    if key != 'index' and coll.ddata[key]['data'].dtype == np.str_:
        msg = (
            f"Arg {keyname} must refer to a float/int/bool data (not char)\n"
            f"\t- {keyname}: {key}\n"
        )
        raise Exception(msg)


def _plot_as_mobile_lines_check(
    ndim=None,
    coll=None,
    keyX=None,
    keyY=None,
    key_time=None,
    key_chan=None,
    bck=None,
    bck_color=None,
    ind=None,
    aspect=None,
    nmax=None,
    color_dict=None,
    rotation=None,
    # figure
    dleg=None,
    connect=None,
):

    # groups
    if ndim == 2:
        groups = ['chan']
    elif ndim == 3:
        groups = ['time', 'chan']
    else:
        msg = f"ndim must be in [2, 3]\n\t- Provided: {ndim}"
        raise Exception(msg)

    # keyX vs keyY
    refs = coll._ddata[keyX]['ref']
    if refs != coll.ddata[keyY]['ref']:
        msg = (
            "Arg keyX and keyY must refer to data of same ref!\n"
            f"\t- keyX ref: {coll.ddata[keyX]['ref']}\n"
            f"\t- keyY ref: {coll.ddata[keyY]['ref']}\n"
        )
        raise Exception(msg)

    _check_notchar(key=keyX, keyname='keyX', coll=coll)
    _check_notchar(key=keyY, keyname='keyY', coll=coll)

    # keyt, keych
    keyt, reft, islogt = _check_keyXYZ(
        coll=coll, refs=refs, keyX=key_time, ndim=ndim, dim_min=1,
        uniform=False,
    )
    keych, refch, islogch = _check_keyXYZ(
        coll=coll, refs=refs, keyX=key_chan, ndim=ndim, dim_min=2,
        uniform=False,
    )

    if keyt is not None:
        _check_notchar(key=keyt, keyname='keyt', coll=coll)
    if keych is not None:
        _check_notchar(key=keych, keyname='keych', coll=coll)

    # bck
    bck = _generic_check._check_var(
        bck, 'bck',
        default=True,
        types=bool,
    )

    # bck_color
    if bck_color is None:
        bck_color = (0.8, 0.8, 0.8, 0.8)
    if not mcolors.is_color_like(bck_color):
        msg = (
            "Arg bck_color must be a matplotlib color-like!\n"
            f"Provided: {bck_color}"
        )
        raise Exception(msg)

    # ind
    ind = _generic_check._check_var(
        ind, 'ind',
        default=[0 for ii in range(ndim-1)],
        types=(list, tuple, np.ndarray),
    )
    c0 = (
        len(ind) == ndim-1
        and all([
            np.isscalar(ii) and isinstance(ii, (int, np.integer))
            for ii in ind
        ])
    )
    if not c0:
        msg = (
            "Arg ind must be an iterable of integer indices!\n"
            f"Provided: {ind}"
        )
        raise Exception(msg)

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

    # connect
    connect = _generic_check._check_var(
        connect, 'connect',
        default=_CONNECT,
        types=bool,
    )

    return (
        keyX, keyY, refs,
        keyt, reft, islogt,
        keych, refch, islogch,
        bck, bck_color, ind,
        aspect, nmax,
        color_dict,
        rotation,
        connect,
    )


# #############################################################################
# #############################################################################
#                       plot_as_mobile_lines2d
# #############################################################################


def _plot_as_mobile_lines2d(
    # parameters
    coll=None,
    keyX=None,
    keyY=None,
    keyt=None,
    keych=None,
    refs=None,
    reft=None,
    refch=None,
    islogt=None,
    islogch=None,
    # parameters
    bck=None,
    bck_color=None,
    ind=None,
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
    dleg=None,
    interactive=None,
):

    # --------------
    #  Prepare data

    dataX = coll.ddata[keyX]['data']
    dataY = coll.ddata[keyY]['data']
    if hasattr(dataX, 'nnz'):
        dataX = dataX.toarray()
    if hasattr(dataY, 'nnz'):
        dataY = dataY.toarray()
    assert dataX.ndim == len(coll.ddata[keyX]['ref']) == 2

    axisch = refs.index(refch)
    nch = dataX.shape[axisch]

    # -----------
    # background

    if bck is True:
        bckx = np.insert(dataX, dataX.shape[1-axisch], np.nan, axis=1-axisch)
        bcky = np.insert(dataY, dataY.shape[1-axisch], np.nan, axis=1-axisch)
        if axisch == 0:
            bckx = bckx.ravel()
            bcky = bcky.ravel()
        else:
            bckx = bckx.T.ravel()
            bcky = bcky.T.ravel()

    # ----------------------
    #  labels and data

    labx = f"{keyX} ({coll.ddata[keyX]['units']})"
    laby = f"{keyY} ({coll.ddata[keyY]['units']})"

    keych, chstr, dch2, labch = _get_str_datadlab(
        keyX=keych, nx=nch, islogX=islogch, coll=coll,
    )
    datach = coll.ddata[keych]['data']

    # -----------------
    #  prepare slicing

    # here slice X and Y alike => slice in dim Y and vice-versa
    sli = _class1_compute._get_slice(laxis=[axisch], ndim=2)

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
        fig.suptitle(f'{keyX} vs {keyY}', size=14, fontweight='bold')
        gs = gridspec.GridSpec(ncols=5, nrows=2, **dmargin)

        # ax1 = chan
        ax1 = fig.add_subplot(gs[1, :2])
        ax1.set_ylabel(labch)
        ax1.set_xlabel('index')

        # ax2 = lines
        ax2 = fig.add_subplot(gs[:, 2:-1], aspect=aspect)
        ax2.set_ylabel(laby)
        ax2.set_xlabel(labx)
        if bck is False:
            ax2.set_xlim(np.nanmin(dataX), np.nanmax(dataX))
            ax2.set_ylim(np.nanmin(dataY), np.nanmax(dataY))

        # axes for text
        ax4 = fig.add_subplot(gs[1, -1], frameon=False)
        ax4.set_xticks([])
        ax4.set_yticks([])

        dax = {
            # data
            'chan': {'handle': ax1},
            'lines': {'handle': ax2},
            # text
            'text1': {'handle': ax4, 'type': 'text'},
        }

    dax = _generic_check._check_dax(dax=dax, main='lines')

    # ---------------
    # plot fixed part

    kax = 'chan'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        im = ax.plot(
            range(nch),
            datach,
            c='k',
            ls='-',
            lw=1.,
        )

    kax = 'lines'
    if dax.get(kax) is not None and bck is True:
        ax = dax[kax]['handle']
        ax.plot(
            bckx,
            bcky,
            c=bck_color,
            lw=1.,
            ls='-',
        )

    # ----------------
    # define and set dgroup

    dgroup = {
        'chan': {
            'ref': [refch],
            'data': ['index'],
            'nmax': nmax,
        },
    }

    # ----------------
    # plot mobile part

    kax = 'lines'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l0, = ax.plot(
                dataX[sli(ind[0])],
                dataY[sli(ind[0])],
                c=color_dict['chan'][ii],
                lw=1.,
                ls='-',
            )

            k0 = f'l{ii}'
            coll.add_mobile(
                key=k0,
                handle=l0,
                refs=((refch,), (refch,)),
                data=(keyX, keyY),
                dtype=['xdata', 'ydata'],
                axes=kax,
                ind=ii,
            )

    kax = 'chan'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l0 = ax.axvline(
                datach[0],
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['chan'][ii],
            )

            km = f'ch{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(refch,),
                data='index',
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[refch], datax='index')

    # ---------
    # add text

    kax = 'text1'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            ax=ax,
            ref=refch,
            group='chan',
            ind=ind[0],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    return coll, dax, dgroup


# #############################################################################
# #############################################################################
#                       plot_as_mobile_lines3d
# #############################################################################


def _plot_as_mobile_lines3d(
    # parameters
    coll=None,
    keyX=None,
    keyY=None,
    keyt=None,
    keych=None,
    refs=None,
    reft=None,
    refch=None,
    islogt=None,
    islogch=None,
    # parameters
    bck=None,
    bck_color=None,
    ind=None,
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
    dleg=None,
    interactive=None,
):

    # --------------
    #  Prepare data

    dataX = coll.ddata[keyX]['data']
    dataY = coll.ddata[keyY]['data']
    if hasattr(dataX, 'nnz'):
        dataX = dataX.toarray()
    if hasattr(dataY, 'nnz'):
        dataY = dataY.toarray()
    assert dataX.ndim == len(coll.ddata[keyX]['ref']) == 3

    axist = refs.index(reft)
    axisch = refs.index(refch)
    axispts = [ii for ii in range(3) if refs[ii] not in [reft, refch]][0]
    nt = dataX.shape[axist]
    nch = dataX.shape[axisch]
    npts = dataX.shape[axispts]

    # ----------------------
    #  labels and data

    labx = f"{keyX} ({coll.ddata[keyX]['units']})"
    laby = f"{keyY} ({coll.ddata[keyY]['units']})"

    keyt, tstr, dt2, labt = _get_str_datadlab(
        keyX=keyt, nx=nt, islogX=islogt, coll=coll,
    )
    datat = coll.ddata[keyt]['data']
    keych, chstr, dch2, labch = _get_str_datadlab(
        keyX=keych, nx=nch, islogX=islogch, coll=coll,
    )
    datach = coll.ddata[keych]['data']

    # -----------
    # background

    if bck is True:

        # append nan
        bckx = np.insert(dataX, dataX.shape[axispts], np.nan, axis=axispts)
        bcky = np.insert(dataY, dataY.shape[axispts], np.nan, axis=axispts)

        # reshape into (nt, nch*(npts+1))
        ntot = nch*(npts + 1)
        order = 'C' if axisch < axispts else 'F'
        slibck = _class1_compute._get_slice(laxis=[axist], ndim=3)
        bckx = np.array([
            bckx[slibck(ii)].reshape((ntot,), order=order)
            for ii in range(nt)
        ])
        bcky = np.array([
            bcky[slibck(ii)].reshape((ntot,), order=order)
            for ii in range(nt)
        ])

        # add ref
        kbck = 'nch*(npts+1)'
        coll.add_ref(
            key=kbck,
            size=ntot,
        )

        # add data
        kbckx = f'{keyX}-bck'
        coll.add_data(
            key=kbckx,
            data=bckx,
            ref=(reft, kbck),
        )
        kbcky = f'{keyY}-bck'
        coll.add_data(
            key=kbcky,
            data=bcky,
            ref=(reft, kbck),
        )

    # -----------------
    #  prepare slicing

    # here slice X and Y alike => slice in dim Y and vice-versa
    sli = _class1_compute._get_slice(laxis=[axist, axisch], ndim=3)

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
        fig.suptitle(f'{keyX} vs {keyY}', size=14, fontweight='bold')
        gs = gridspec.GridSpec(ncols=5, nrows=2, **dmargin)

        # ax0 = time
        ax0 = fig.add_subplot(gs[0, :2])
        ax0.set_ylabel(labt)
        ax0.set_xlabel('index')

        # ax1 = chan
        ax1 = fig.add_subplot(gs[1, :2])
        ax1.set_ylabel(labch)
        ax1.set_xlabel('index')

        # ax2 = lines
        ax2 = fig.add_subplot(gs[:, 2:-1], aspect=aspect)
        ax2.set_ylabel(laby)
        ax2.set_xlabel(labx)
        ax2.set_xlim(np.nanmin(dataX), np.nanmax(dataX))
        ax2.set_ylim(np.nanmin(dataY), np.nanmax(dataY))

        # axes for text
        ax3 = fig.add_subplot(gs[0, -1], frameon=False)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4 = fig.add_subplot(gs[1, -1], frameon=False)
        ax4.set_xticks([])
        ax4.set_yticks([])

        dax = {
            # data
            'time': {'handle': ax0},
            'chan': {'handle': ax1},
            'lines': {'handle': ax2},
            # text
            'text0': {'handle': ax3, 'type': 'text'},
            'text1': {'handle': ax4, 'type': 'text'},
        }

    dax = _generic_check._check_dax(dax=dax, main='lines')

    # ---------------
    # plot fixed part

    kax = 'time'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        im = ax.plot(
            range(nt),
            datat,
            c='k',
            ls='-',
            lw=1.,
        )

    kax = 'chan'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        im = ax.plot(
            range(nch),
            datach,
            c='k',
            ls='-',
            lw=1.,
        )

    # ----------------
    # define and set dgroup

    dgroup = {
        'time': {
            'ref': [reft],
            'data': ['index'],
            'nmax': nmax,
        },
        'chan': {
            'ref': [refch],
            'data': ['index'],
            'nmax': nmax,
        },
    }

    # ----------------
    # plot mobile part

    kax = 'lines'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for ii in range(nmax):

            if bck is True:
                l0, = ax.plot(
                    bckx[ind[0]],
                    bcky[ind[0]],
                    c=bck_color,
                    lw=1.,
                    ls='-',
                )

                k0 = f'bck{ii}'
                coll.add_mobile(
                    key=k0,
                    handle=l0,
                    refs=(reft, reft),
                    data=[kbckx, kbcky],
                    dtype=['xdata', 'ydata'],
                    axes=kax,
                    ind=ii,
                )

            # individual lines
            l0, = ax.plot(
                dataX[sli(ind[0], ind[1])],
                dataY[sli(ind[0], ind[1])],
                c=color_dict['time'][ii],
                lw=1.,
                ls='-',
            )

            k0 = f'l{ii}'
            coll.add_mobile(
                key=k0,
                handle=l0,
                refs=((reft, refch), (reft, refch)),
                data=(keyX, keyY),
                dtype=['xdata', 'ydata'],
                axes=kax,
                ind=ii,
            )

    kax = 'time'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l0 = ax.axvline(
                0,
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['time'][ii],
            )

            km = f't{ii}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(reft,),
                data='index',
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[reft], datax='index')

    kax = 'chan'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l0 = ax.axvline(
                datach[0],
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['chan'][ii],
            )

            km = f'ch{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(refch,),
                data='index',
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[refch], datax='index')

    # ---------
    # add text

    kax = 'text0'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            ax=ax,
            ref=reft,
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
            ref=refch,
            group='chan',
            ind=ind[1],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    return coll, dax, dgroup
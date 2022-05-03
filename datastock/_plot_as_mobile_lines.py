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
from ._plot_as_array import _check_keyXYZ, _get_str_datadlab


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


def plot_as_mobile_lines(
    # parameters
    coll=None,
    keyX=None,
    keyY=None,
    key_time=None,
    key_chan=None,
    aspect=None,
    ind=None,
    nmax=None,
    color_dict=None,
    dinc=None,
    lkeys=None,
    bstr_dict=None,
    rotation=None,
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
    [keyX, keyY], inplace, coll2 = _generic_check._check_inplace(
        coll=coll,
        keys=[keyX, keyY],
        inplace=inplace,
    )
    ndim = coll2._ddata[keyX]['data'].ndim

    # --------------
    # check input

    (
        keyX, keyY, refs,
        keyt, reft, islogt,
        keych, refch, islogch,
        ind,
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

        coll2.show_commands()
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


def _plot_as_mobile_lines_check(
    ndim=None,
    coll=None,
    keyX=None,
    keyY=None,
    key_time=None,
    key_chan=None,
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
            "\t- keyX ref: {coll.ddata[keyX]['ref']}\n"
            "\t- keyY ref: {coll.ddata[keyY]['ref']}\n"
        )
        raise Exception(msg)

    # key_time, keyX
    keyt, reft, islogt = _check_keyXYZ(
        coll=coll, refs=refs, keyX=key_time, ndim=ndim, dimlim=1,
        uniform=False,
    )
    keych, refch, islogch = _check_keyXYZ(
        coll=coll, refs=refs, keyX=key_chan, ndim=ndim, dimlim=2,
        uniform=False,
    )

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
        ind,
        aspect, nmax,
        color_dict,
        rotation,
        connect,
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
    n0, n1 = dataX.shape

    axisch = refs.index(refch)
    nch = dataX.shape[axisch]

    # ----------------------
    #  labels and data

    keych, chstr, datach, dch2, labch = _get_str_datadlab(
        keyX=keych, nx=nch, islogX=islogch, coll=coll,
    )

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

        # ax0 = time
        ax0 = fig.add_subplot(gs[0, :2])

        # ax1 = chan
        ax1 = fig.add_subplot(gs[1, :2])
        ax1.set_ylabel('data')
        ax1.set_xlabel(labt)

        # ax2 = lines
        ax2 = fig.add_subplot(gs[:, 2:-1], aspect=aspect)
        ax2.set_ylabel('data')
        ax2.set_xlabel(labch)

        # axes for text
        ax3 = fig.add_subplot(gs[0, -1], frameon=False)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4 = fig.add_subplot(gs[1, -1], frameon=False)
        ax4.set_xticks([])
        ax4.set_yticks([])

        if tstr:
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

        dax[kax].update(refx=[reft], datax=keyt)

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

        dax[kax].update(refx=[refch], datax=keych)

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
    n0, n1, n2 = dataX.shape

    axist = refs.index(reft)
    axisch = refs.index(refch)
    nt = dataX.shape[axist]
    nch = dataX.shape[axisch]

    # ----------------------
    #  labels and data

    keyt, tstr, datat, dt2, labt = _get_str_datadlab(
        keyX=keyt, nx=nt, islogX=islogt, coll=coll,
    )
    keych, chstr, datach, dch2, labch = _get_str_datadlab(
        keyX=keych, nx=nch, islogX=islogch, coll=coll,
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

        # ax1 = chan
        ax1 = fig.add_subplot(gs[1, :2])
        ax1.set_ylabel('data')
        ax1.set_xlabel(labt)

        # ax2 = lines
        ax2 = fig.add_subplot(gs[:, 2:-1], aspect=aspect)
        ax2.set_ylabel('data')
        ax2.set_xlabel(labch)

        # axes for text
        ax3 = fig.add_subplot(gs[0, -1], frameon=False)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4 = fig.add_subplot(gs[1, -1], frameon=False)
        ax4.set_xticks([])
        ax4.set_yticks([])

        if tstr:
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

        dax[kax].update(refx=[reft], datax=keyt)

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

        dax[kax].update(refx=[refch], datax=keych)

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

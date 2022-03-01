# coding utf-8


# Built-in
# import itertools as itt
# import warnings


# Common
import numpy as np
import scipy.stats as scpstats
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.transforms as transforms
import matplotlib.lines as mlines
import matplotlib.colors as mcolors


# library-specific
from . import _generic_check
from . import _plot_text


# __all__ = ['plot_as_array']


_CONNECT = True
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


def plot_BvsA_as_distribution(
    # parameters
    coll=None,
    keyA=None,
    keyB=None,
    # customization of scatter plot
    dlim=None,
    color_dict=None,
    color_dict_logic=None,
    color_map=None,
    color_map_key=None,
    color_map_vmin=None,
    color_map_vmax=None,
    Amin=None,
    Amax=None,
    Bmin=None,
    Bmax=None,
    # customization of distribution plot
    nAbin=None,
    nBbin=None,
    dist_cmap=None,
    dist_min=None,
    dist_max=None,
    # customization of interactivity
    ind0=None,
    nmax=None,
    dinc=None,
    lkeys=None,
    bstr_dict=None,
    group_color_dict=None,
    inplace=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    aspect=None,
    connect=None,
):


    # ------------
    #  check inputs

    # keyA, keyB
    keyA = _generic_check._check_var(
        keyA, 'keyA',
        allowed=coll._ddata.keys(),
    )
    lkok = [
        k0 for k0, v0 in coll._ddata.items()
        if v0['ref'] == coll._ddata[keyA]['ref']
    ]
    keyB = _generic_check._check_var(
        keyB, 'keyB',
        allowed=lkok,
    )

    # check key, inplace flag and extract sub-collection
    keys, inplace, coll2 = _generic_check._check_inplace(
        coll=coll,
        keys=[keyA, keyB],
        inplace=inplace,
    )
    keyA, keyB = keys
    ndim = coll._ddata[keyA]['data'].ndim

    # -------------------------
    #  call appropriate routine

    if ndim == 1:

        return _plot_BvsA_1d(
            # parameters
            coll=coll2,
            keyA=keyA,
            keyB=keyB,
            # customization of scatter plot
            dlim=dlim,
            color_dict=color_dict,
            color_dict_logic=color_dict_logic,
            color_map=color_map,
            color_map_key=color_map_key,
            color_map_vmin=color_map_vmin,
            color_map_vmax=color_map_vmax,
            Amin=Amin,
            Amax=Amax,
            Bmin=Bmin,
            Bmax=Bmax,
            # customization of distribution plot
            nAbin=nAbin,
            nBbin=nBbin,
            dist_cmap=dist_cmap,
            dist_min=dist_min,
            dist_max=dist_max,
            # customization of interactivity
            ind0=ind0,
            nmax=nmax,
            dinc=dinc,
            lkeys=lkeys,
            bstr_dict=bstr_dict,
            group_color_dict=group_color_dict,
            # figure-specific
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
            aspect=aspect,
            connect=connect,
        )


    elif ndim == 2:

        return _plot_AvsB_2d(
        )


# #############################################################################
# #############################################################################
#                       check
# #############################################################################


def _plot_BvsA_check(
    ndim=None,
    # parameters
    coll=None,
    keyA=None,
    keyB=None,
    ind0=None,
    # customization of scatter plot
    dlim=None,
    color_dict=None,
    color_dict_logic=None,
    color_map=None,
    color_map_key=None,
    color_map_vmin=None,
    color_map_vmax=None,
    Amin=None,
    Amax=None,
    Bmin=None,
    Bmax=None,
    # customization of distribution plot
    nAbin=None,
    nBbin=None,
    dist_cmap=None,
    dist_min=None,
    dist_max=None,
    # customization of interactivity
    nmax=None,
    dinc=None,
    lkeys=None,
    bstr_dict=None,
    group_color_dict=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    aspect=None,
    connect=None,
    groups=None,
):

    # ind0
    ind0 = _generic_check._check_var(
        ind0, 'ind0',
        default=[0 for ii in range(ndim)],
        types=(list, tuple, np.ndarray),
    )
    c0 = (
        len(ind0) == ndim
        and all([
            np.isscalar(ii) and isinstance(ii, (int, np.integer))
            for ii in ind0
        ])
    )
    if not c0:
        msg = (
            f"Arg ind0 must be an iterable of {ndim} integer indices!\n"
            f"Provided: {ind0}"
        )
        raise Exception(msg)

    # color_dict vs color_cmap
    if color_dict is not None and color_map_key is not None:
        msg = (
            "Please provide either color_dict xor color_cmap!\n"
            f"\t- color_dict: {color_dict}\n"
            f"\t- color_map_key: {color_map_key}\n"
        )
        raise Exception(msg)

    # color_dict
    if color_map_key is None:

        if color_dict is None:
            shape = coll._ddata[keyA]['data'].shape
            color_dict = {'k': {'ind': np.ones(shape, dtype=bool)}}

        c0 = (
            isinstance(color_dict, dict)
            and all([
                mcolors.is_color_like(k0)
                and isinstance(v0, dict)
                for k0, v0 in color_dict.items()
            ])
        )
        if not c0:
            msg = (
                "Arg color_dict must be a dict of the form:\n"
                "  {'color0': {'data0': [data0lim0, data0lim1], ...}}\n"
                +  f"Provided:\n{color_dict}"
            )
            raise Exception(msg)

        for k0, v0 in color_dict.items():
            if v0.get('dlim') is None:
                lk1 = [k1 for k1 in v0.keys() if k1 not in ['ind', 'color']]
                if len(lk1) > 0:
                    color_dict[k0]['dlim'] = {k1: v0[k1] for k1 in lk1}
            if v0.get('color') is None:
                color_dict[k0]['color'] = k0

            if v0.get('ind') is None:
                color_dict[k0]['ind'] = _generic_check._apply_dlim(
                    dlim=color_dict[k0]['dlim'],
                    logic_intervals='all',
                    logic=color_dict_logic,
                    ddata=coll._ddata,
                )

    # color_map
    if color_map_key is not None:

        # color_map_key
        refs = coll._ddata[keyA]['ref']
        c0 = (
            color_map_key in coll._ddata.keys()
            and coll._ddata[color_map_key]['ref'] == refs
        )
        if not c0:
            msg = (
                "Arg color_map_key must be a valid coll.ddata key!\n"
                f"It must have ref = {refs}\n"
                "Provided:\n"
                f"\t- color_map_key: {color_map_key}\n"
                f"\t- ref: {coll._ddata[color_map_key]['ref']}\n"
            )
            raise Exception(msg)

        (
            color_map, color_map_vmin, color_map_vmax,
        ) = _generic_check._check_cmap_vminvmax(
            data=coll._ddata[color_map_key]['data'],
            cmap=color_map,
            vmin=color_map_vmin,
            vmax=color_map_vmax,
        )

    # Amin, Amax, Bmin, Bmax
    if Amin is None:
        Amin = np.nanmin(coll._ddata[keyA]['data'])
    if Amax is None:
        Amax = np.nanmax(coll._ddata[keyA]['data'])
    if Bmin is None:
        Bmin = np.nanmin(coll._ddata[keyB]['data'])
    if Bmax is None:
        Bmax = np.nanmax(coll._ddata[keyB]['data'])

    # distribution binning
    if nAbin is None:
        nAbin = 100
    if nBbin is None:
        nBbin = 100

    # nmax
    nmax = _generic_check._check_var(
        nmax, 'nmax',
        default=3,
        types=int,
    )

    # group_color_dict
    cdef = {
        k0: _LCOLOR_DICT[0] for ii, k0 in enumerate(groups)
    }
    group_color_dict = _generic_check._check_var(
        group_color_dict, 'group_color_dict',
        default=cdef,
        types=dict,
    )
    dout = {
        k0: str(v0)
        for k0, v0 in group_color_dict.items()
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
            "The following entries of group_color_dict are invalid:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

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

    # aspect
    aspect = _generic_check._check_var(
        aspect, 'aspect',
        default='equal',
        types=str,
        allowed=['equal', 'auto'],
    )

    # connect
    connect = _generic_check._check_var(
        connect, 'connect',
        default=_CONNECT,
        types=bool,
    )

    return (
        keyA,
        keyB,
        ind0,
        # customization of scatter plot
        dlim,
        color_dict,
        color_dict_logic,
        color_map,
        color_map_key,
        color_map_vmin,
        color_map_vmax,
        Amin,
        Amax,
        Bmin,
        Bmax,
        # customization of distribution plot
        nAbin,
        nBbin,
        dist_cmap,
        dist_min,
        dist_max,
        # customization of interactivity
        nmax,
        dinc,
        lkeys,
        bstr_dict,
        group_color_dict,
        # figure-specific
        dax,
        dmargin,
        fs,
        dcolorbar,
        dleg,
        aspect,
        connect,
        groups,
    )


# #############################################################################
# #############################################################################
#                       plot_as_array: 1d
# #############################################################################


def _plot_BvsA_1d(
    # parameters
    coll=None,
    keyA=None,
    keyB=None,
    # customization of scatter plot
    dlim=None,
    color_dict=None,
    color_dict_logic=None,
    color_map=None,
    color_map_key=None,
    color_map_vmin=None,
    color_map_vmax=None,
    Amin=None,
    Amax=None,
    Bmin=None,
    Bmax=None,
    # customization of distribution plot
    nAbin=None,
    nBbin=None,
    dist_cmap=None,
    dist_min=None,
    dist_max=None,
    # customization of interactivity
    ind0=None,
    nmax=None,
    dinc=None,
    lkeys=None,
    bstr_dict=None,
    group_color_dict=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    aspect=None,
    connect=None,
):

    # --------------
    # check input

    groups = ['ref']
    (
        keyA,
        keyB,
        ind0,
        # customization of scatter plot
        dlim,
        color_dict,
        color_dict_logic,
        color_map,
        color_map_key,
        color_map_vmin,
        color_map_vmax,
        Amin,
        Amax,
        Bmin,
        Bmax,
        # customization of distribution plot
        nAbin,
        nBbin,
        dist_cmap,
        dist_min,
        dist_max,
        # customization of interactivity
        nmax,
        dinc,
        lkeys,
        bstr_dict,
        group_color_dict,
        # figure-specific
        dax,
        dmargin,
        fs,
        dcolorbar,
        dleg,
        aspect,
        connect,
        groups,
    ) = _plot_BvsA_check(
        ndim=1,
        # parameters
        coll=coll,
        keyA=keyA,
        keyB=keyB,
        # customization of scatter plot
        dlim=dlim,
        color_dict=color_dict,
        color_dict_logic=color_dict_logic,
        color_map=color_map,
        color_map_key=color_map_key,
        color_map_vmin=color_map_vmin,
        color_map_vmax=color_map_vmax,
        Amin=Amin,
        Amax=Amax,
        Bmin=Bmin,
        Bmax=Bmax,
        # customization of distribution plot
        nAbin=nAbin,
        nBbin=nBbin,
        dist_cmap=dist_cmap,
        dist_min=dist_min,
        dist_max=dist_max,
        # customization of interactivity
        ind0=ind0,
        nmax=nmax,
        dinc=dinc,
        lkeys=lkeys,
        bstr_dict=bstr_dict,
        group_color_dict=group_color_dict,
        # misc
        dcolorbar=dcolorbar,
        dleg=dleg,
        aspect=aspect,
        connect=connect,
        groups=groups,
    )

    # --------------
    #  Prepare data

    dataA = coll.ddata[keyA]['data']
    dataB = coll.ddata[keyB]['data']
    if hasattr(dataA, 'nnz'):
        dataA = dataA.toarray()
    if hasattr(dataB, 'nnz'):
        dataB = dataB.toarray()
    c0 = (
        dataA.dtype == dataB.dtype
        and dataA.dtype in [int, float, bool]
    )
    if not c0:
        msg = (
            "Data type should be in [int, float, bool]\n"
            f"\t- Provided: {dataA.dtype}"
        )
        raise Exception(msg)

    ref = coll._ddata[keyA]['ref'][0]
    unitsA = coll._ddata[keyA]['units']
    unitsB = coll._ddata[keyB]['units']
    laby = f'{keyB} ({unitsB})'
    labx = f'{keyA} ({unitsA})'

    # ----------
    # set limits

    if dlim is not None:
        # get indices
        ind = _generic_check._apply_dlim(
            dlim=dlim,
            logic_intervals='all',
            logic=logic,
            ddata=coll._ddata,
        )

        # implement
        dataA[~ind] = np.nan
        dataB[~ind] = np.nan

    # ----------------------------
    #  Binning for 2d distribution

    Agrid = np.linspace(Amin, Amax, nAbin)
    Agridplot = 0.5*(Agrid[1:] + Agrid[:-1])
    Agridplot = np.tile(Agridplot, (nAbin-1, 1))
    Bgrid = np.linspace(Bmin, Bmax, nBbin)
    Bgridplot = 0.5*(Bgrid[1:] + Bgrid[:-1])
    Bgridplot = np.tile(Bgridplot, (nBbin-1, 1))
    extent = (Amin, Amax, Bmin, Bmax)

    iok = np.isfinite(dataA) & np.isfinite(dataB)
    databin = scpstats.binned_statistic_2d(
        dataA[iok],
        dataB[iok],
        np.ones((iok.sum(),), dtype=int),
        statistic='sum',
        bins=(Agrid, Bgrid),
    )[0]
    databin[databin == 0] = np.nan

    # --------------
    # plot - prepare

    if dax is None:

        if fs is None:
            fs = (12, 8)

        if dmargin is None:
            dmargin = {
                'left': 0.08, 'right': 0.95,
                'bottom': 0.08, 'top': 0.95,
                'hspace': 0.20, 'wspace': 0.20,
            }

        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(ncols=3, nrows=2, **dmargin)

        ax0 = fig.add_subplot(gs[0, :2], aspect=aspect)
        ax0.set_xlabel(labx)
        ax0.set_ylabel(laby)

        ax1 = fig.add_subplot(gs[1, :2], sharex=ax0, sharey=ax0)
        ax1.set_xlabel(labx)
        ax1.set_ylabel(laby)

        ax2 = fig.add_subplot(gs[:, 2], frameon=False)
        ax2.set_xticks([])
        ax2.set_yticks([])

        dax = {
            'scatter': {'handle': ax0, 'type': 'misc'},
            'dist': {'handle': ax1, 'type': 'misc'},
            'text': {'handle': ax2, 'type': 'text'},
        }

    dax = _generic_check._check_dax(dax=dax, main='misc')

    # ---------------
    # plot fixed part

    kax = 'dist'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        im = ax.imshow(
            databin.T,
            cmap=dist_cmap,
            vmin=dist_min,
            vmax=dist_max,
            interpolation='nearest',
            extent=extent,
            origin='lower',
            aspect=aspect,
        )

        plt.colorbar(im, ax=ax)

    kax = 'scatter'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        if color_dict is None:
            im = ax.scatter(
                dataA,
                dataB,
                s=4,
                c=coll._ddata[color_map_key]['data'],
                marker='.',
                edgecolors='None',
                cmap=color_map,
                vmin=color_map_vmin,
                vmax=color_map_vmax,
            )
            plt.colorbar(im, ax=ax)
        else:
            for k0, v0 in color_dict.items():
                ax.plot(
                    dataA[v0['ind']],
                    dataB[v0['ind']],
                    color=v0['color'],
                    marker='.',
                    ls='None',
                    ms=4.,
                )

    # ----------------
    # define and set dgroup

    # only ref / data used for index propagation
    # list unique ref and a single data per ref
    dgroup = {
        ref: {
            'ref': [ref],
            'data': ['index'],
            'nmax': nmax,
        },
    }

    # ----------------
    # plot mobile part

    kax = 'scatter'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # ind0
        for ii in range(nmax):
            mi, = ax.plot(
                dataA[ind0[0]],
                dataB[ind0[0]],
                marker='o',
                markeredgewidth=3.,
                markerfacecolor='none',
                markeredgecolor=group_color_dict['ref'][ii],
                ls='None',
                ms=8.,
            )

            # update coll
            km = f'm{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=mi,
                ref=[ref, ref],
                data=[keyA, keyB],
                dtype=['xdata', 'ydata'],
                ax=kax,
                ind=ii,
            )

        dax[kax].update(
            refx=[ref],
            refy=[ref],
            datax=[keyA],
            datay=[keyB],
        )

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
            group='ref',
            ind=ind0[0],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=group_color_dict,
            bstr_dict=bstr_dict,
        )

    # --------------------------------
    # add axes and setup interactivity

    # add axes
    for kax in dax.keys():
        coll.add_axes(key=kax, **dax[kax])

    # setup interactivity
    coll.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)

    # connect
    if connect is True:
        coll.disconnect_old()
        coll.connect()

    return coll

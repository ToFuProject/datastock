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
from . import _class2_interactivity


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
    keyX=None,
    axis=None,
    # customization of scatter plot
    dlim=None,
    dlim_logic=None,
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
    marker_size=None,
    # customization of distribution plot
    nAbin=None,
    nBbin=None,
    dist_cmap=None,
    dist_min=None,
    dist_max=None,
    dist_sample_min=None,
    dist_rel=None,
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
            dlim_logic=dlim_logic,
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
            marker_size=marker_size,
            # customization of distribution plot
            nAbin=nAbin,
            nBbin=nBbin,
            dist_cmap=dist_cmap,
            dist_min=dist_min,
            dist_max=dist_max,
            dist_sample_min=dist_sample_min,
            dist_rel=dist_rel,
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

        return _plot_BvsA_2d(
            # parameters
            coll=coll2,
            keyA=keyA,
            keyB=keyB,
            keyX=keyX,
            axis=axis,
            # customization of scatter plot
            dlim=dlim,
            dlim_logic=dlim_logic,
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
            marker_size=marker_size,
            # customization of distribution plot
            nAbin=nAbin,
            nBbin=nBbin,
            dist_cmap=dist_cmap,
            dist_min=dist_min,
            dist_max=dist_max,
            dist_sample_min=dist_sample_min,
            dist_rel=dist_rel,
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
    keyX=None,
    axis=None,
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
    marker_size=None,
    # customization of distribution plot
    nAbin=None,
    nBbin=None,
    dist_cmap=None,
    dist_min=None,
    dist_max=None,
    dist_sample_min=None,
    dist_rel=None,
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

    # keyX vs axis
    shape = coll._ddata[keyA]['data'].shape
    ref = coll._ddata[keyA]['ref']
    assert ndim == len(shape) == len(ref), ref

    if ndim == 1:
        refX = None
    elif ndim == 2:
        if keyX is None:
            if axis is None:
                axis = 1
            assert axis in [0, 1], axis
            keyX = ref[axis]
            refX = keyX

        else:
            c0 = (
                keyX in ref
                or (
                    keyX in coll._ddata.keys()
                    and (
                        coll._ddata[keyX]['ref'] == ref
                        or (
                            len(coll._ddata[keyX]['ref']) == 1
                            and coll._ddata[keyX]['ref'][0] in ref
                        )
                    )
                )
            )
            if not c0:
                msg = (
                    "Arg keyX must be a valid ref / data key with same ref as keyA"
                    f"\nProvided: {keyX}"
                )
                raise Exception(msg)

            # Deduce refX and axis
            refX = keyX if keyX in ref else coll._ddata[keyX]['ref']
            if isinstance(refX, str):
                axis = ref.index(keyX)
            elif len(refX) == 1:
                refX = refX[0]
                axis = ref.index(refX)
            else:
                if axis is None:
                    axis = 1
                refX = refX[axis]

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

            if v0['ind'].shape != shape:
                c0 = (
                    v0['ind'].shape
                    == tuple([aa for aa in shape if aa in v0['ind'].shape])
                )
                if not c0:
                    msg = (
                        f"data used in color_dict['{k0}'] has wrong shape!\n"
                        f"\t- Provided: {v0['ind'].shape} vs {shape}"
                    )
                    raise Exception(msg)

                # reshape
                shap = tuple([
                    aa if aa in v0['ind'].shape else 1
                    for aa in shape
                ])
                color_dict[k0]['ind'] = v0['ind'].reshape(shap)

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

    # marker_size
    marker_size = _generic_check._check_var(
        marker_size, 'markr_size',
        default=2,
        types=int,
    )

    # dist_sample_min
    dist_sample_min = _generic_check._check_var(
        dist_sample_min, 'dist_sample_min',
        default=1,
        types=int,
    )

    # dist_rel
    dist_rel = _generic_check._check_var(
        dist_rel, 'dist_rel',
        default=False,
        types=bool,
    )

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
        keyX,
        refX,
        axis,
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
        marker_size,
        # customization of distribution plot
        nAbin,
        nBbin,
        dist_cmap,
        dist_min,
        dist_max,
        dist_sample_min,
        dist_rel,
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
    dlim_logic=None,
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
    marker_size=None,
    # customization of distribution plot
    nAbin=None,
    nBbin=None,
    dist_cmap=None,
    dist_min=None,
    dist_max=None,
    dist_sample_min=None,
    dist_rel=None,
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
        _,
        _,
        _,
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
        marker_size,
        # customization of distribution plot
        nAbin,
        nBbin,
        dist_cmap,
        dist_min,
        dist_max,
        dist_sample_min,
        dist_rel,
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
        marker_size=marker_size,
        # customization of distribution plot
        nAbin=nAbin,
        nBbin=nBbin,
        dist_cmap=dist_cmap,
        dist_min=dist_min,
        dist_max=dist_max,
        dist_sample_min=dist_sample_min,
        dist_rel=dist_rel,
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
            logic=dlim_logic,
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

    databin[databin < dist_sample_min] = np.nan

    if dist_rel is True:
        databin = databin / np.nansum(databin)

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
                s=marker_size,
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
                    ms=marker_size,
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
                ms=marker_size,
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


# #############################################################################
# #############################################################################
#                       plot_as_array: 2d
# #############################################################################


def _plot_BvsA_2d(
    # parameters
    coll=None,
    keyA=None,
    keyB=None,
    keyX=None,
    axis=None,
    # customization of scatter plot
    dlim=None,
    dlim_logic=None,
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
    marker_size=None,
    # customization of distribution plot
    nAbin=None,
    nBbin=None,
    dist_cmap=None,
    dist_min=None,
    dist_max=None,
    dist_sample_min=None,
    dist_rel=None,
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
        keyX,
        refX,
        axis,
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
        marker_size,
        # customization of distribution plot
        nAbin,
        nBbin,
        dist_cmap,
        dist_min,
        dist_max,
        dist_sample_min,
        dist_rel,
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
        ndim=2,
        # parameters
        coll=coll,
        keyA=keyA,
        keyB=keyB,
        keyX=keyX,
        axis=axis,
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
        marker_size=marker_size,
        # customization of distribution plot
        nAbin=nAbin,
        nBbin=nBbin,
        dist_cmap=dist_cmap,
        dist_min=dist_min,
        dist_max=dist_max,
        dist_sample_min=dist_sample_min,
        dist_rel=dist_rel,
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

    # ----------------------
    #  Transpose if axis = 1

    sli = _class2_interactivity._get_slice(laxis=[1-axis], ndim=2)

    # --------------
    #  Prepare data

    dataA = coll.ddata[keyA]['data']
    dataB = coll.ddata[keyB]['data']
    refs = coll._ddata[keyA]['ref']
    assert refX == refs[axis], refs
    refselect = refs[1-axis]
    if keyX in refs:
        dataX = np.arange(0, coll._dref[keyX]['size'])
    else:
        dataX = coll.ddata[keyX]['data']
    if hasattr(dataA, 'nnz'):
        dataA = dataA.toarray()
    if hasattr(dataB, 'nnz'):
        dataB = dataB.toarray()
    if hasattr(dataX, 'nnz'):
        dataX = dataX.toarray()
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

    unitsA = coll._ddata[keyA]['units']
    unitsB = coll._ddata[keyB]['units']
    if keyX in refs:
        unitsX = 'index'
    else:
        unitsX = coll._ddata[keyX]['units']
    laby = f'{keyB} ({unitsB})'
    labx = f'{keyA} ({unitsA})'
    labX = f'{keyX} ({unitsX})'
    labdata = 'data'

    # ----------
    # set limits

    if dlim is not None:
        # get indices
        ind = _generic_check._apply_dlim(
            dlim=dlim,
            logic_intervals='all',
            logic=dlim_logic,
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
    databin[databin < dist_sample_min] = np.nan

    if dist_rel is True:
        databin = databin / np.nansum(databin)


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
        gs = gridspec.GridSpec(ncols=2, nrows=2, **dmargin)

        ax0 = fig.add_subplot(gs[0, 0], aspect=aspect)
        ax0.set_xlabel(labx)
        ax0.set_ylabel(laby)

        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0, sharey=ax0)
        ax1.set_xlabel(labx)
        ax1.set_ylabel(laby)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_xlabel(labX)
        ax2.set_ylabel(labdata)
        ax2.set_ylim(min(Amin, Bmin), max(Amax, Bmax))
        ax2.set_xlim(np.nanmin(dataX), np.nanmax(dataX))

        ax3 = fig.add_subplot(gs[1, 1], frameon=False)
        ax3.set_xticks([])
        ax3.set_yticks([])

        dax = {
            'scatter': {'handle': ax0, 'type': 'misc'},
            'dist': {'handle': ax1, 'type': 'misc'},
            'profile': {'handle': ax2, 'type': 'misc'},
            'text': {'handle': ax3, 'type': 'text'},
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
                dataA.ravel(),
                dataB.ravel(),
                s=marker_size,
                c=coll._ddata[color_map_key]['data'].ravel(),
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
                    ms=marker_size,
                )

    # ----------------
    # define and set dgroup

    # only ref / data used for index propagation
    # list unique ref and a single data per ref
    dgroup = {
        refselect: {
            'ref': [refselect],
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
                dataA[sli(ind0[0])],
                dataB[sli(ind0[0])],
                marker='o',
                markeredgewidth=3.,
                markerfacecolor='none',
                markeredgecolor=group_color_dict['ref'][ii],
                ls='-',
                ms=marker_size,
            )

            # update coll
            km = f'm{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=mi,
                ref=[refselect, refselect],
                data=[keyA, keyB],
                dtype=['xdata', 'ydata'],
                ax=kax,
                ind=ii,
            )

        dax[kax].update(
            refx=[refselect],
            refy=[refselect],
            datax=[keyA],
            datay=[keyB],
        )

    # ----------------
    # plot mobile profile

    kax = 'profile'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # ind0
        for dd, kk, ls in [(dataA, keyA, '-'), (dataB, keyB, '--')]:
            if dataX.ndim == 1:
                for ii in range(nmax):
                    li, = ax.plot(
                        dataX,
                        dd[sli(ind0[0])],
                        marker='o',
                        ms=marker_size,
                        color=group_color_dict['ref'][ii],
                        ls=ls,
                        lw=1.,
                    )

                    # update coll
                    km = f'l{kk}{ii:02.0f}'
                    coll.add_mobile(
                        key=km,
                        handle=li,
                        ref=[refselect],
                        data=[kk],
                        dtype=['ydata'],
                        ax=kax,
                        ind=ii,
                    )

            else:
                for ii in range(nmax):
                    li, = ax.plot(
                        dataX[sli(ind0[0])],
                        dd[sli(ind0[0])],
                        marker='o',
                        ms=marker_size,
                        color=group_color_dict['ref'][ii],
                        ls=ls,
                        lw=1.,
                    )

                    # update coll
                    km = f'l{kk}{ii:02.0f}'
                    coll.add_mobile(
                        key=km,
                        handle=li,
                        ref=[refselect, refselect],
                        data=[keyX, kk],
                        dtype=['xdata', 'ydata'],
                        ax=kax,
                        ind=ii,
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
            ref=refselect,
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

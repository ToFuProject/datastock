# coding utf-8


# Built-in
# import itertools as itt
# import warnings


# Common
import numpy as np
import scipy.stats as scpstats
import matplotlib.pyplot as plt
from matplotlib import gridspec


# library-specific
from . import _generic_check
from . import _plot_BvsA_as_distribution_check
from . import _plot_text
from . import _class2_interactivity


__all__ = ['plot_BvsA_as_distribution']


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
    add_bisector=None,
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

    # ----------------------------------
    #  check inputs - all others

    (
        coll2,
        keyA,
        refA,
        dataA,
        keyB,
        refB,
        dataB,
        refs,
        ref0,
        keyX,
        refX,
        dataX,
        ndim,
        shape,
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
        color_map_data,
        Amin,
        Amax,
        Bmin,
        Bmax,
        marker_size,
        linestyle,
        add_bisector,
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
    ) = _plot_BvsA_as_distribution_check._plot_BvsA_check(
        inplace=inplace,
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
        add_bisector=add_bisector,
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
    )

    # -------------------------
    #  Prepare data

    if ndim == 1:
        sli = lambda ind: ind
    else:
        sli = _class2_interactivity._get_slice(laxis=[1-axis], ndim=2)

    # --------------
    #  Prepare data

    unitsA = coll2._ddata[keyA]['units']
    unitsB = coll2._ddata[keyB]['units']
    laby = f'{keyB} ({unitsB})'
    labx = f'{keyA} ({unitsA})'
    if ndim == 2:
        if keyX in refs:
            unitsX = 'index'
        else:
            unitsX = coll2._ddata[keyX]['units']
        labX = f'{keyX} ({unitsX})'
    else:
        labX = None
    labdata = 'data'

    ABmin = min(Amin, Bmin)
    ABmax = max(Amax, Bmax)
    if ndim == 2:
        Xmin = np.nanmin(dataX)
        Xmax = np.nanmax(dataX)
    else:
        Xmin, Xmax = None, None

    # ----------
    # set limits

    if dlim is not None:
        # get indices
        ind = _generic_check._apply_dlim(
            dlim=dlim,
            logic_intervals='all',
            logic=dlim_logic,
            ddata=coll2._ddata,
        )

        # implement
        dataA[~ind] = np.nan
        dataB[~ind] = np.nan

    # ----------------------------
    #  Binning for 2d distribution

    databin, extent = _compute_dist(
        # A grid
        Amin=Amin,
        Amax=Amax,
        nAbin=nAbin,
        # B grid
        Bmin=Bmin,
        Bmax=Bmax,
        nBbin=nBbin,
        # A, B data
        dataA=dataA,
        dataB=dataB,
        # parameters
        dist_sample_min=dist_sample_min,
        dist_rel=dist_rel,
    )

    # -------------------
    # plot - prepare axes

    dax = _prepare_dax(
        ndim=ndim,
        dax=dax,
        fs=fs,
        dmargin=dmargin,
        aspect=aspect,
        labx=labx,
        laby=laby,
        labX=labX,
        labdata=labdata,
        ABmin=ABmin,
        ABmax=ABmax,
        Xmin=Xmin,
        Xmax=Xmax,
    )

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
                s=marker_size**2,
                c=color_map_data,
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

        # Add bisector
        if add_bisector is True:
            ax.plot(
                [ABmin, ABmax],
                [ABmin, ABmax],
                ls='--',
                c='k',
                lw=1.,
            )

    # ----------------
    # define and set dgroup

    # only ref / data used for index propagation
    # list unique ref and a single data per ref
    dgroup = {
        ref0: {
            'ref': [ref0],
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
                ls=linestyle,
                ms=marker_size,
            )

            # update coll
            km = f'm{ii:02.0f}'
            coll2.add_mobile(
                key=km,
                handle=mi,
                ref=[ref0, ref0],
                data=[keyA, keyB],
                dtype=['xdata', 'ydata'],
                axes=kax,
                ind=ii,
            )

        dax[kax].update(
            refx=[ref0],
            refy=[ref0],
            datax=[keyA],
            datay=[keyB],
        )

    # ----------------
    # plot mobile profile

    if ndim == 2:

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
                        coll2.add_mobile(
                            key=km,
                            handle=li,
                            ref=[ref0],
                            data=[kk],
                            dtype=['ydata'],
                            axes=kax,
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
                        coll2.add_mobile(
                            key=km,
                            handle=li,
                            ref=[ref0, ref0],
                            data=[keyX, kk],
                            dtype=['xdata', 'ydata'],
                            axes=kax,
                            ind=ii,
                        )

    # ---------
    # add text

    kax = 'text'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll2,
            kax=kax,
            ax=ax,
            ref=ref0,
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
    for ii, kax in enumerate(dax.keys()):
        harmonize = ii == len(dax) - 1
        coll2.add_axes(key=kax, harmonize=harmonize, **dax[kax])

    # connect
    if connect is True:
        coll2.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)
        coll2.disconnect_old()
        coll2.connect()

    return coll2


# #############################################################################
# #############################################################################
#                      Utilities in common
# #############################################################################


def _compute_dist(
    # A grid
    Amin=None,
    Amax=None,
    nAbin=None,
    # B grid
    Bmin=None,
    Bmax=None,
    nBbin=None,
    # A, B data
    dataA=None,
    dataB=None,
    # parameters
    dist_sample_min=None,
    dist_rel=None,
):
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
    return databin, extent


def _prepare_dax(
    ndim=None,
    dax=None,
    fs=None,
    dmargin=None,
    aspect=None,
    labx=None,
    laby=None,
    labX=None,
    labdata=None,
    ABmin=None,
    ABmax=None,
    Xmin=None,
    Xmax=None,
):
    if dax is None:

        if ndim == 1:
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

        elif ndim == 2:
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
            ax2.set_ylim(ABmin, ABmax)
            ax2.set_xlim(Xmin, Xmax)

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
    return dax

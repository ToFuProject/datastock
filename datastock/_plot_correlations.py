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


__all__ = ['plot_crosscorrelations']


# #############################################################################
# #############################################################################
#                       generic entry point
# #############################################################################


def plot_crosscorrelations(
    # parameters
    coll=None,
    ref=None,
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

    dcross = {
        rr
    }

    



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

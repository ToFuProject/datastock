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


__all__ = ['plot_correlations']


# #############################################################################
# #############################################################################
#                       generic entry point
# #############################################################################


def plot_correlations(
    coll=None,
    # parameters
    dcross=None,
    # customization of scatter plot
    cmap=None,
    vmin=None,
    vmax=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    aspect=None,
    rotation=None,
    inverty=None,
    # interactivity
    connect=None,
):

    # ----------------------------------
    #  check inputs - all others

    (
        # customization of scatter plot
        cmap,
        vmin,
        vmax,
        # figure-specific
        dax,
        dmargin,
        fs,
        aspect,
        rotation,
        inverty,
        # interactivity
        connect,
    ) = _check(
        coll=coll,
        dcross=dcross,
        # customization of scatter plot
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        # figure
        aspect=aspect,
        rotation=rotation,
        inverty=inverty,
        # interactivity
        connect=connect,
    )

    # -----------------
    #  Prepare data

    correlations = [
        k0 for k0 in list(dcross.values())[0].keys()
        if k0 not in ['reshape', 'keys0', 'keys1']
    ]

    from . import _class
    st = _class.DataStock()

    for ii, (k0, v0) in enumerate(dcross.items()):
        shape = (len(v0['keys0']), len(v0['keys1']))
        if ii == 0:
            st.add_ref(key='base', size=shape[0])
        st.add_ref(key=str(k0), size=shape[1])

    for k0, v0 in dcross.items():
        for k1 in correlations:
            st.add_data(key=f'{k1} - {k0}', data=v0[k1], ref=('base', str(k0)))

    # -------------------
    # plot - prepare axes

    lcross = list(dcross.keys())
    dax = _prepare_dax(
        dcross=dcross,
        lcorr=correlations,
        dax=dax,
        fs=fs,
        dmargin=dmargin,
        aspect=aspect,
        rotation=rotation,
        inverty=inverty,
    )

    # ---------------
    # plot fixed part

    for k0, v0 in dcross.items():

        shape = (len(v0['keys0']), len(v0['keys1']))
        extent = (-0.5, shape[1]-0.5, -0.5, shape[0]-0.5)

        for k1 in correlations:

            kax = f'{k1} - {k0}'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                ax.imshow(
                    v0[k1],
                    extent=extent,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation='nearest',
                    origin='lower',
                )

                if inverty is True:
                    ax.invert_yaxis()

    # --------------------------------
    # add axes and setup interactivity

    # add axes
    # for ii, kax in enumerate(dax.keys()):
        # harmonize = ii == len(dax) - 1
        # coll.add_axes(key=kax, harmonize=harmonize, **dax[kax])

    # connect
    # if connect is True:
        # coll.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)
        # coll.disconnect_old()
        # coll.connect()

    return coll


# #############################################################################
# #############################################################################
#                       utilities
# #############################################################################


def _check(
    coll=None,
    # parameters
    dcross=None,
    # customization of scatter plot
    cmap=None,
    vmin=None,
    vmax=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    aspect=None,
    rotation=None,
    inverty=None,
    # interactivity
    connect=None,
):

    # check dimensions
    ndim = [
        coll._ddata[v0['keys1'][0]]['data'].ndim for v0 in dcross.values()
    ]
    if max(ndim) > 2:
        msg = "plot_correlations() not implemented for arary of dim > 2!"
        raise NotImplementedError(msg)

    # color map, min, max
    (
        cmap, vmin, vmax,
    ) = _generic_check._check_cmap_vminvmax(
        data=[-1, 1],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # aspect
    aspect = _generic_check._check_var(
        aspect, 'aspect',
        default='equal',
        types=str,
        allowed=['auto', 'equal'],
    )

    # rotation
    rotation = _generic_check._check_var(
        rotation, 'rotation',
        default=45,
        types=(float, int),
    )

    # inverty
    inverty = _generic_check._check_var(
        inverty, 'inverty',
        default=True,
        types=bool,
    )

    # connect
    connect = _generic_check._check_var(
        connect, 'connect',
        default=True,
        types=bool,
    )

    return (
        # customization of scatter plot
        cmap,
        vmin,
        vmax,
        # figure-specific
        dax,
        dmargin,
        fs,
        aspect,
        rotation,
        inverty,
        # interactivity
        connect,
    )


def _prepare_dax(
    dcross=None,
    lcorr=None,
    dax=None,
    fs=None,
    dmargin=None,
    aspect=None,
    rotation=None,
    inverty=None,
):
    if dax is None:

        nax = len(dcross)
        if len(lcorr) > 1:
            nrows = 2*len(lcorr) + 1
        else:
            nrows = 2

        if fs is None:
            fs = (13, 9)

        if dmargin is None:
            dmargin = {
                'left': 0.08, 'right': 0.95,
                'bottom': 0.08, 'top': 0.95,
                'hspace': 0.20, 'wspace': 0.20,
            }

        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(
            ncols=nax + 1,
            nrows=nrows,
            **dmargin,
        )

        dax = {}
        axref = None
        daxref = dict.fromkeys(dcross.keys())

        for ii in range(len(lcorr) + 1):
            if ii < len(lcorr):
                k0 = lcorr[ii]
            if len(lcorr) == 1 and ii > 0:
                continue

            for jj, (k1, v1) in enumerate(dcross.items()):

                # create ax
                if jj == 0:
                    if ii == 0:
                        ax = fig.add_subplot(gs[2*ii:2*ii+2, :2], aspect=aspect)
                        axref = ax
                    elif ii < len(lcorr):
                        ax = fig.add_subplot(
                            gs[2*ii:2*ii+2, :2],
                            sharex=axref,
                            sharey=axref,
                            aspect=aspect,
                        )
                    else:
                        ax = fig.add_subplot(gs[2*ii, :2])
                else:
                    if ii == 0:
                        ax = fig.add_subplot(
                            gs[2*ii:2*ii+2, jj+1],
                            sharey=axref,
                            aspect=aspect,
                        )
                        daxref[k1] = ax
                    elif ii < len(lcorr):
                        ax = fig.add_subplot(
                            gs[2*ii:2*ii+2, jj+1],
                            sharey=axref,
                            sharex=daxref[k1],
                            aspect=aspect,
                        )
                    else:
                        ax = fig.add_subplot(gs[2*ii, jj+1])

                # set xlim, ylim
                if ii == len(lcorr):
                    ax.set_xlim(-0.5, len(lcorr)-0.5)
                    ax.set_ylim(-1, 1)
                    ax.axhline(0, ls='--', lw=1., c='k')

                # set ticks
                if ii == len(lcorr):
                    ax.set_xticks(range(len(lcorr)))
                    ax.set_xticklabels(lcorr, rotation=rotation)
                elif ii == 0:
                    shape = (len(v1['keys0']), len(v1['keys1']))
                    ax.set_xticks(range(shape[1]))
                    ax.set_xticklabels(
                        v1['keys1'],
                        rotation=rotation,
                        horizontalalignment='left',
                        verticalalignment='bottom',
                    )
                    ax.xaxis.set_ticks_position('top')
                    ax.set_yticks(range(shape[0]))
                    ax.set_yticklabels(v1['keys0'])
                    ax.yaxis.set_ticks_position('right')
                else:
                    plt.setp(ax.get_xticklabels(), visible=False)
                    ax.yaxis.set_ticks_position('right')

                if jj > 0 and ii < len(lcorr):
                    plt.setp(ax.get_yticklabels(), visible=False)

                # set ylabels
                if jj == 0:
                    if ii == len(lcorr):
                        ax.set_ylabel('corr. coefs', size=12)
                    else:
                        ax.set_ylabel(k0, size=12, fontweight='bold')

                # define key
                if ii == len(lcorr):
                    kax = f'all - {k1}'
                    dax[kax] = {'handle': ax, 'type': 'misc'}
                else:
                    kax = f'{k0} - {k1}'
                    dax[kax] = {'handle': ax, 'inverty': inverty}

    dax = _generic_check._check_dax(dax=dax, main='misc')
    return dax

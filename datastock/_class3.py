

from ._class2 import *
from . import _plot_as_array
from . import _plot_correlations
from . import _plot_BvsA_as_distribution


class DataStock3(DataStock2):
    """ Provide default interactive plots """

    # -------------------
    # Generic plotting
    # -------------------

    def plot_as_array(
        self,
        # parameters
        key=None,
        keyX=None,
        keyY=None,
        keyZ=None,
        ind=None,
        vmin=None,
        vmax=None,
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
        # figure-specific
        dax=None,
        dmargin=None,
        fs=None,
        dcolorbar=None,
        dleg=None,
        connect=None,
        inplace=None,
    ):
        """ Plot the desired 2d data array as a matrix """
        return _plot_as_array.plot_as_array(
            # parameters
            coll=self,
            key=key,
            keyX=keyX,
            keyY=keyY,
            keyZ=keyZ,
            ind=ind,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            aspect=aspect,
            nmax=nmax,
            color_dict=color_dict,
            dinc=dinc,
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
            connect=connect,
            inplace=inplace,
        )

    def plot_correlations(
        self,
        # correlations
        data=None,
        ref=None,
        correlations=None,
        # plotting
        cmap=None,
        vmin=None,
        vmax=None,
        # figure
        dax=None,
        dmargin=None,
        fs=None,
        aspect=None,
        # interactivity
        connect=None,
    ):

        # compute
        dcross = self.compute_correlations(
            data=data,
            ref=ref,
            correlations=correlations,
            verb=False,
            returnas=dict,
        )

        # plot
        return _plot_correlations.plot_correlations(
            coll=self,
            # correlations
            dcross=dcross,
            # plot
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            # figure
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            aspect=aspect,
            # interactivity
            connect=connect,
        )

    def plot_BvsA_as_distribution(
        self,
        # parameters
        keyA=None,
        keyB=None,
        keyX=None,
        axis=None,
        # customization of scatter plot
        dlim=None,
        color_dict=None,
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
        inplace=None,
        # figure-specific
        dax=None,
        dmargin=None,
        fs=None,
        dcolorbar=None,
        dleg=None,
        connect=None,
    ):
        return _plot_BvsA_as_distribution.plot_BvsA_as_distribution(
            # parameters
            coll=self,
            keyA=keyA,
            keyB=keyB,
            keyX=keyX,
            axis=axis,
            # customization of scatter plot
            dlim=dlim,
            color_dict=color_dict,
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
            inplace=inplace,
            # misc
            dcolorbar=dcolorbar,
            dleg=dleg,
            connect=connect,
        )


    # def _plot_timetraces(self, ntmax=1,
                         # key=None, ind=None, Name=None,
                         # color=None, ls=None, marker=None, ax=None,
                         # axgrid=None, fs=None, dmargin=None,
                         # legend=None, draw=None, connect=None, lib=None):
        # plotcoll = self.to_PlotCollection(ind=ind, key=key,
                                          # Name=Name, dnmax={})
        # return _DataCollection_plot.plot_DataColl(
            # plotcoll,
            # color=color, ls=ls, marker=marker, ax=ax,
            # axgrid=axgrid, fs=fs, dmargin=dmargin,
            # draw=draw, legend=legend,
            # connect=connect, lib=lib,
        # )

    # def _plot_axvlines(
        # self,
        # which=None,
        # key=None,
        # ind=None,
        # param_x=None,
        # param_txt=None,
        # sortby=None,
        # sortby_def=None,
        # sortby_lok=None,
        # ax=None,
        # ymin=None,
        # ymax=None,
        # ls=None,
        # lw=None,
        # fontsize=None,
        # side=None,
        # dcolor=None,
        # dsize=None,
        # fraction=None,
        # figsize=None,
        # dmargin=None,
        # wintit=None,
        # tit=None,
    # ):
        # """ plot rest wavelengths as vertical lines """

        # # Check inputs
        # which, dd = self.__check_which(
            # which=which, return_dict=True,
        # )
        # key = self._ind_tofrom_key(which=which, key=key, ind=ind, returnas=str)

        # if sortby is None:
            # sortby = sortby_def
        # if sortby not in sortby_lok:
            # msg = (
                # """
                # For plotting, sorting can be done only by:
                # {}

                # You provided:
                # {}
                # """.format(sortby_lok, sortby)
            # )
            # raise Exception(msg)

        # return _DataCollection_plot.plot_axvline(
            # din=dd,
            # key=key,
            # param_x='lambda0',
            # param_txt='symbol',
            # sortby=sortby, dsize=dsize,
            # ax=ax, ymin=ymin, ymax=ymax,
            # ls=ls, lw=lw, fontsize=fontsize,
            # side=side, dcolor=dcolor,
            # fraction=fraction,
            # figsize=figsize, dmargin=dmargin,
            # wintit=wintit, tit=tit,
        # )


# #############################################################################
# #############################################################################
#            set __all__
# #############################################################################


__all__ = [
    sorted([k0 for k0 in locals() if k0.startswith('DataStock')])[-1]
]



from ._class2 import *
from . import _plot_as_array


class DataStock3(DataStock2):
    """ Provide default interactive plots """

    # -------------------
    # Generic plotting
    # -------------------

    def plot_as_array(
        self,
        # parameters
        key=None,
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
            # figure-specific
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
            connect=connect,
            inplace=inplace,
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

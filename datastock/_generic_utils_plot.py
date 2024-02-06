# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 17:06:23 2024

@author: dvezinet
"""


import numpy as np
from mpl_toolkits.mplot3d import Axes3D


from . import _generic_check


__all__ = [
    'set_aspect3d',
]


# ###############################################################
# ###############################################################
#                   set_aspect3d
# ###############################################################


def set_aspect3d(
    ax=None,
    margin=None,
):

    # ------------
    # check inputs

    ax, margin = _set_aspect3d_check(
        ax=ax,
        margin=margin,
    )

    # ------------------
    # set box aspect

    ax.set_box_aspect([1, 1, 1])

    # ------------------
    # get current limits

    # limits
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    # radius with margin
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0])) * margin

    # ------------------
    # set new limits

    # origin
    x0, y0, z0 = np.mean(limits, axis=1)

    # set limits
    ax.set_xlim3d([x0 - radius, x0 + radius])
    ax.set_ylim3d([y0 - radius, y0 + radius])
    ax.set_zlim3d([z0 - radius, z0 + radius])

    return


# ################
# check inputs
# ################


def _set_aspect3d_check(
    ax=None,
    margin=None,
):

    # --------------------
    # ax

    if not isinstance(ax, Axes3D):
        msg = (
            "Arg ax must be a Axes3DSubplot instance!\n"
            "Provided:\n"
            f"type: {type(ax)}\n"
            f"value: {ax}\n"
        )
        raise Exception(msg)

    # -------------------
    # margin

    margin = _generic_check._check_var(
        margin, 'margin',
        default=1.05,
        types=(int, float),
        sign='>=0',
    )

    return ax, margin


# ############################################################
# ############################################################
#                _get_str_datadlab
# ############################################################


def _get_str_datadlab(
    coll=None,
    keyX=None,
    refX=None,
    nx=None,
    islogX=None,
):

    keyX2 = keyX
    xstr = (
        (keyX != 'index')
        and coll.ddata[keyX]['data'].dtype.type == np.str_
    )

    # --------------------
    # add index vector

    if keyX == 'index' or xstr:
        keyX2 = f"{refX}_index"

        c0 = (
            keyX2 in coll.ddata.keys()
            and coll.ddata[keyX2]['ref'] == (refX,)
            and np.allclose(coll.ddata[keyX2]['data'], np.arange(nx))
        )

        if not c0:
            coll.add_data(
                key=keyX2,
                data=np.arange(0, nx),
                ref=refX,
                units='',
            )
        dX2 = 0.5
        if keyX == 'index':
            labX = "index"
        else:
            labX = ''

        if xstr is True:
            xstr = coll.ddata[keyX]['data']

    # -------------------------------------
    # keyX refers to exising numerical data

    else:
        if islogX is True:
            keyX2 = f"{keyX}_log10"
            coll.add_data(
                key=keyX2,
                data=np.log10(coll.ddata[keyX]['data']),
                ref=coll.ddata[keyX]['ref'],
                units=coll.ddata[keyX]['units'],
            )
            labX = r"$\log_{10}$" + f"({keyX} ({coll._ddata[keyX]['units']}))"
            dataX = coll.ddata[keyX2]['data']
            coll.remove_data(keyX, propagate=False)

        else:
            labX = f"{keyX} ({coll._ddata[keyX]['units']})"
            dataX = coll.ddata[keyX]['data']
        dX2 = np.nanmean(np.diff(dataX)) / 2.

    return keyX2, xstr, dX2, labX
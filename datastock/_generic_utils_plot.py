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

    ax, margin =  _check(
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


def _check(
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
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 08:53:00 2025

@author: dvezinet
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datastock as ds


# ###############################################################
# ###############################################################
#                  Main
# ###############################################################


def main(
    coll=None,
    data=None,
    dcolor=None,
    # options
    color_default=None,
    vmin=None,
    vmax=None,
    log=None,
):

    # ------------------
    # check inputs
    # ------------------

    data, dcolor, color_default, vmin, vmax, log = _check(
        coll=coll,
        data=data,
        dcolor=dcolor,
        color_default=color_default,
        vmin=vmin,
        vmax=vmax,
        log=log,
    )

    # ------------------
    # initialize
    # ------------------

    shape = data.shape + (4,)
    color = np.zeros(shape, dtype=float)

    # ------------------
    # compute - alpha
    # ------------------

    if log is True:
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)

        alpha = (np.log10(data) - vmin) / (vmax - vmin)

    else:
        alpha = (data - vmin) / (vmax - vmin)

    # ------------------
    # compute - colors
    # ------------------

    for k0, v0 in dcolor.items():

        sli = (v0['ind'], slice(0, 3))
        color[sli] = v0['color']

        sli = tuple([slice(None) for ii in range(data.ndim)] + [-1])
        color[sli] = alpha

    # ------------------
    # output
    # ------------------

    lcol = set([v0['color'] for v0 in dcolor.values()])
    dcolor = {
        'color': color,
        'meaning': {
            kc: [k0 for k0, v0 in dcolor.items() if v0['color'] == kc]
            for kc in lcol
        },
    }

    return dcolor


# ###############################################################
# ###############################################################
#                  check
# ###############################################################


def _check(
    coll=None,
    data=None,
    dcolor=None,
    # options
    color_default=None,
    vmin=None,
    vmax=None,
    log=None,
):

    # ------------------
    # data
    # ------------------

    lc = [
        isinstance(data, np.ndarray),
        isinstance(data, str) and data in coll.ddata.keys(),
    ]
    if lc[0]:
        pass
    elif lc[1]:
        data = coll.ddata[data]['data']
    else:
        msg = (
            "Arg data must be a np.ndarray or a key to an existing data!\n"
            f"Provided: {data}\n"
        )
        raise Exception(msg)


    # ------------------
    # dcolor
    # ------------------

    # --------------------
    # dcolor format check

    c0 = (
        isinstance(dcolor, dict)
        and all([
            isinstance(k0, str)
            and isinstance(v0, dict)
            and sorted(v0.keys()) == ['color', 'ind']
            for k0, v0 in dcolor.items()
        ])
    )
    if not c0:
        msg = (
            "Arg dcolor must be a dict of sub-dicts of shape:\n"
            "\t- 'key0': {'ind': ..., 'color': ...}\n"
            "\t-  ...\n"
            "\t- 'keyN': {'ind': ..., 'color': ...}\n"
            f"Provided:\n{dcolor}\n"
        )
        raise Exception(msg)

    # --------------------
    # ind and color checks

    dfail = {}
    shape = data.shape
    for k0, v0 in dcolor.items():

        c0 = (
            isinstance(v0['ind'], np.ndarray)
            and v0['ind'].shape == data.shape
            and v0['ind'].dtype == bool
        )
        if not c0:
            msg = f"'ind' must be a {shape} bool array, not {v0['ind']}"
            dfail[k0] = (msg,)

        if not mcolors.is_color_like(v0['color']):
            msg = f"'color' must be color-like, not {v0['color']}"
            if k0 in dfail:
                dfail[k0] = dfail[k0] + (msg,)
            else:
                dfail[k0] = (msg,)

    # raise exception
    if len(dfail) > 0:
        lmax = np.max([len(f"\t- {k0}: ") for k0 in dfail.keys()])
        lstr = [
            f"\t- {k0}:\n".ljust(lmax) + '\n'.join([
                "".ljust(lmax+4) + f"\t- {v1}".rjust(lmax)
                for ii, v1 in enumerate(v0)
            ])
            for k0, v0 in dfail.items()
        ]
        msg = (
            "Arg dcolor, the following keys have incorrect keys / values:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # ----------------------
    # format colors to rgb

    dcol = {}
    for k0, v0 in dcolor.items():
        if np.any(v0['ind']):
            dcol[k0] = {
                'ind': v0['ind'],
                'color': mcolors.to_rgb(v0['color']),
            }

    # ------------------
    # color_default
    # ------------------

    if color_default is None:
        color_default = 'k'
    if not mcolors.is_color_like(color_default):
        msg = (
            "Arg color_default must be color-like!\n"
            f"Provided: {color_default}\n"
        )
        raise Exception(msg)

    color_default = mcolors.to_rgb(color_default)

    # ------------------
    # vmin, vmax
    # ------------------

    vmin0 = np.nanmin(data)
    vmax0 = np.nanmax(data)

    # vmin
    if vmin is None:
        vmin = vmin0
    c0 = (np.isscalar(vmin) and np.isfinite(vmin) and vmin < vmax0)
    if not c0:
        msg = (
            f"Arg vmin must be a finite scalar below max ({vmax0})\n"
            f"Provided: {vmin}\n"
        )
        raise Exception(msg)

    # vmax
    if vmax is None:
        vmax = vmax0
    c0 = (np.isscalar(vmax) and np.isfinite(vmax) and vmax > vmin0)
    if not c0:
        msg = (
            f"Arg vmax must be a finite scalar above min ({vmin0})\n"
            f"Provided: {vmax}\n"
        )
        raise Exception(msg)

    # ordering
    if vmin >= vmax:
        msg = (
            "Arg vmin must be below vmax!\n"
            f"Provided:\n\t- vmin = {vmin}\n\t- vmax = {vmax}\n"
        )
        raise Exception(msg)

    # ------------------
    # log
    # ------------------

    log = ds._generic_check._check_var(
        log, 'log',
        types=bool,
        default=False,
    )

    return data, dcol, color_default, vmin, vmax, log
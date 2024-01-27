# coding utf-8


# Common
import numpy as np
import matplotlib.colors as mcolors


# library-specific
from . import _generic_check
from . import _plot_as_array_1d
from . import _plot_as_array_2d
from . import _plot_as_array_3d
from . import _plot_as_array_4d


__all__ = ['plot_as_array']


# ###############################################################
# ###############################################################
#                       DEFAULTS
# ###############################################################


_CONNECT = True
_LCOLOR = [
    [
        'tab:blue', 'tab:orange', 'tab:green',
        'tab:red', 'tab:purple',  'tab:brown',
        'tab:pink', 'tab:gray', 'tab:olive',
        'tab:cyan',
    ],
    ['r', 'g', 'b'],
    ['m', 'y', 'c'],
]


# ###############################################################
# ###############################################################
#                       generic entry point
# ###############################################################


def plot_as_array(
    # resource
    coll=None,
    # data
    key=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    keyU=None,
    # index
    ind=None,
    # scales
    vmin=None,
    vmax=None,
    cmap=None,
    ymin=None,
    ymax=None,
    aspect=None,
    # interactivity
    nmax=None,
    uniform=None,
    color_dict=None,
    dinc=None,
    lkeys=None,
    bstr_dict=None,
    rotation=None,
    inverty=None,
    bck=None,
    interp=None,
    show_commands=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    wintit=None,
    tit=None,
    dcolorbar=None,
    dleg=None,
    label=None,
    connect=None,
    inplace=None,
):

    # --------------
    #  check inputs
    # --------------

    # check key, inplace flag and extract sub-collection
    key, inplace, coll2 = _generic_check._check_inplace(
        coll=coll,
        keys=None if key is None else [key],
        inplace=inplace,
    )
    key = key[0]
    ndim = coll2._ddata[key]['data'].ndim

    # --------------
    # check input
    # --------------

    (
        key,
        keyX, refX, islogX,
        keyY, refY, islogY,
        keyZ, refZ, islogZ,
        keyU, refU, islogU,
        sameref, ind,
        cmap, vmin, vmax,
        ymin, ymax,
        aspect, nmax,
        color_dict,
        rotation,
        inverty,
        bck,
        interp,
        wintit, tit,
        dcolorbar, dleg, label, connect,
    ) = _check(**locals())

    # --------------------------------
    # Particular case: same references
    # --------------------------------

    if sameref:
        from ._class import DataStock
        cc = DataStock()
        lr = [('refX', refX), ('refY', refY), ('refZ', refZ), ('refU', refU)]
        lr = [tt for tt in lr if tt[1] is not None]
        for ii, (ss, rr) in enumerate(lr):
            cc.add_ref(key=f'{rr}-{ii}', size=coll.dref[rr]['size'])
        ref = tuple([f'{rr}-{ii}' for ii, (ss, rr) in enumerate(lr)])
        cc.add_data(key=key, data=coll2.ddata[key]['data'], ref=ref)
        return cc.plot_as_array()

    # -------------------------
    #  call appropriate routine
    # -------------------------

    if ndim == 1:
        func = _plot_as_array_1d.main

    elif ndim == 2:
        func = _plot_as_array_2d.main

    elif ndim == 3:
        func = _plot_as_array_3d.main

    elif ndim == 4:
        func = _plot_as_array_4d.main

    # -------------------------
    # call appropriate function
    # -------------------------

    coll2, dax, dgroup = func(
        # parameters
        coll=coll2,
        key=key,
        keyX=keyX,
        keyY=keyY,
        keyZ=keyZ,
        keyU=keyU,
        refX=refX,
        refY=refY,
        refZ=refZ,
        refU=refU,
        islogX=islogX,
        islogY=islogY,
        islogZ=islogZ,
        islogU=islogU,
        ind=ind,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        ymin=ymin,
        ymax=ymax,
        aspect=aspect,
        nmax=nmax,
        color_dict=color_dict,
        lkeys=lkeys,
        bstr_dict=bstr_dict,
        rotation=rotation,
        inverty=inverty,
        bck=bck,
        interp=interp,
        # figure-specific
        dax=dax,
        dmargin=dmargin,
        fs=fs,
        dcolorbar=dcolorbar,
        dleg=dleg,
        label=label,
    )

    # ----------------------------
    # add axes for interactivity
    # ----------------------------

    # add axes
    for ii, kax in enumerate(dax.keys()):

        harmonize = ii == len(dax.keys()) - 1
        if kax not in coll2.dax.keys():
            coll2.add_axes(key=kax, harmonize=harmonize, **dax[kax])

        else:
            dnc = {
                k0: f"{v0} vs {coll2.dax[kax][k0]}"
                for k0, v0 in dax[kax].items()
                if v0 != coll2.dax[kax][k0]
            }
            if len(dnc) != 0:
                lstr = [f"\t- {k0}: {v0}" for k0, v0 in dnc.items()]
                msg = (
                    f"Mismatching dax['{kax}']!\n"
                    + "\n".join(lstr)
                )
                raise Exception(msg)

    # ----------------------
    # connect interactivity
    # ----------------------

    if connect is True:
        coll2.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)
        coll2.disconnect_old()
        coll2.connect()

        coll2.show_commands(verb=show_commands)
        return coll2
    else:
        return coll2, dgroup


# ##############################################################
# ##############################################################
#                       check
# ##############################################################


def _check_uniform_lin(k0=None, ddata=None):

    v0 = ddata[k0]

    c0 = (
        v0['data'].dtype.type != np.str_
        and v0['monot'] == (True,)
        and np.allclose(
            np.diff(v0['data']),
            v0['data'][1] - v0['data'][0],
            equal_nan=False,
        )
    )
    return c0


def _check_uniform_log(k0=None, ddata=None):

    v0 = ddata[k0]

    c0 = (
        v0['data'].dtype.type != np.str_
        and v0['monot'] == (True,)
        and np.all(v0['data'] > 0.)
        and np.allclose(
            np.diff(np.log(v0['data'])),
            np.log(v0['data'][1]) - np.log(v0['data'][0]),
            equal_nan=False,
            atol=0.,
            rtol=1e-10,
        )
    )

    return c0


def _check_keyXYZ(
    coll=None,
    refs=None,
    keyX=None,
    keyXstr=None,
    ndim=None,
    dim_min=None,
    uniform=None,
    already=None,
):
    """ Ensure keyX refers to a monotonic and (optionally) uniform data

    """

    if uniform is None:
        uniform = True

    refX = None
    islog = False
    if ndim >= dim_min:
        if keyX is not None:
            if keyX in coll._ddata.keys():
                lok = [
                    k0 for k0, v0 in coll._ddata.items()
                    if len(v0['ref']) == 1
                    and v0['ref'][0] in refs
                    and (
                        v0['data'].dtype.type == np.str_
                        or v0['monot'] == (True,)
                    )
                ]

                # optional uniformity
                if uniform:
                    lok = [
                        k0 for k0 in lok
                        if _check_uniform_lin(k0=k0, ddata=coll._ddata)
                        or _check_uniform_log(k0=k0, ddata=coll._ddata)
                    ]

                try:
                    keyX = _generic_check._check_var(
                        keyX, keyXstr,
                        allowed=lok,
                    )
                except Exception as err:
                    msg = (
                        err.args[0]
                        + f"\nContraint on uniformity: {uniform}"
                    )
                    err.args = (msg,)
                    raise err

                refX = coll._ddata[keyX]['ref'][0]

                # islog
                islog = _check_uniform_log(k0=keyX, ddata=coll._ddata)

            elif keyX in refs:
                keyX, refX = 'index', keyX

            elif keyX == 'index':
                if already is None:
                    refX = refs[dim_min - 1]
                elif all([kk in already for kk in refs]):  # TBC
                    # sameref
                    refX = refs[dim_min - 1]
                    msg = (
                        "Special case\n"
                        "\t- refs: {refs}\n"
                        f"\t- '{keyXstr}': {keyX}\n"
                        f"\t- already: {already}"
                    )
                    raise Exception(msg)
                else:
                    refX = [kk for kk in refs if kk not in already][0]

            else:
                msg = (
                    f"Arg '{keyXstr}' refers to unknow data:\n"
                    f"\t- Provided: {keyX}"
                )
                raise Exception(msg)
        else:
            keyX = 'index'
            if already is None:
                refX = refs[dim_min - 1]
            elif all([kk in already for kk in refs]):  # TBC
                # sameref
                refX = refs[dim_min - 1]
                msg = (
                    "Special case\n"
                    "\t- refs: {refs}\n"
                    f"\t- '{keyXstr}': {keyX}\n"
                    f"\t- already: {already}"
                )
                raise Exception(msg)
            else:
                refX = [kk for kk in refs if kk not in already][0]
    else:
        keyX, refX, islog = None, None, None

    return keyX, refX, islog


def _check(
    ndim=None,
    coll=None,
    key=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    keyU=None,
    ind=None,
    cmap=None,
    vmin=None,
    vmax=None,
    ymin=None,
    ymax=None,
    aspect=None,
    nmax=None,
    uniform=None,
    color_dict=None,
    rotation=None,
    inverty=None,
    bck=None,
    interp=None,
    # figure
    wintit=None,
    tit=None,
    dcolorbar=None,
    dleg=None,
    data=None,
    label=None,
    connect=None,
    # unused
    **kwdargs,
):

    # --------
    # groups
    # --------

    if ndim == 1:
        groups = ['X']
    elif ndim == 2:
        groups = ['X', 'Y']
    elif ndim == 3:
        groups = ['X', 'Y', 'Z']
    elif ndim == 4:
        groups = ['X', 'Y', 'Z', 'U']
    else:
        msg = "ndim must be in [1, 2, 3]"
        raise Exception(msg)

    # ----------------------
    # keyX, keyY, keyZ, keyU
    # ----------------------

    refs = coll._ddata[key]['ref']

    (
        keyX, refX, islogX,
        keyY, refY, islogY,
        keyZ, refZ, islogZ,
        keyU, refU, islogU,
        sameref,
    ) = get_keyrefs(
        coll=coll,
        refs=refs,
        keyX=keyX,
        keyY=keyY,
        keyZ=keyZ,
        keyU=keyU,
        ndim=ndim,
        uniform=uniform,
    )

    # -----
    # ind
    # -----

    ind = _generic_check._check_var(
        ind, 'ind',
        default=[0 for ii in range(ndim)],
        types=(list, tuple, np.ndarray),
    )
    c0 = (
        len(ind) == ndim
        and all([
            np.isscalar(ii) and isinstance(ii, (int, np.integer))
            for ii in ind
        ])
    )
    if not c0:
        msg = (
            "Arg ind must be an iterable of 2 integer indices!\n"
            f"Provided: {ind}"
        )
        raise Exception(msg)

    # ---------
    # cmap
    # ---------

    if cmap is None or vmin is None or vmax is None:
        if isinstance(coll.ddata[key]['data'], np.ndarray):
            nanmax = np.nanmax(coll.ddata[key]['data'])
            nanmin = np.nanmin(coll.ddata[key]['data'])
        else:
            nanmax = coll.ddata[key]['data'].max()
            nanmin = coll.ddata[key]['data'].min()

        # diverging
        delta = nanmax - nanmin
        diverging = (
            nanmin * nanmax < 0
            and min(abs(nanmin), abs(nanmax)) > 0.1*delta
        )

    if cmap is None:
        if diverging:
            cmap = 'seismic'
        else:
            cmap = 'viridis'

    # -----------
    # vmin, vmax
    # -----------

    if vmin is None:
        if diverging:
            if isinstance(nanmin, np.bool_):
                vmin = 0
            else:
                vmin = -max(abs(nanmin), nanmax)
        else:
            vmin = nanmin

    if vmax is None:
        if diverging:
            vmax = max(abs(nanmin), nanmax)
        else:
            vmax = nanmax

    # ymin, ymax
    if ymin is None:
        ymin = vmin
    if ymax is None:
        ymax = vmax

    # -------
    # aspect

    aspect = _generic_check._check_var(
        aspect, 'aspect',
        default='equal',
        types=str,
        allowed=['auto', 'equal'],
    )

    # ------
    # nmax

    nmax = _generic_check._check_var(
        nmax, 'nmax',
        default=3,
        types=int,
    )

    # -----------
    # color_dict

    if color_dict is not None and not isinstance(color_dict, dict):
        if isinstance(color_dict, (list, tuple)):
            color_dict = {k0: color_dict for k0 in groups}
        elif mcolors.is_color_like(color_dict):
            color_dict = {k0: [color_dict]*nmax for k0 in groups}

    cdef = {
        k0: _LCOLOR[0] for ii, k0 in enumerate(groups)
    }
    color_dict = _generic_check._check_var(
        color_dict, 'color_dict',
        default=cdef,
        types=dict,
    )
    dout = {
        k0: str(v0)
        for k0, v0 in color_dict.items()
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
            "The following entries of color_dict are invalid:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # -----------------
    # other parameters

    # rotation
    rotation = _generic_check._check_var(
        rotation, 'rotation',
        default=45,
        types=(int, float),
    )

    # inverty
    inverty = _generic_check._check_var(
        inverty, 'inverty',
        default=keyY == 'index',
        types=bool,
    )

    # bck
    if coll.ddata[key]['data'].size > 10000:
        bckdef = 'envelop'
    else:
        bckdef = 'lines'
    bck = _generic_check._check_var(
        bck, 'bck',
        default=bckdef,
        allowed=['lines', 'envelop', False],
    )

    # interp
    interp = _generic_check._check_var(
        interp, 'interp',
        default='nearest',
        types=str,
        allowed=['nearest', 'bilinear', 'bicubic']
    )

    # --------------------
    # figure-specific
    # -------------------

    # wintit
    if wintit is not None:
        wintit = _generic_check._check_var(
            wintit, 'wintit',
            types=str,
        )

    # tit
    tit = _generic_check._check_var(
        tit, 'tit',
        default=key,
        types=str,
    )

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

    # label
    label = _generic_check._check_var(
        label, 'label',
        default=True,
        types=bool,
    )

    # connect
    connect = _generic_check._check_var(
        connect, 'connect',
        default=_CONNECT,
        types=bool,
    )

    return (
        key,
        keyX, refX, islogX,
        keyY, refY, islogY,
        keyZ, refZ, islogZ,
        keyU, refU, islogU,
        sameref, ind,
        cmap, vmin, vmax,
        ymin, ymax,
        aspect, nmax,
        color_dict,
        rotation,
        inverty,
        bck,
        interp,
        wintit, tit,
        dcolorbar, dleg, label, connect,
    )


def get_keyrefs(
    coll=None,
    refs=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    keyU=None,
    ndim=None,
    uniform=None,
):

    # -----------
    # find order

    dk = {
        'keyX': {'key': keyX, 'ref': None, 'islog': None, 'dim_min': 1},
        'keyY': {'key': keyY, 'ref': None, 'islog': None, 'dim_min': 2},
        'keyZ': {'key': keyZ, 'ref': None, 'islog': None, 'dim_min': 3},
        'keyU': {'key': keyU, 'ref': None, 'islog': None, 'dim_min': 4},
    }

    lk_in = sorted([k0 for k0, v0 in dk.items() if v0['key'] is not None])
    lk_out = sorted([k0 for k0, v0 in dk.items() if v0['key'] is None])
    assert len(lk_in) <= ndim

    # -----------
    # find order

    already = []
    for k0 in lk_in + lk_out:

        dk[k0]['key'], dk[k0]['ref'], dk[k0]['islog'] = _check_keyXYZ(
            coll=coll,
            refs=refs,
            keyX=dk[k0]['key'],
            keyXstr=k0,
            ndim=ndim,
            dim_min=dk[k0]['dim_min'],
            uniform=uniform,
            already=already,
        )

        already.append(dk[k0]['ref'])

    # unicity of refX vs refY
    lk_done = [v0['ref'] for k0, v0 in dk.items() if v0['key'] is not None]
    sameref = len(set(lk_done)) < ndim

    return (
        dk['keyX']['key'], dk['keyX']['ref'], dk['keyX']['islog'],
        dk['keyY']['key'], dk['keyY']['ref'], dk['keyY']['islog'],
        dk['keyZ']['key'], dk['keyZ']['ref'], dk['keyZ']['islog'],
        dk['keyU']['key'], dk['keyU']['ref'], dk['keyU']['islog'],
        sameref,
    )

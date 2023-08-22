# coding utf-8


# Common
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors


# library-specific
from . import _generic_check
from . import _plot_text
from . import _class1_compute


__all__ = ['plot_as_array']


__github = 'https://github.com/ToFuProject/datacollection/issues'
_WINTIT = f'report issues at {__github}'


_CONNECT = True
_BCKCOLOR = 'w'

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


def plot_as_array(
    # parameters
    coll=None,
    key=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    keyU=None,
    ind=None,
    vmin=None,
    vmax=None,
    cmap=None,
    ymin=None,
    ymax=None,
    aspect=None,
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
    dcolorbar=None,
    dleg=None,
    label=None,
    connect=None,
    inplace=None,
):


    # ------------
    #  check inputs

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
        dcolorbar, dleg, label, connect,
    ) = _plot_as_array_check(
        ndim=ndim,
        coll=coll2,
        key=key,
        keyX=keyX,
        keyY=keyY,
        keyZ=keyZ,
        keyU=keyU,
        ind=ind,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ymin=ymin,
        ymax=ymax,
        aspect=aspect,
        nmax=nmax,
        uniform=uniform,
        color_dict=color_dict,
        rotation=rotation,
        inverty=inverty,
        bck=bck,
        interp=interp,
        # figure
        dcolorbar=dcolorbar,
        dleg=dleg,
        label=label,
        connect=connect,
    )

    # --------------------------------
    # Particular case: same references

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

    if ndim == 1:
        coll2, dax, dgroup = plot_as_array_1d(
            # parameters
            coll=coll2,
            key=key,
            keyX=keyX,
            refX=refX,
            islogX=islogX,
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
            # figure-specific
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
        )

    elif ndim == 2:
        coll2, dax, dgroup = plot_as_array_2d(
            # parameters
            coll=coll2,
            key=key,
            keyX=keyX,
            keyY=keyY,
            refX=refX,
            refY=refY,
            islogX=islogX,
            islogY=islogY,
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
            interp=interp,
            # figure-specific
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
        )

    elif ndim == 3:
        coll2, dax, dgroup = plot_as_array_3d(
            # parameters
            coll=coll2,
            key=key,
            keyX=keyX,
            keyY=keyY,
            keyZ=keyZ,
            refX=refX,
            refY=refY,
            refZ=refZ,
            islogX=islogX,
            islogY=islogY,
            islogZ=islogZ,
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

    elif ndim == 4:
        coll2, dax, dgroup = plot_as_array_4d(
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

    # --------------------------
    # add axes and interactivity

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

    # connect
    if connect is True:
        coll2.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)
        coll2.disconnect_old()
        coll2.connect()

        coll2.show_commands(verb=show_commands)
        return coll2
    else:
        return coll2, dgroup


# #############################################################################
# #############################################################################
#                       check
# #############################################################################


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
                elif all([kk in already for kk in refs]): # TBC
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
            elif all([kk in already for kk in refs]): # TBC
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


def _plot_as_array_check(
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
    dcolorbar=None,
    dleg=None,
    data=None,
    label=None,
    connect=None,
):


    # --------
    # groups

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

    # ----
    # cmap

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
        k0: _LCOLOR_DICT[0] for ii, k0 in enumerate(groups)
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


def _get_str_datadlab(keyX=None, nx=None, islogX=None, coll=None):

    keyX2 = keyX
    xstr = keyX != 'index' and coll.ddata[keyX]['data'].dtype.type == np.str_
    if keyX == 'index':
        dataX = np.arange(0, nx)
        labX = keyX
        dX2 = 0.5
    elif xstr:
        dataX = np.arange(0, nx)
        labX = ''
        dX2 = 0.5
    else:
        if islogX is True:
            keyX2 = f"{keyX}-log10"
            coll.add_data(
                key=keyX2,
                data=np.log10(coll.ddata[keyX]['data']),
                ref=coll.ddata[keyX]['ref'],
            )
            labX = r"$\log_{10}$" + f"({keyX} ({coll._ddata[keyX]['units']}))"
            dataX = coll.ddata[keyX2]['data']
        else:
            labX = f"{keyX} ({coll._ddata[keyX]['units']})"
            dataX = coll.ddata[keyX]['data']
        dX2 = np.nanmean(np.diff(dataX)) / 2.

    return keyX2, xstr, dataX, dX2, labX


# #############################################################################
# #############################################################################
#                       plot_as_array: 1d
# #############################################################################


def plot_as_array_1d(
    # parameters
    coll=None,
    key=None,
    keyX=None,
    refX=None,
    islogX=None,
    ind=None,
    vmin=None,
    vmax=None,
    cmap=None,
    ymin=None,
    ymax=None,
    aspect=None,
    nmax=None,
    color_dict=None,
    lkeys=None,
    bstr_dict=None,
    rotation=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
):

    # --------------
    #  Prepare data

    data = coll.ddata[key]['data']
    if hasattr(data, 'nnz'):
        data = data.toarray()
    assert data.ndim == len(coll.ddata[key]['ref']) == 1
    n0, = data.shape

    keyX, xstr, dataX, dX2, labX = _get_str_datadlab(
        keyX=keyX, nx=n0, islogX=islogX, coll=coll,
    )
    ref = coll._ddata[key]['ref'][0]
    units = coll._ddata[key]['units']
    lab0 = f'ind ({ref})'
    lab1 = f'{key} ({units})'

    # --------------
    # plot - prepare

    if dax is None:

        if fs is None:
            fs = (12, 8)

        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.95,
                'bottom': 0.10, 'top': 0.90,
                'hspace': 0.15, 'wspace': 0.2,
            }

        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(ncols=4, nrows=1, **dmargin)

        ax0 = fig.add_subplot(gs[0, :3], aspect='auto')
        ax0.set_ylabel(lab1)
        ax0.set_title(key, size=14, fontweight='bold')
        if xstr:
            ax0.set_xticks(dataX)
            ax0.set_xticklabels(
                coll.ddata[keyX]['data'],
                rotation=rotation,
            )
        else:
            ax0.set_xlabel(lab0)

        ax1 = fig.add_subplot(gs[0, 3], frameon=False)
        ax1.set_xticks([])
        ax1.set_yticks([])

        dax = {
            'matrix': {'handle': ax0},
            'text': {'handle': ax1},
        }

    dax = _generic_check._check_dax(dax=dax, main='matrix')

    # ---------------
    # plot fixed part

    axtype = 'matrix'
    lkax = [kk for kk, vv in dax.items() if axtype in vv['type']]
    for kax in lkax:
        ax = dax[kax]['handle']

        ax.plot(
            dataX,
            data,
            color='k',
            marker='.',
            ms=6,
        )

        # plt.colorbar(im, ax=ax, **dcolorbar)
        if dleg is not False:
            ax.legend(**dleg)

    # ----------------
    # define and set dgroup

    dgroup = {
        'X': {
            'ref': [ref],
            'data': ['index'],
            'nmax': nmax,
        },
    }

    # ----------------
    # plot mobile part

    axtype = 'matrix'
    lkax = [kk for kk, vv in dax.items() if axtype in vv['type']]
    for kax in lkax:
        ax = dax[kax]['handle']

        # ind0, ind1
        for ii in range(nmax):
            lv = ax.axvline(ind[0], c=color_dict['X'][ii], lw=1., ls='-')

            # update coll
            kv = f'{key}_v{ii:02.0f}'
            coll.add_mobile(
                key=kv,
                handle=lv,
                refs=ref,
                data=keyX,
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[ref], datax=[keyX])

    # ---------
    # add text

    kax = 'text'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            key=key,
            ax=ax,
            ref=ref,
            group='X',
            ind=ind[0],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    return coll, dax, dgroup


# #############################################################################
# #############################################################################
#                       plot_as_array: 2d
# #############################################################################


def plot_as_array_2d(
    # parameters
    coll=None,
    key=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    refX=None,
    refY=None,
    refZ=None,
    islogX=None,
    islogY=None,
    islogZ=None,
    ind=None,
    vmin=None,
    vmax=None,
    cmap=None,
    ymin=None,
    ymax=None,
    aspect=None,
    nmax=None,
    color_dict=None,
    lkeys=None,
    bstr_dict=None,
    rotation=None,
    inverty=None,
    interp=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    interactive=None,
):

    # --------------
    #  Prepare data

    data = coll.ddata[key]['data']
    refs = coll.ddata[key]['ref']
    if hasattr(data, 'nnz'):
        data = data.toarray()
    assert data.ndim == len(coll.ddata[key]['ref']) == 2
    n0, n1 = data.shape

    # check if transpose is necessary
    if refs.index(refX) == 0:
        dataplot = data.T
        nx, ny = n0, n1
        axisX, axisY = 0, 1
    else:
        dataplot = data
        nx, ny = n1, n0
        axisX, axisY = 1, 0

    # -----------------
    #  prepare slicing

    # here slice X => slice in dim Y and vice-versa
    sliX = _class1_compute._get_slice(laxis=[1-axisX], ndim=2)
    sliY = _class1_compute._get_slice(laxis=[1-axisY], ndim=2)

    # ----------------------
    #  labels and data

    keyX, xstr, dataX, dX2, labX = _get_str_datadlab(
        keyX=keyX, nx=nx, islogX=islogX, coll=coll,
    )
    keyY, ystr, dataY, dY2, labY = _get_str_datadlab(
        keyX=keyY, nx=ny, islogX=islogY, coll=coll,
    )

    extent = (
        dataX[0] - dX2, dataX[-1] + dX2,
        dataY[0] - dY2, dataY[-1] + dY2,
    )

    # --------------
    # plot - prepare

    if dax is None:

        if fs is None:
            fs = (14, 8)

        if dmargin is None:
            dmargin = {
                'left': 0.05, 'right': 0.95,
                'bottom': 0.06, 'top': 0.90,
                'hspace': 0.2, 'wspace': 0.3,
            }

        fig = plt.figure(figsize=fs)
        fig.suptitle(key, size=14, fontweight='bold')
        gs = gridspec.GridSpec(ncols=4, nrows=6, **dmargin)

        # axes for image
        ax0 = fig.add_subplot(gs[:4, :2], aspect='auto')
        ax0.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax0.xaxis.set_label_position('top')

        # axes for vertical profile
        ax1 = fig.add_subplot(gs[:4, 2], sharey=ax0)
        ax1.set_xlabel('data')
        ax1.set_ylabel(labY)
        ax1.tick_params(
            axis="y",
            left=False, right=True,
            labelleft=False, labelright=True,
        )
        ax1.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax1.yaxis.set_label_position('right')
        ax1.xaxis.set_label_position('top')

        # axes for horizontal profile
        ax2 = fig.add_subplot(gs[4:, :2], sharex=ax0)
        ax2.set_ylabel('data')
        ax2.set_xlabel(labX)

        if np.isfinite(ymin):
            ax1.set_xlim(left=ymin)
            ax2.set_ylim(bottom=ymin)
        if np.isfinite(ymax):
            ax1.set_xlim(right=ymax)
            ax2.set_ylim(top=ymax)


        # axes for text
        ax3 = fig.add_subplot(gs[:3, 3], frameon=False)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4 = fig.add_subplot(gs[3:, 3], frameon=False)
        ax4.set_xticks([])
        ax4.set_yticks([])

        if xstr:
            ax0.set_xticks(dataX)
            ax0.set_xticklabels(
                coll.ddata[keyX]['data'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
            ax2.set_xticks(dataX)
            ax2.set_xticklabels(
                coll.ddata[keyX]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax0.set_xlabel(labX)
            ax2.set_xlabel(labX)

        if ystr:
            ax0.set_yticks(dataY)
            ax0.set_yticklabels(
                coll.ddata[keyY]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
            ax1.set_yticks(dataY)
            ax1.set_yticklabels(
                coll.ddata[keyY]['data'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
        else:
            ax0.set_ylabel(labY)
            ax1.set_ylabel(labY)

        dax = {
            # data
            'matrix': {'handle': ax0, 'inverty': inverty},
            'vertical': {'handle': ax1, 'inverty': inverty},
            'horizontal': {'handle': ax2},
            # text
            'text0': {'handle': ax3},
            'text1': {'handle': ax4},
        }

    dax = _generic_check._check_dax(dax=dax, main='matrix')

    # ---------------
    # plot fixed part

    axtype = 'matrix'
    kax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(kax) == 1:
        kax = kax[0]
        ax = dax[kax]['handle']

        im = ax.imshow(
            dataplot,
            extent=extent,
            interpolation=interp,
            origin='lower',
            aspect=aspect,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if inverty is True:
            ax.invert_yaxis()

    # ----------------
    # define and set dgroup

    dgroup = {
        'X': {
            'ref': [refX],
            'data': ['index'],
            'nmax': nmax,
        },
        'Y': {
            'ref': [refY],
            'data': ['index'],
            'nmax': nmax,
        },
    }

    # ----------------
    # plot mobile part

    axtype = 'matrix'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        # ind0, ind1
        axtype = 'vertical'
        lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
        if len(lax) == 1:
            for ii in range(nmax):
                lh = ax.axhline(
                    dataY[ind[1]], c=color_dict['X'][ii], lw=1., ls='-',
                )

                # update coll
                kh = f'{key}_h{ii:02.0f}'
                coll.add_mobile(
                    key=kh,
                    handle=lh,
                    refs=refY,
                    data=keyY,
                    dtype='ydata',
                    axes=kax,
                    ind=ii,
                )

            # for ax clic
            ax_refx = [refX]
            ax_datax = [keyX]
        else:
            # for ax clic
            ax_refx = None
            ax_datax = None

        # ind0
        axtype = 'horizontal'
        lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
        if len(lax) == 1:
            for ii in range(nmax):
                lv = ax.axvline(
                    dataX[ind[0]], c=color_dict['Y'][ii], lw=1., ls='-',
                )

                # update coll
                kv = f'{key}_v{ii:02.0f}'
                coll.add_mobile(
                    key=kv,
                    handle=lv,
                    refs=refX,
                    data=keyX,
                    dtype='xdata',
                    axes=kax,
                    ind=ii,
                )

            # for ax clic
            ax_refy = [refY]
            ax_datay = [keyY]
        else:
            # for ax clic
            ax_refy = None
            ax_datay = None

        dax[kax].update(
            refx=ax_refx,
            datax=ax_datax,
            refy=ax_refy,
            datay=ax_datay,
        )

    axtype = 'vertical'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l0, = ax.plot(
                data[sliY(ind[0])],
                dataY,
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['Y'][ii],
                label=f'ind0 = {ind[0]}',
            )

            km = f'{key}_vprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(refX,),
                data=key,
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

            l0 = ax.axhline(
                dataY[ind[1]],
                c=color_dict['X'][ii],
            )
            km = f'{key}_lh-v{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(refY,),
                data=keyY,
                dtype='ydata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refy=[refY], datay=[keyY])

    axtype = 'horizontal'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l1, = ax.plot(
                dataX,
                data[sliX(ind[1])],
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['X'][ii],
                label=f'ind1 = {ind[1]}',
            )

            km = f'{key}_hprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l1,
                refs=(refY,),
                data=[key],
                dtype='ydata',
                axes=kax,
                ind=ii,
            )

            l0 = ax.axvline(
                dataX[ind[0]],
                c=color_dict['Y'][ii],
            )
            km = f'{key}_lv-h{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(refX,),
                data=keyX,
                dtype='xdata',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[refX], datax=[keyX])

    # ---------
    # add text

    axtype = 'text0'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            key=key,
            ax=ax,
            ref=refX,
            group='X',
            ind=ind[0],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    axtype = 'text1'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            key=key,
            ax=ax,
            ref=refY,
            group='Y',
            ind=ind[1],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    return coll, dax, dgroup


# #############################################################################
# #############################################################################
#                       plot_as_array: 3d
# #############################################################################


def plot_as_array_3d(
    # parameters
    coll=None,
    key=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    refX=None,
    refY=None,
    refZ=None,
    islogX=None,
    islogY=None,
    islogZ=None,
    ind=None,
    vmin=None,
    vmax=None,
    cmap=None,
    ymin=None,
    ymax=None,
    aspect=None,
    nmax=None,
    color_dict=None,
    lkeys=None,
    bstr_dict=None,
    rotation=None,
    inverty=None,
    bck=None,
    interp=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    label=None,
):

    # --------------
    #  Prepare data

    data = coll.ddata[key]['data']
    refs = coll.ddata[key]['ref']
    if hasattr(data, 'nnz'):
        data = data.toarray()
    assert data.ndim == len(coll.ddata[key]['ref']) == 3
    n0, n1, n2 = data.shape

    # check if transpose is necessary
    [axX, axY, axZ] = [refs.index(rr) for rr in [refX, refY, refZ]]
    [nx, ny, nz] = [data.shape[aa] for aa in [axX, axY, axZ]]

    # -----------------
    #  prepare slicing

    # here slice X => slice in dim Y and vice-versa
    sliX = _class1_compute._get_slice(laxis=[axY, axZ], ndim=3)
    sliY = _class1_compute._get_slice(laxis=[axX, axZ], ndim=3)
    sliZ = _class1_compute._get_slice(laxis=[axX, axY], ndim=3)
    sliZ2 = _class1_compute._get_slice(laxis=[axZ], ndim=3)

    if axX < axY:
        datatype = 'data.T'
        dataplot = data[sliZ2(ind[2])].T
    else:
        datatype = 'data'
        dataplot = data[sliZ2(ind[2])]

    # ----------------------
    #  labels and data

    keyX, xstr, dataX, dX2, labX = _get_str_datadlab(
        keyX=keyX, nx=nx, islogX=islogX, coll=coll,
    )
    keyY, ystr, dataY, dY2, labY = _get_str_datadlab(
        keyX=keyY, nx=ny, islogX=islogY, coll=coll,
    )

    keyZ, zstr, dataZ, dZ2, labZ = _get_str_datadlab(
        keyX=keyZ, nx=nz, islogX=islogZ, coll=coll,
    )

    extent = (
        dataX[0] - dX2, dataX[-1] + dX2,
        dataY[0] - dY2, dataY[-1] + dY2,
    )

    # --------------
    # plot - prepare

    if dax is None:
        dax = _plot_as_array_3d_create_axes(
            fs=fs,
            dmargin=dmargin,
        )

    dax = _generic_check._check_dax(dax=dax, main='matrix')

    if label:
        _plot_as_array_3d_label_axes(
            coll=coll,
            dax=dax,
            key=key,
            labX=labX,
            labY=labY,
            labZ=labZ,
            ymin=ymin,
            ymax=ymax,
            xstr=xstr,
            ystr=ystr,
            zstr=zstr,
            keyX=keyX,
            keyY=keyY,
            keyZ=keyZ,
            dataX=dataX,
            dataY=dataY,
            dataZ=dataZ,
            inverty=inverty,
            rotation=rotation,
        )

    # ---------------
    # plot fixed part

    axtype = 'tracesZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        if bck == 'lines':
            shap = list(data.shape)
            shap[axZ] = 1
            bckl = np.concatenate((data, np.full(shap, np.nan)), axis=axZ)
            bckl = np.swapaxes(bckl, axZ, -1).ravel()
            zdat = np.tile(np.r_[dataZ, np.nan], nx*ny)
            ax.plot(
                zdat,
                bckl,
                c=(0.8, 0.8, 0.8),
                ls='-',
                lw=1.,
                marker='None',
            )
        else:
            bckenv = [
                np.nanmin(data, axis=(axX, axY)),
                np.nanmax(data, axis=(axX, axY)),
            ]
            zdat = dataZ
            ax.fill_between(
                zdat,
                bckenv[0],
                bckenv[1],
                facecolor=(0.8, 0.8, 0.8, 0.8),
                edgecolor='None',
            )

    # ----------------
    # define and set dgroup

    dgroup = {
        'X': {
            'ref': [refX],
            'data': ['index'],
            'nmax': nmax,
        },
        'Y': {
            'ref': [refY],
            'data': ['index'],
            'nmax': nmax,
        },
        'Z': {
            'ref': [refZ],
            'data': ['index'],
            'nmax': 1,
        },
    }

    # ----------------
    # plot mobile part

    axtype = 'matrix'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        # image
        im = ax.imshow(
            dataplot,
            extent=extent,
            interpolation=interp,
            origin='lower',
            aspect=aspect,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        km = f'{key}_im'
        coll.add_mobile(
            key=km,
            handle=im,
            refs=refZ,
            data=key,
            dtype=datatype,
            axes=kax,
            ind=0,
            harmonize=False,
        )

        if inverty is True:
            ax.invert_yaxis()

        # ind0, ind1
        for ii in range(nmax):
            lh = ax.axhline(
                dataY[ind[1]], c=color_dict['X'][ii], lw=1., ls='-',
            )
            lv = ax.axvline(
                dataX[ind[0]], c=color_dict['Y'][ii], lw=1., ls='-',
            )
            mi, = ax.plot(
                dataX[ind[0]],
                dataY[ind[1]],
                marker='s',
                ms=6,
                markeredgecolor=color_dict['X'][ii],
                markerfacecolor='None',
            )

            # update coll
            kh = f'{key}_h{ii:02.0f}'
            kv = f'{key}_v{ii:02.0f}'
            coll.add_mobile(
                key=kh,
                handle=lh,
                refs=refY,
                data=keyY,
                dtype='ydata',
                axes=kax,
                ind=ii,
                harmonize=False,
            )
            coll.add_mobile(
                key=kv,
                handle=lv,
                refs=refX,
                data=keyX,
                dtype='xdata',
                axes=kax,
                ind=ii,
                harmonize=False,
            )
            km = f'{key}_m{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=mi,
                refs=[refX, refY],
                data=[keyX, keyY],
                dtype=['xdata', 'ydata'],
                axes=kax,
                ind=ii,
                harmonize=False,
            )

        dax[kax].update(
            refx=[refX],
            refy=[refY],
            datax=[keyX],
            datay=[keyY],
        )

    axtype = 'vertical'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l0, = ax.plot(
                data[sliY(ind[0], ind[2])],
                dataY,
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['Y'][ii],
                label=f'ind0 = {ind[0]}',
            )

            km = f'{key}_vprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=((refX, refZ),),
                data=[key],
                dtype=['xdata'],
                group_vis='X',
                axes=kax,
                ind=ii,
                harmonize=False,
            )

            l0 = ax.axhline(
                dataY[ind[1]],
                c=color_dict['X'][ii],
            )
            km = f'{key}_lh-v{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(refY,),
                data=keyY,
                dtype='ydata',
                group_vis='Y',
                axes=kax,
                ind=ii,
                harmonize=False,
            )

        dax[kax].update(refy=[refY], datay=[keyY])

    axtype = 'horizontal'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l1, = ax.plot(
                dataX,
                data[sliX(ind[1], ind[2])],
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['X'][ii],
            )

            km = f'{key}_hprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l1,
                refs=((refY, refZ),),
                data=[key],
                dtype=['ydata'],
                group_vis='Y',
                axes=kax,
                ind=ii,
                harmonize=False,
            )

            l0 = ax.axvline(
                dataX[ind[0]],
                c=color_dict['Y'][ii],
            )
            km = f'{key}_lv-h{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(refX,),
                data=keyX,
                dtype='xdata',
                group_vis='X',
                axes=kax,
                ind=ii,
                harmonize=False,
            )

        dax[kax].update(refx=[refX], datax=[keyX])

    # traces
    axtype = 'tracesZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l1, = ax.plot(
                dataZ,
                data[sliZ(ind[0], ind[1])],
                ls='-',
                marker='None',
                color=color_dict['X'][ii],
            )

            km = f'{key}_traceZ{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l1,
                refs=((refX, refY),),
                data=[key],
                dtype=['ydata'],
                axes=kax,
                ind=ii,
                harmonize=False,
            )

        l0 = ax.axvline(
            dataZ[ind[2]],
            c='k',
        )
        km = f'{key}_lv-z'
        coll.add_mobile(
            key=km,
            handle=l0,
            refs=(refZ,),
            data=keyZ,
            dtype='xdata',
            axes=kax,
            ind=0,
            harmonize=False,
        )

        dax[kax].update(refx=[refZ], datax=[keyZ])

    # ---------
    # add text

    axtype = 'textX'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            key=key,
            ax=ax,
            ref=refX,
            group='X',
            ind=ind[0],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    axtype = 'textY'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            key=key,
            ax=ax,
            ref=refY,
            group='Y',
            ind=ind[1],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    axtype = 'textZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            key=key,
            ax=ax,
            ref=refZ,
            group='Z',
            ind=ind[2],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    return coll, dax, dgroup


def _plot_as_array_3d_create_axes(
    fs=None,
    dmargin=None,
):

    if fs is None:
        fs = (15, 9)

    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.95,
            'bottom': 0.06, 'top': 0.90,
            'hspace': 0.2, 'wspace': 0.3,
        }

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(ncols=6, nrows=6, **dmargin)

    # axes for image
    ax0 = fig.add_subplot(gs[:4, 2:4], aspect='auto')

    # axes for vertical profile
    ax1 = fig.add_subplot(gs[:4, 4], sharey=ax0)

    # axes for horizontal profile
    ax2 = fig.add_subplot(gs[4:, 2:4], sharex=ax0)

    # axes for traces
    ax3 = fig.add_subplot(gs[:3, :2])

    # axes for text
    ax4 = fig.add_subplot(gs[:3, 5], frameon=False)
    ax5 = fig.add_subplot(gs[3:, 5], frameon=False)
    ax6 = fig.add_subplot(gs[4:, :2], frameon=False)

    # dax
    dax = {
        # data
        'matrix': {'handle': ax0},
        'vertical': {'handle': ax1},
        'horizontal': {'handle': ax2},
        'tracesZ': {'handle': ax3},
        # text
        'textX': {'handle': ax4},
        'textY': {'handle': ax5},
        'textZ': {'handle': ax6},
    }
    return dax


def _plot_as_array_3d_label_axes(
    coll=None,
    dax=None,
    key=None,
    labX=None,
    labY=None,
    labZ=None,
    ymin=None,
    ymax=None,
    xstr=None,
    ystr=None,
    zstr=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    dataX=None,
    dataY=None,
    dataZ=None,
    inverty=None,
    rotation=None,
):

    # fig
    fig = list(dax.values())[0]['handle'].figure
    fig.suptitle(key, size=14, fontweight='bold')

    # axes for image
    axtype = 'matrix'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax.xaxis.set_label_position('top')

        # x text ticks
        if xstr:
            ax.set_xticks(dataX)
            ax.set_xticklabels(
                coll.ddata[keyX]['data'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
        else:
            ax.set_xlabel(labX)

        # y text ticks
        if ystr:
            ax.set_yticks(dataY)
            ax.set_yticklabels(
                coll.ddata[keyY]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax.set_ylabel(labY)

        dax[kax]['inverty'] = inverty

    # axes for vertical profile
    axtype = 'vertical'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xlabel('data')
        ax.set_ylabel(labY)
        ax.tick_params(
            axis="y",
            left=False, right=True,
            labelleft=False, labelright=True,
        )
        ax.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax.yaxis.set_label_position('right')
        ax.xaxis.set_label_position('top')

        if np.isfinite(ymin):
            ax.set_xlim(left=ymin)
        if np.isfinite(ymax):
            ax.set_xlim(right=ymax)

        # y text ticks
        if ystr:
            ax.set_yticks(dataY)
            ax.set_yticklabels(
                coll.ddata[keyY]['data'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
        else:
            ax.set_ylabel(labY)

        dax[kax]['inverty'] = inverty

    # axes for horizontal profile
    axtype = 'horizontal'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_ylabel('data')
        ax.set_xlabel(labX)

        if np.isfinite(ymin):
            ax.set_ylim(bottom=ymin)
        if np.isfinite(ymax):
            ax.set_ylim(top=ymax)

        # x text ticks
        if xstr:
            ax.set_xticks(dataX)
            ax.set_xticklabels(
                coll.ddata[keyX]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax.set_xlabel(labX)

    # axes for traces
    axtype = 'tracesZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_ylabel('data')
        ax.set_xlabel(labZ)

        if np.isfinite(ymin):
            ax.set_ylim(bottom=ymin)
        if np.isfinite(ymax):
            ax.set_ylim(top=ymax)

        # z text ticks
        if zstr:
            ax.set_yticks(dataZ)
            ax.set_yticklabels(
                coll.ddata[keyZ]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax.set_ylabel(labZ)

    # axes for text
    axtype = 'textX'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xticks([])
        ax.set_yticks([])

    axtype = 'textY'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xticks([])
        ax.set_yticks([])

    axtype = 'textZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xticks([])
        ax.set_yticks([])

    return dax


# #############################################################################
# #############################################################################
#                       plot_as_array: 4d
# #############################################################################


def plot_as_array_4d(
    # parameters
    coll=None,
    key=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    keyU=None,
    refX=None,
    refY=None,
    refZ=None,
    refU=None,
    islogX=None,
    islogY=None,
    islogZ=None,
    islogU=None,
    ind=None,
    vmin=None,
    vmax=None,
    cmap=None,
    ymin=None,
    ymax=None,
    aspect=None,
    nmax=None,
    color_dict=None,
    lkeys=None,
    bstr_dict=None,
    rotation=None,
    inverty=None,
    bck=None,
    interp=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    label=None,
):

    # --------------
    #  Prepare data

    data = coll.ddata[key]['data']
    refs = coll.ddata[key]['ref']
    if hasattr(data, 'nnz'):
        data = data.toarray()
    assert data.ndim == len(coll.ddata[key]['ref']) == 4
    n0, n1, n2, n3 = data.shape

    # check if transpose is necessary
    [axX, axY, axZ, axU] = [refs.index(rr) for rr in [refX, refY, refZ, refU]]
    [nx, ny, nz, nu] = [data.shape[aa] for aa in [axX, axY, axZ, axU]]

    # -----------------
    #  prepare slicing

    # here slice X => slice in dim Y and vice-versa
    sliX = _class1_compute._get_slice(laxis=[axY, axZ, axU], ndim=4)
    sliY = _class1_compute._get_slice(laxis=[axX, axZ, axU], ndim=4)
    sliZ = _class1_compute._get_slice(laxis=[axX, axY, axU], ndim=4)
    sliU = _class1_compute._get_slice(laxis=[axX, axY, axZ], ndim=4)
    sliZ2 = _class1_compute._get_slice(laxis=[axZ, axU], ndim=4)

    if axX < axY:
        datatype = 'data.T'
        dataplot = data[sliZ2(ind[2], ind[3])].T
    else:
        datatype = 'data'
        dataplot = data[sliZ2(ind[2], ind[3])]

    # ----------------------
    #  labels and data

    keyX, xstr, dataX, dX2, labX = _get_str_datadlab(
        keyX=keyX, nx=nx, islogX=islogX, coll=coll,
    )
    keyY, ystr, dataY, dY2, labY = _get_str_datadlab(
        keyX=keyY, nx=ny, islogX=islogY, coll=coll,
    )
    keyZ, zstr, dataZ, dZ2, labZ = _get_str_datadlab(
        keyX=keyZ, nx=nz, islogX=islogZ, coll=coll,
    )
    keyU, ustr, dataU, dU2, labU = _get_str_datadlab(
        keyX=keyU, nx=nu, islogX=islogU, coll=coll,
    )

    extent = (
        dataX[0] - dX2, dataX[-1] + dX2,
        dataY[0] - dY2, dataY[-1] + dY2,
    )

    # --------------
    # plot - prepare

    if dax is None:
        dax = _plot_as_array_4d_create_axes(
            fs=fs,
            dmargin=dmargin,
        )

    dax = _generic_check._check_dax(dax=dax, main='matrix')

    if label:
        _plot_as_array_4d_label_axes(
            coll=coll,
            dax=dax,
            key=key,
            labX=labX,
            labY=labY,
            labZ=labZ,
            labU=labU,
            ymin=ymin,
            ymax=ymax,
            xstr=xstr,
            ystr=ystr,
            zstr=zstr,
            ustr=ustr,
            keyX=keyX,
            keyY=keyY,
            keyZ=keyZ,
            keyU=keyU,
            dataX=dataX,
            dataY=dataY,
            dataZ=dataZ,
            dataU=dataU,
            inverty=inverty,
            rotation=rotation,
        )

    # ---------------
    # plot fixed part

    # tracesZ
    axtype = 'tracesZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        if bck == 'lines':
            shap = list(data.shape)
            shap[axZ] = 1
            bckl = np.concatenate((data, np.full(shap, np.nan)), axis=axZ)
            bckl = np.swapaxes(bckl, axZ, -1).ravel()
            zdat = np.tile(np.r_[dataZ, np.nan], nx*ny*nu)
            ax.plot(
                zdat,
                bckl,
                c=(0.8, 0.8, 0.8),
                ls='-',
                lw=1.,
                marker='None',
            )
        else:
            bckenv = [
                np.nanmin(data, axis=(axX, axY, axU)),
                np.nanmax(data, axis=(axX, axY, axU)),
            ]
            zdat = dataZ
            ax.fill_between(
                zdat,
                bckenv[0],
                bckenv[1],
                facecolor=(0.8, 0.8, 0.8, 0.8),
                edgecolor='None',
            )

    # tracesU
    axtype = 'tracesU'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        if bck == 'lines':
            shap = list(data.shape)
            shap[axU] = 1
            bckl = np.concatenate((data, np.full(shap, np.nan)), axis=axU)
            bckl = np.swapaxes(bckl, axU, -1).ravel()
            udat = np.tile(np.r_[dataU, np.nan], nx*ny*nz)
            ax.plot(
                udat,
                bckl,
                c=(0.8, 0.8, 0.8),
                ls='-',
                lw=1.,
                marker='None',
            )
        else:
            bckenv = [
                np.nanmin(data, axis=(axX, axY, axZ)),
                np.nanmax(data, axis=(axX, axY, axZ)),
            ]
            udat = dataU
            ax.fill_between(
                udat,
                bckenv[0],
                bckenv[1],
                facecolor=(0.8, 0.8, 0.8, 0.8),
                edgecolor='None',
            )

    # ----------------
    # define and set dgroup

    dgroup = {
        'X': {
            'ref': [refX],
            'data': ['index'],
            'nmax': nmax,
        },
        'Y': {
            'ref': [refY],
            'data': ['index'],
            'nmax': nmax,
        },
        'Z': {
            'ref': [refZ],
            'data': ['index'],
            'nmax': 1,
        },
        'U': {
            'ref': [refU],
            'data': ['index'],
            'nmax': 1,
        },
    }

    # -----------------
    # plot mobile parts

    # matrix
    axtype = 'matrix'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        # image
        im = ax.imshow(
            dataplot,
            extent=extent,
            interpolation=interp,
            origin='lower',
            aspect=aspect,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        km = f'{key}_im'
        coll.add_mobile(
            key=km,
            handle=im,
            refs=((refZ, refU),),
            data=key,
            dtype=datatype,
            axes=kax,
            ind=0,
        )

        if inverty is True:
            ax.invert_yaxis()

        # ind0, ind1
        for ii in range(nmax):
            lh = ax.axhline(
                dataY[ind[1]], c=color_dict['X'][ii], lw=1., ls='-',
            )
            lv = ax.axvline(
                dataX[ind[0]], c=color_dict['Y'][ii], lw=1., ls='-',
            )
            mi, = ax.plot(
                dataX[ind[0]],
                dataY[ind[1]],
                marker='s',
                ms=6,
                markeredgecolor=color_dict['X'][ii],
                markerfacecolor='None',
            )

            # update coll
            kh = f'{key}_h{ii:02.0f}'
            kv = f'{key}_v{ii:02.0f}'
            coll.add_mobile(
                key=kh,
                handle=lh,
                refs=refY,
                data=keyY,
                dtype='ydata',
                axes=kax,
                ind=ii,
            )
            coll.add_mobile(
                key=kv,
                handle=lv,
                refs=refX,
                data=keyX,
                dtype='xdata',
                axes=kax,
                ind=ii,
            )
            km = f'{key}_m{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=mi,
                refs=[refX, refY],
                data=[keyX, keyY],
                dtype=['xdata', 'ydata'],
                axes=kax,
                ind=ii,
            )

        dax[kax].update(
            refx=[refX],
            refy=[refY],
            datax=[keyX],
            datay=[keyY],
        )

    # vertical
    axtype = 'vertical'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l0, = ax.plot(
                data[sliY(ind[0], ind[2], ind[3])],
                dataY,
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['Y'][ii],
                label=f'ind0 = {ind[0]}',
            )

            km = f'{key}_vprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=((refX, refZ, refU),),
                data=[key],
                dtype=['xdata'],
                group_vis='X',
                axes=kax,
                ind=ii,
            )

            l0 = ax.axhline(
                dataY[ind[1]],
                c=color_dict['X'][ii],
            )
            km = f'{key}_lh-v{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(refY,),
                data=keyY,
                dtype='ydata',
                group_vis='Y',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refy=[refY], datay=[keyY])

    # horizontal
    axtype = 'horizontal'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l1, = ax.plot(
                dataX,
                data[sliX(ind[1], ind[2], ind[3])],
                ls='-',
                marker='.',
                lw=1.,
                color=color_dict['X'][ii],
            )

            km = f'{key}_hprof{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l1,
                refs=((refY, refZ, refU),),
                data=[key],
                dtype=['ydata'],
                group_vis='Y',
                axes=kax,
                ind=ii,
            )

            l0 = ax.axvline(
                dataX[ind[0]],
                c=color_dict['Y'][ii],
            )
            km = f'{key}_lv-h{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l0,
                refs=(refX,),
                data=keyX,
                dtype='xdata',
                group_vis='X',
                axes=kax,
                ind=ii,
            )

        dax[kax].update(refx=[refX], datax=[keyX])

    # tracesZ
    axtype = 'tracesZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l1, = ax.plot(
                dataZ,
                data[sliZ(ind[0], ind[1], ind[3])],
                ls='-',
                marker='None',
                color=color_dict['X'][ii],
            )

            km = f'{key}_trace{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l1,
                refs=((refX, refY, refU),),
                data=[key],
                dtype=['ydata'],
                axes=kax,
                ind=ii,
            )

        l0 = ax.axvline(
            dataZ[ind[2]],
            c='k',
        )
        km = f'{key}_lv-z'
        coll.add_mobile(
            key=km,
            handle=l0,
            refs=(refZ,),
            data=keyZ,
            dtype='xdata',
            axes=kax,
            ind=0,
        )

        dax[kax].update(refx=[refZ], datax=[keyZ])

    # tracesU
    axtype = 'tracesU'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        for ii in range(nmax):
            l1, = ax.plot(
                dataU,
                data[sliU(ind[0], ind[1], ind[2])],
                ls='-',
                marker='None',
                color=color_dict['U'][ii],
            )

            km = f'{key}_traceU{ii:02.0f}'
            coll.add_mobile(
                key=km,
                handle=l1,
                refs=((refX, refY, refZ),),
                data=[key],
                dtype=['ydata'],
                axes=kax,
                ind=ii,
            )

        l0 = ax.axvline(
            dataU[ind[2]],
            c='k',
        )
        km = f'{key}_lv-u'
        coll.add_mobile(
            key=km,
            handle=l0,
            refs=(refU,),
            data=keyU,
            dtype='xdata',
            axes=kax,
            ind=0,
        )

        dax[kax].update(refx=[refU], datax=[keyU])

    # ---------
    # add text

    # textX
    axtype = 'textX'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            key=key,
            ax=ax,
            ref=refX,
            group='X',
            ind=ind[0],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    # textY
    axtype = 'textY'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            key=key,
            ax=ax,
            ref=refY,
            group='Y',
            ind=ind[1],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    # textZ
    axtype = 'textZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            key=key,
            ax=ax,
            ref=refZ,
            group='Z',
            ind=ind[2],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    # textU
    axtype = 'textU'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']

        _plot_text.plot_text(
            coll=coll,
            kax=kax,
            key=key,
            ax=ax,
            ref=refU,
            group='U',
            ind=ind[3],
            lkeys=lkeys,
            nmax=nmax,
            color_dict=color_dict,
            bstr_dict=bstr_dict,
        )

    return coll, dax, dgroup


def _plot_as_array_4d_create_axes(
    fs=None,
    dmargin=None,
):

    if fs is None:
        fs = (17, 9)

    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.95,
            'bottom': 0.06, 'top': 0.90,
            'hspace': 0.5, 'wspace': 0.4,
        }

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(ncols=7, nrows=6, **dmargin)

    # axes for image
    ax0 = fig.add_subplot(gs[:4, 2:4], aspect='auto')

    # axes for vertical profile
    ax1 = fig.add_subplot(gs[:4, 4], sharey=ax0)

    # axes for horizontal profile
    ax2 = fig.add_subplot(gs[4:, 2:4], sharex=ax0)

    # axes for tracesZ
    ax3 = fig.add_subplot(gs[:3, :2])

    # axes for tracesU
    ax4 = fig.add_subplot(gs[3:, :2])

    # axes for text
    ax5 = fig.add_subplot(gs[:3, 5], frameon=False)
    ax6 = fig.add_subplot(gs[3:, 5], frameon=False)
    ax7 = fig.add_subplot(gs[:3, 6], frameon=False)
    ax8 = fig.add_subplot(gs[3:, 6], frameon=False)

    # dax
    dax = {
        # data
        'matrix': {'handle': ax0},
        'vertical': {'handle': ax1},
        'horizontal': {'handle': ax2},
        'tracesZ': {'handle': ax3},
        'tracesU': {'handle': ax4},
        # text
        'textX': {'handle': ax5},
        'textY': {'handle': ax6},
        'textZ': {'handle': ax7},
        'textU': {'handle': ax8},
    }
    return dax


def _plot_as_array_4d_label_axes(
    coll=None,
    dax=None,
    key=None,
    labX=None,
    labY=None,
    labZ=None,
    labU=None,
    ymin=None,
    ymax=None,
    xstr=None,
    ystr=None,
    zstr=None,
    ustr=None,
    keyX=None,
    keyY=None,
    keyZ=None,
    keyU=None,
    dataX=None,
    dataY=None,
    dataZ=None,
    dataU=None,
    inverty=None,
    rotation=None,
):

    # fig
    fig = list(dax.values())[0]['handle'].figure
    fig.suptitle(key, size=14, fontweight='bold')

    # axes for image
    axtype = 'matrix'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax.xaxis.set_label_position('top')

        # x text ticks
        if xstr:
            ax.set_xticks(dataX)
            ax.set_xticklabels(
                coll.ddata[keyX]['data'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
        else:
            ax.set_xlabel(labX)

        # y text ticks
        if ystr:
            ax.set_yticks(dataY)
            ax.set_yticklabels(
                coll.ddata[keyY]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax.set_ylabel(labY)

        dax[kax]['inverty'] = inverty

    # axes for vertical profile
    axtype = 'vertical'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xlabel('data')
        ax.set_ylabel(labY)
        ax.tick_params(
            axis="y",
            left=False, right=True,
            labelleft=False, labelright=True,
        )
        ax.tick_params(
            axis="x",
            bottom=False, top=True,
            labelbottom=False, labeltop=True,
        )
        ax.yaxis.set_label_position('right')
        ax.xaxis.set_label_position('top')

        if np.isfinite(ymin):
            ax.set_xlim(left=ymin)
        if np.isfinite(ymax):
            ax.set_xlim(right=ymax)

        # y text ticks
        if ystr:
            ax.set_yticks(dataY)
            ax.set_yticklabels(
                coll.ddata[keyY]['data'],
                rotation=rotation,
                horizontalalignment='left',
                verticalalignment='bottom',
            )
        else:
            ax.set_ylabel(labY)

        dax[kax]['inverty'] = inverty

    # axes for horizontal profile
    axtype = 'horizontal'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_ylabel('data')
        ax.set_xlabel(labX)

        ax.set_ylim(ymin, ymax)

        # x text ticks
        if xstr:
            ax.set_xticks(dataX)
            ax.set_xticklabels(
                coll.ddata[keyX]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax.set_xlabel(labX)

    # axes for tracesZ
    axtype = 'tracesZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_ylabel('data')
        ax.set_xlabel(labZ)

        ax.set_ylim(ymin, ymax)

        # z text ticks
        if zstr:
            ax.set_yticks(dataZ)
            ax.set_yticklabels(
                coll.ddata[keyZ]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax.set_ylabel(labZ)

    # axes for tracesU
    axtype = 'tracesU'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_ylabel('data')
        ax.set_xlabel(labU)

        ax.set_ylim(ymin, ymax)

        # z text ticks
        if zstr:
            ax.set_yticks(dataU)
            ax.set_yticklabels(
                coll.ddata[keyU]['data'],
                rotation=rotation,
                horizontalalignment='right',
                verticalalignment='top',
            )
        else:
            ax.set_ylabel(labU)

    # axes for text
    axtype = 'textX'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xticks([])
        ax.set_yticks([])

    axtype = 'textY'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xticks([])
        ax.set_yticks([])

    axtype = 'textZ'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xticks([])
        ax.set_yticks([])

    axtype = 'textU'
    lax = [k0 for k0, v0 in dax.items() if axtype in v0['type']]
    if len(lax) == 1:
        kax = lax[0]
        ax = dax[kax]['handle']
        ax.set_xticks([])
        ax.set_yticks([])

    return dax

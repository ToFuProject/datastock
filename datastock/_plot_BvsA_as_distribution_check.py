

import numpy as np
import matplotlib.colors as mcolors


from . import _generic_check


_CONNECT = True
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
#                    check all inputs
# #############################################################################


def _plot_BvsA_check(
    inplace=None,
    # parameters
    coll=None,
    keyA=None,
    keyB=None,
    keyX=None,
    axis=None,
    ind0=None,
    # customization of scatter plot
    dlim=None,
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
    nmax=None,
    dinc=None,
    lkeys=None,
    bstr_dict=None,
    group_color_dict=None,
    # figure-specific
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    aspect=None,
    connect=None,
    groups=None,
):

    # keyA, keyB
    lok = [
        k0 for k0, v0 in coll._ddata.items()
        if v0['data'].ndim <= 2
        and v0['data'].dtype in [int, float, bool]
    ]
    keyA = _generic_check._check_var(
        keyA, 'keyA',
        allowed=lok,
    )
    keyB = _generic_check._check_var(
        keyB, 'keyB',
        allowed=lok,
    )
    refA = coll._ddata[keyA]['ref']
    refB = coll._ddata[keyB]['ref']

    lc = [
        refA == refB
        or refA == tuple([rr for rr in refB if rr in refA])
        or refB == tuple([rr for rr in refA if rr in refB])
    ]
    if not any(lc):
        msg = (
            "keyA and keyB must point to data with references in common!\n"
            f"\t- keyA, refA: {keyA}, {refA}\n"
            f"\t- keyB, refB: {keyB}, {refB}\n"
        )
        raise Exception(msg)
    elif not lc[0]:
        msg = "Different references not implemented yet!"
        raise NotImplementedError(msg)

    # dataA, dataB
    dataA = coll.ddata[keyA]['data']
    dataB = coll.ddata[keyB]['data']

    if hasattr(dataA, 'nnz'):
        dataA = dataA.toarray()
    if hasattr(dataB, 'nnz'):
        dataB = dataB.toarray()

    # check key, inplace flag and extract sub-collection
    keys, inplace, coll2 = _generic_check._check_inplace(
        coll=coll,
        keys=[keyA, keyB],
        inplace=inplace,
    )
    keyA, keyB = keys
    dimA, dimB = dataA.ndim, dataB.ndim
    if dimA > dimB:
        ndim, refs, shape = dimA, refA, dataA.shape
    else:
        ndim, refs, shape = dimB, refB, dataB.shape
    groups = ['ref']

    # keyX vs axis
    if ndim == 1:
        refX = None
        dataX = None
        ref0 = refs[0]

    elif ndim == 2:
        if keyX is None:
            if axis is None:
                axis = 1
            assert axis in [0, 1], axis
            keyX = refs[axis]
            refX = keyX

        else:
            c0 = (
                keyX in refs
                or (
                    keyX in coll._ddata.keys()
                    and (
                        coll._ddata[keyX]['ref'] == refs
                        or (
                            len(coll._ddata[keyX]['ref']) == 1
                            and coll._ddata[keyX]['ref'][0] in refs
                        )
                    )
                )
            )
            if not c0:
                msg = (
                    "Arg keyX must be a valid data key with same ref as keyA\n"
                    f"Provided: {keyX}"
                )
                raise Exception(msg)

            # Deduce refX and axis
            refX = keyX if keyX in refs else coll._ddata[keyX]['ref']
            if isinstance(refX, str):
                axis = refs.index(keyX)
            elif len(refX) == 1:
                refX = refX[0]
                axis = refs.index(refX)
            else:
                if axis is None:
                    axis = 1
                refX = refX[axis]
        assert refX == refs[axis], refs

        # ref0
        ref0 = refs[1-axis]

        # dataX
        if keyX in refs:
            dataX = np.arange(0, coll._dref[keyX]['size'])
        else:
            dataX = coll.ddata[keyX]['data']

        if hasattr(dataX, 'nnz'):
            dataX = dataX.toarray()

    # ind0
    ind0 = _generic_check._check_var(
        ind0, 'ind0',
        default=[0 for ii in range(ndim)],
        types=(list, tuple, np.ndarray),
    )
    c0 = (
        len(ind0) == ndim
        and all([
            np.isscalar(ii) and isinstance(ii, (int, np.integer))
            for ii in ind0
        ])
    )
    if not c0:
        msg = (
            f"Arg ind0 must be an iterable of {ndim} integer indices!\n"
            f"Provided: {ind0}"
        )
        raise Exception(msg)

    # color_dict vs color_cmap
    if color_dict is not None and color_map_key is not None:
        msg = (
            "Please provide either color_dict xor color_cmap!\n"
            f"\t- color_dict: {color_dict}\n"
            f"\t- color_map_key: {color_map_key}\n"
        )
        raise Exception(msg)

    # color_dict
    if color_map_key is None:

        if color_dict is None:
            color_dict = {'k': {'ind': np.ones(shape, dtype=bool)}}

        c0 = (
            isinstance(color_dict, dict)
            and all([
                mcolors.is_color_like(k0)
                and isinstance(v0, dict)
                for k0, v0 in color_dict.items()
            ])
        )
        if not c0:
            msg = (
                "Arg color_dict must be a dict of the form:\n"
                "  {'color0': {'data0': [data0lim0, data0lim1], ...}}\n"
                +  f"Provided:\n{color_dict}"
            )
            raise Exception(msg)

        for k0, v0 in color_dict.items():
            if v0.get('dlim') is None:
                lk1 = [k1 for k1 in v0.keys() if k1 not in ['ind', 'color']]
                if len(lk1) > 0:
                    color_dict[k0]['dlim'] = {k1: v0[k1] for k1 in lk1}
            if v0.get('color') is None:
                color_dict[k0]['color'] = k0

            if v0.get('ind') is None:
                color_dict[k0]['ind'] = _generic_check._apply_dlim(
                    dlim=color_dict[k0]['dlim'],
                    logic_intervals='all',
                    logic=color_dict_logic,
                    ddata=coll._ddata,
                )

            if v0['ind'].shape != shape:
                c0 = (
                    v0['ind'].shape
                    == tuple([aa for aa in shape if aa in v0['ind'].shape])
                )
                if not c0:
                    msg = (
                        f"data used in color_dict['{k0}'] has wrong shape!\n"
                        f"\t- Provided: {v0['ind'].shape} vs {shape}"
                    )
                    raise Exception(msg)

                # reshape
                ind = np.zeros(shape, dtype=bool)
                shap = tuple([
                    aa if aa in v0['ind'].shape else 1
                    for aa in shape
                ])
                ind[...] = v0['ind'].reshape(shap)
                color_dict[k0]['ind'] = ind

    # color_map
    color_map_data = None
    if color_map_key is not None:

        # color_map_key
        c0 = (
            color_map_key in coll._ddata.keys()
            and (
                coll._ddata[color_map_key]['ref']
                == tuple([
                    rr for rr in refs
                    if rr in coll._ddata[color_map_key]['ref']
                ])
            )
        )
        if not c0:
            msg = (
                "Arg color_map_key must be a valid coll.ddata key!\n"
                f"It must have ref = {refs}\n"
                "Provided:\n"
                f"\t- color_map_key: {color_map_key}\n"
                f"\t- ref: {coll._ddata[color_map_key]['ref']}\n"
            )
            raise Exception(msg)

        # color_map_data
        if coll._ddata[color_map_key]['ref'] == refs:
            color_map_data = coll._ddata[color_map_key]['data']
        else:
            color_map_data = np.full(shape, np.nan)
            shap = tuple([
                aa if aa in coll._ddata[color_map_key]['data'].shape else 1
                for aa in shape
            ])
            color_map_data[...] = np.reshape(
                coll._ddata[color_map_key]['data'],
                shap,
            )

        # color map, min, max
        (
            color_map, color_map_vmin, color_map_vmax,
        ) = _generic_check._check_cmap_vminvmax(
            data=coll._ddata[color_map_key]['data'],
            cmap=color_map,
            vmin=color_map_vmin,
            vmax=color_map_vmax,
        )

    # Amin, Amax, Bmin, Bmax
    if Amin is None:
        Amin = np.nanmin(coll._ddata[keyA]['data'])
    if Amax is None:
        Amax = np.nanmax(coll._ddata[keyA]['data'])
    if Bmin is None:
        Bmin = np.nanmin(coll._ddata[keyB]['data'])
    if Bmax is None:
        Bmax = np.nanmax(coll._ddata[keyB]['data'])

    # distribution binning
    if nAbin is None:
        nAbin = 100
    if nBbin is None:
        nBbin = nAbin

    # marker_size
    marker_size = _generic_check._check_var(
        marker_size, 'marker_size',
        default=1,
        types=int,
    )

    # linestyle
    if ndim == 1:
        linestyle = 'None'
    else:
        linestyle = '-'

    # add_bisector
    add_bisector = _generic_check._check_var(
        add_bisector, 'add_bisector',
        default=True,
        types=bool,
    )

    # dist_sample_min
    dist_sample_min = _generic_check._check_var(
        dist_sample_min, 'dist_sample_min',
        default=1,
        types=int,
    )

    # dist_rel
    dist_rel = _generic_check._check_var(
        dist_rel, 'dist_rel',
        default=False,
        types=bool,
    )

    # nmax
    nmax = _generic_check._check_var(
        nmax, 'nmax',
        default=3,
        types=int,
    )

    # group_color_dict
    cdef = {
        k0: _LCOLOR_DICT[0] for ii, k0 in enumerate(groups)
    }
    group_color_dict = _generic_check._check_var(
        group_color_dict, 'group_color_dict',
        default=cdef,
        types=dict,
    )
    dout = {
        k0: str(v0)
        for k0, v0 in group_color_dict.items()
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
            "The following entries of group_color_dict are invalid:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

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

    # aspect
    aspect = _generic_check._check_var(
        aspect, 'aspect',
        default='equal',
        types=str,
        allowed=['equal', 'auto'],
    )

    # connect
    connect = _generic_check._check_var(
        connect, 'connect',
        default=_CONNECT,
        types=bool,
    )

    return (
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
    )

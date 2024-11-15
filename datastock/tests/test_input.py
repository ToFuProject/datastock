

import numpy as np


# ###############################################################
# ###############################################################
#
# ###############################################################


def add_bins(coll):

    # ---------------
    # check if needed

    wbins = coll._which_bins
    if coll.dobj.get(wbins) is not None:
        return

    # -------------------------
    # define bins from scratch
    # -------------------------

    # linear uniform 1d
    coll.add_bins('b1d_lin', edges=np.linspace(0, 1, 10), units='m')

    # log uniform 1d
    coll.add_bins('b1d_log', edges=np.logspace(0, 1, 10), units='eV')

    # non-uniform 1d
    coll.add_bins('b2d_rand', edges=np.r_[1, 2, 5, 10, 12, 20], units='s')

    # linear uniform 2d
    coll.add_bins(
        'b2d_lin',
        edges=(np.linspace(0, 1, 10), np.linspace(0, 3, 20)),
        units='m',
    )

    # log uniform mix 2d
    coll.add_bins(
        'b2d_mix',
        edges=(np.logspace(0, 1, 10), np.pi*np.r_[0, 0.5, 1, 1.2, 1.5, 2]),
        units=('eV', 'rad'),
    )

    # -------------------------
    # define bins pre-existing
    # -------------------------

    return


def binning(coll):
    bins = np.linspace(1, 5, 8)
    lk = [
        ('y', 'nx', bins, 0, False, False, 'y_bin0'),
        ('y', 'nx', bins, 0, True, False, 'y_bin1'),
        ('y', 'nx', 'x', 0, False, True, 'y_bin2'),
        ('y', 'nx', 'x', 0, True, True, 'y_bin3'),
        ('prof0', 'x', 'nt0', 1, False, True, 'p0_bin0'),
        ('prof0', 'x', 'nt0', 1, True, True, 'p0_bin1'),
        ('prof0-bis', 'prof0', 'x', [0, 1], False, True, 'p1_bin0'),
    ]

    for ii, (k0, kr, kb, ax, integ, store, kbin) in enumerate(lk):
        dout = coll.binning(
            data=k0,
            bin_data0=kr,
            bins0=kb,
            axis=ax,
            integrate=integ,
            store=store,
            store_keys=kbin,
            safety_ratio=0.95,
            returnas=True,
        )

        if np.isscalar(ax):
            ax = [ax]

        if isinstance(kb, str):
            if kb in coll.ddata:
                nb = coll.ddata[kb]['data'].size
            else:
                nb = coll.dref[kb]['size']
        else:
            nb = bins.size

        k0 = list(dout.keys())[0]
        shape = [
            ss for ii, ss in enumerate(coll.ddata[k0]['data'].shape)
            if ii not in ax
        ]

        shape.insert(ax[0], nb)
        if dout[k0]['data'].shape != tuple(shape):
            shstr = dout[k0]['data'].shape
            msg = (
                "Mismatching shapes for case {ii}!\n"
                f"\t- dout['{k0}']['data'].shape = {shstr}\n"
                f"\t- expected: {tuple(shape)}"
            )
            raise Exception(msg)

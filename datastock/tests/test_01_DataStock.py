
"""
This module contains tests for tofu.geom in its structured version
"""


# Built-in
import os
import warnings


# Standard
import numpy as np
import matplotlib.pyplot as plt

# datastock-specific
from .._class import DataStock
from .._saveload import load


_PATH_HERE = os.path.dirname(__file__)
_PATH_OUTPUT = os.path.join(_PATH_HERE, 'output')


#######################################################
#
#     Setup and Teardown
#
#######################################################


def clean(path=_PATH_OUTPUT):
    """ Remove all temporary output files that may have been forgotten """
    lf = [ff for ff in os.listdir(path) if ff.endswith('.npz')]
    if len(lf) > 0:
        for ff in lf:
            os.remove(os.path.join(path, ff))


def setup_module(module):
    clean()


def teardown_module(module):
    clean()


#######################################################
#
#     Utilities
#
#######################################################


def _add_ref(st=None, nc=None, nx=None, lnt=None):
    # add references (i.e.: store size of each dimension under a unique key)
    st.add_ref(key='nc', size=nc)
    st.add_ref(key='nx', size=nx)
    for ii, nt in enumerate(lnt):
        st.add_ref(key=f'nt{ii}', size=nt)


def _add_data(st=None, nc=None, nx=None, lnt=None):

    x = np.linspace(1, 2, nx)
    y = np.exp((x - 0.5)**2)
    y[-5] = np.nan
    lt = [np.linspace(1, 10, nt) for nt in lnt]
    lprof = [(1 + np.cos(t)[:, None]) * x[None, :] for t in lt]
    lprof[0][10, -5] = np.nan

    # add data dependening on these references
    st.add_data(
        key='x',
        data=x,
        dim='distance',
        quant='radius',
        units='m',
        ref='nx',
    )

    st.add_data(
        key='y',
        data=y,
        dim='distance',
        quant='radius',
        units='m',
        ref='nx',
    )

    for ii, nt in enumerate(lnt):
        st.add_data(
            key=f't{ii}',
            data=lt[ii],
            dim='time',
            units='s',
            ref=f'nt{ii}',
        )
        st.add_data(
            key=f'prof{ii}',
            data=lprof[ii],
            dim='velocity',
            units='m/s',
            ref=(f'nt{ii}', 'nx'),
        )

    # add replication of prof2 for plot_as_mobile_lines in 2d
    st.add_data(
        key='prof2-bis',
        data=lprof[2] + np.random.normal(scale=0.1, size=(lnt[2], nx)),
        dim='velocity',
        units='m/s',
        ref=(f'nt{2}', 'nx'),
    )

    # add 3d array
    st.add_data(
        key='prof0-bis',
        data=lprof[0] + np.random.normal(scale=0.1, size=lprof[0].shape),
        dim='blabla',
        ref=('nt0', 'nx'),
    )

    # add 3d array
    st.add_data(
        key='3d',
        data=np.arange(nc)[:, None, None] + lprof[0][None, :, :],
        dim='blabla',
        ref=('nc', 'nt0', 'nx'),
    )
    st.add_data(
        key='3d-bis',
        data=(
            np.arange(nc)[:, None, None]
            + lprof[0][None, :, :]
            + np.random.normal(scale=0.01, size=(nc, lnt[0], nx))
        ),
        dim='blabla',
        ref=('nc', 'nt0', 'nx'),
    )


def _add_obj(st=None, nc=None):
    for ii in range(nc):
        st.add_obj(
            which='campaign',
            key=f'c{ii}',
            index=ii,
            start_date=f'{ii}.04.2022',
            end_date=f'{ii+5}.05.2022',
            operator='Barnaby' if ii > 2 else 'Jack Sparrow',
            comment='leak on tube' if ii == 1 else 'none',
        )


#######################################################
#
#     Instanciate
#
#######################################################


class Test01_Instanciate():

    @classmethod
    def setup_class(cls):
        cls.st = DataStock()
        cls.nc = 5
        cls.nx = 80
        cls.lnt = [100, 90, 80, 120, 80]

    # ------------------------
    #   Populating
    # ------------------------

    def test01_add_ref(self):
        _add_ref(st=self.st, nc=self.nc, nx=self.nx, lnt=self.lnt)

    def test02_add_data(self):
        _add_data(st=self.st, nc=self.nc, nx=self.nx, lnt=self.lnt)

    def test03_add_obj(self):
        _add_obj(st=self.st, nc=self.nc)


#######################################################
#
#     Main testing class
#
#######################################################


class Test02_Manipulate():

    @classmethod
    def setup_class(cls):
        cls.st = DataStock()
        cls.nc = 5
        cls.nx = 80
        cls.lnt = [100, 90, 80, 120, 80]

        _add_ref(st=cls.st, nc=cls.nc, nx=cls.nx, lnt=cls.lnt)
        _add_data(st=cls.st, nc=cls.nc, nx=cls.nx, lnt=cls.lnt)
        _add_obj(st=cls.st, nc=cls.nc)

    # ------------------------
    #   Add / remove
    # ------------------------

    def test01_add_param(self):
        # create new 'campaign' parameter for data arrays
        self.st.add_param('campaign', which='data')

        # tag each data with its campaign
        for ii in range(self.nc):
            self.st.set_param(
                which='data',
                key=f't{ii}',
                param='campaign',
                value=f'c{ii}',
            )
            self.st.set_param(
                which='data',
                key=f'prof{ii}',
                param='campaign',
                value=f'c{ii}',
            )

    def test02_remove_param(self):
        self.st.add_param('blabla', which='campaign')
        self.st.remove_param('blabla', which='campaign')

    # ------------------------
    #   Selection / sorting
    # ------------------------

    def test03_select(self):
        key = self.st.select(which='data', units='s', returnas=str)
        assert key.tolist() == ['t0', 't1', 't2', 't3', 't4']

        out = self.st.select(dim='time', returnas=int)
        assert len(out) == 5, out

        # test quantitative param selection
        out = self.st.select(which='campaign', index=[2, 4])
        assert len(out) == 3

        out = self.st.select(which='campaign', index=(2, 4))
        assert len(out) == 2

    def test04_sortby(self):
        self.st.sortby(which='data', param='units')

    # ------------------------
    #   show
    # ------------------------

    def test05_show(self):
        self.st.show()
        self.st.show_data()
        self.st.show_obj()
        self.st.show_interactive()

    # ------------------------
    #   Interpolate
    # ------------------------

    def test06_get_ref_vector(self):
        (
            hasref, hasvector,
            ref, key_vector,
            values, dind,
        ) = self.st.get_ref_vector(
            key='prof0',
            ref='nx',
            values=[1, 2, 2.01, 3],
            ind_strict=False,
        )
        assert hasref is True and hasvector is True
        assert values.size == dind['ind'].size == 4
        assert dind['indr'].shape == (2, 4)

    def test07_get_ref_vector_common(self):
        hasref, ref, key, val, dout = self.st.get_ref_vector_common(
            keys=['t0', 'prof0', 'prof1', 't3'],
            dim='time',
        )

    def test08_domain_ref(self):

        domain = {
            'nx': [1.5, 2],
            'x': (1.5, 2),
            'y': [[0, 0.9], (0.1, 0.2)],
            't0': {'domain': [2, 3]},
            't1': {'domain': [[2, 3], (2.5, 3), [4, 6]]},
            't2': {'ind': self.st.ddata['t2']['data'] > 5},
        }

        dout = self.st.get_domain_ref(domain=domain)

        lk = list(domain.keys())
        assert all([isinstance(dout[k0]['ind'], np.ndarray) for k0 in lk])

    def test09_binning(self):

        bins = np.linspace(1, 5, 10)
        lk = [
            ('y', None, 0),
            ('y', 'nx', 0),
            (None, 'nt0', 0),
            ('prof0', 'nt0', 0),
            ('prof0', 'x', 1),
        ]

        for (k0, kr, ax) in lk:
            dout = self.st.binning(
                keys=k0,
                ref_key=kr,
                bins=bins,
            )

            k0 = list(dout.keys())[0]
            shape = list(self.st.ddata[k0]['data'].shape)
            shape[ax] = bins.size - 1
            assert dout[k0]['data'].shape == tuple(shape)

    def test10_interpolate(self):

        lk = ['y', 'y', 'prof0', 'prof0', 'prof0', '3d']
        lref = [None, 'nx', 't0', ['nt0', 'nx'], ['t0', 'x'], ['t0', 'x']]
        lax = [[0], [0], [0], [0, 1], [0, 1], [1, 2]]
        lgrid = [False, False, False, False, True, False]
        llog = [False, False, False, True, False, False]

        x2d = np.array([[1.5, 2.5], [1, 2]])
        x3d = np.random.random((5, 4, 3))
        lx0 = [x2d, [1.5, 2.5], [1.5, 2.5], x2d, [1.5, 2.5], x3d]
        lx1 = [None, None, None, x2d, [1.2, 2.3], x3d]
        ldom = [None, None, {'nx': [1.5, 2]}, None, None, None]

        zipall = zip(lk, lref, lax, llog, lgrid, lx0, lx1, ldom)
        for ii, (kk, rr, aa, lg, gg, x0, x1, dom) in enumerate(zipall):

            domain = self.st.get_domain_ref(domain=dom)

            dout = self.st.interpolate(
                keys=kk,
                ref_key=rr,
                x0=x0,
                x1=x1,
                grid=gg,
                deg=2,
                deriv=None,
                log_log=lg,
                return_params=False,
                domain=dom,
            )

            assert isinstance(dout, dict)
            assert isinstance(dout[kk]['data'], np.ndarray)
            shape = list(self.st.ddata[kk]['data'].shape)
            x0s = np.array(x0).shape if gg is False else (len(x0), len(x1))
            if dom is None:
                shape = tuple(np.r_[shape[:aa[0]], x0s, shape[aa[-1]+1:]])
            else:
                shape = tuple(np.r_[x0s, 39]) if ii == 2 else None
            if dout[kk]['data'].shape != tuple(shape):
                msg = str(dout[kk]['data'].shape, shape, kk, rr)
                raise Exception(msg)

    def test11_interpolate_common_refs(self):
        lk = ['3d', '3d', '3d']
        lref = ['t0', ['nt0', 'nx'], ['nx']]
        lrefc = ['nc', 'nc', 'nt0']
        lax = [[0], [0, 2], [2], [2]]
        llog = [False, True, False]

        # add data for common ref interpolation
        nt0 = self.st.dref['nt0']['size']
        nt1 = self.st.dref['nt1']['size']
        nc = self.st.dref['nc']['size']
        self.st.add_data(
            key='data_com',
            data=1. + np.random.random((nc, nt1, nt0))*2,
            ref=('nc', 'nt1', 'nt0'),
        )

        lx1 = [None, 'data_com', None]
        ls = [(5, 90, 100, 80), (5, 90, 100), (5, 100, 5, 90)]
        lr = [
            ('nc', 'nt1', 'nt0', 'nx'),
            ('nc', 'nt1', 'nt0'),
            ('nc', 'nt0', 'nc', 'nt1'),
        ]

        zipall = zip(lk, lref, lax, llog, lx1, lrefc, ls, lr)
        for ii, (kk, rr, aa, lg, x1, refc, ss, ri) in enumerate(zipall):

            dout, dparams = self.st.interpolate(
                keys=kk,
                ref_key=rr,
                x0='data_com',
                x1=x1,
                grid=False,
                deg=2,
                deriv=None,
                log_log=lg,
                store=False,
                return_params=True,
                domain=None,
                ref_com=refc,
            )

            assert isinstance(dout, dict)
            assert isinstance(dout[kk]['data'], np.ndarray)

            if not (dout[kk]['data'].shape == ss and dout[kk]['ref'] == ri):
                lstr = [f'\t- {k0}: {v0}' for k0, v0 in dparams.items()]
                msg = (
                    "Wrong interpolation shape / ref:\n"
                    f"\t- ii: {ii}\n"
                    f"\t- keys: {kk}\n"
                    f"\t- ref_key: {rr}\n"
                    f"\t- x1: {x1}\n"
                    f"\t- ref_com: {refc}\n"
                    f"\t- log_log: {lg}\n"
                    f"\t- key['ref']: {self.st.ddata[kk]['ref']}\n"
                    f"\t- x0['ref']: {self.st.ddata['data_com']['ref']}\n"
                    "\n"
                    # + "\n".join(lstr)
                    "\n"
                    f"\t- Expected shape: {ss}\n"
                    f"\t- Observed shape: {dout[kk]['data'].shape}\n"
                    f"\t- Expected ref: {ri}\n"
                    f"\t- Observed ref: {dout[kk]['ref']}\n"

                )
                raise Exception(msg)


                # Not tested: float, store=True, inplace

    # ------------------------
    #   Plotting
    # ------------------------

    def test12_plot_as_array(self):
        dax = self.st.plot_as_array(key='t0')
        dax = self.st.plot_as_array(key='prof0')
        dax = self.st.plot_as_array(key='3d')
        plt.close('all')

    def test13_plot_BvsA_as_distribution(self):
        dax = self.st.plot_BvsA_as_distribution(keyA='prof0', keyB='prof0-bis')
        plt.close('all')

    def test14_plot_as_profile1d(self):
        dax = self.st.plot_as_profile1d(
            key='prof0',
            key_time='t0',
            keyX='prof0-bis',
            bck='lines',
        )
        plt.close('all')

    def test15_plot_as_mobile_lines(self):

        # 3d
        dax = self.st.plot_as_mobile_lines(
            keyX='3d',
            keyY='3d-bis',
            key_time='t0',
            key_chan='x',
        )

        # 2d
        dax = self.st.plot_as_mobile_lines(
            keyX='prof2',
            keyY='prof2-bis',
            key_chan='nx',
        )

        plt.close('all')

    # ------------------------
    #   File handling
    # ------------------------

    def test16_copy_equal(self):
        st2 = self.st.copy()
        assert st2 is not self.st

        msg = st2.__eq__(self.st, returnas=str)
        if msg is not True:
            raise Exception(msg)

    def test17_get_nbytes(self):
        nb, dnb = self.st.get_nbytes()

    def test18_saveload(self, verb=False):
        pfe = self.st.save(path=_PATH_OUTPUT, verb=verb, return_pfe=True)
        st2 = load(pfe, verb=verb)
        # Just to check the loaded version works fine
        msg = st2.__eq__(self.st, returnas=str)
        if msg is not True:
            raise Exception(msg)
        os.remove(pfe)

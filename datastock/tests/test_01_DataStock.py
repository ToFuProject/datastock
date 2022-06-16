
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
    lt = [np.linspace(0, 10, nt) for nt in lnt]
    lprof = [(1 + np.cos(t)[:, None]) * x[None, :] for t in lt]

    # add data dependening on these references
    st.add_data(
        key='x',
        data=x,
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

    def setup(self):
        self.st = DataStock()
        self.nc = 5
        self.nx = 80
        self.lnt = [100, 90, 80, 120, 80]

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

    def setup(self):
        self.st = DataStock()
        self.nc = 5
        self.nx = 80
        self.lnt = [100, 90, 80, 120, 80]

        _add_ref(st=self.st, nc=self.nc, nx=self.nx, lnt=self.lnt)
        _add_data(st=self.st, nc=self.nc, nx=self.nx, lnt=self.lnt)
        _add_obj(st=self.st, nc=self.nc)

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

    # ------------------------
    #   Interpolate
    # ------------------------

    def test06_get_ref_vector(self):
        (
            hasref, hasvector,
            ref, key_vector,
            values, indices, indu, ind_reverse, indok,
        ) = self.st.get_ref_vector(
            key='prof0',
            ref='nx',
            values=[1, 2, 2.01, 3],
            ind_strict=False,
        )
        assert hasref is True and hasvector is True
        assert values.size == indices.size == 4
        assert ind_reverse.shape == (2, 4)

    def test07_get_ref_vector_common(self):
        hasref, hasvect, val, dout = self.st.get_ref_vector_common(
            keys=['t0', 'prof0', 'prof1', 't3'],
            dim='time',
        )

    def test08_interpolate(self):
        out = self.st.interpolate(
            keys='prof0',
            ref_keys=None,
            ref_quant=None,
            pts_axis0=[1.5, 2.5],
            pts_axis1=[1., 2.],
            pts_axis2=None,
            grid=False,
            deg=2,
            deriv=None,
            log_log=False,
            return_params=False,
        )
        assert isinstance(out, dict)
        assert isinstance(out['prof0'], np.ndarray)

    # ------------------------
    #   Plotting
    # ------------------------

    def test09_plot_as_array(self):
        dax = self.st.plot_as_array(key='t0')
        dax = self.st.plot_as_array(key='prof0')
        dax = self.st.plot_as_array(key='3d')
        plt.close('all')

    def test10_plot_BvsA_as_distribution(self):
        dax = self.st.plot_BvsA_as_distribution(keyA='prof0', keyB='prof0-bis')
        plt.close('all')

    def test11_plot_as_profile1d(self):
        dax = self.st.plot_as_profile1d(
            key='prof0',
            key_time='t0',
            keyX='prof0-bis',
            bck='lines',
        )
        plt.close('all')

    def test12_plot_as_mobile_lines(self):

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

    def test13_copy_equal(self):
        st2 = self.st.copy()
        assert st2 is not self.st

        msg = st2.__eq__(self.st, returnas=str)
        if msg is not True:
            raise Exception(msg)

    def test14_get_nbytes(self):
        nb, dnb = self.st.get_nbytes()

    def test15_saveload(self, verb=False):
        pfe = self.st.save(path=_PATH_OUTPUT, verb=verb, return_pfe=True)
        st2 = load(pfe, verb=verb)
        # Just to check the loaded version works fine
        msg = st2.__eq__(self.st, returnas=str)
        if msg is not True:
            raise Exception(msg)
        os.remove(pfe)

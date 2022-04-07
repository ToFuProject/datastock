
"""
This module contains tests for tofu.geom in its structured version
"""


# Built-in
import os
import warnings


# Standard
import numpy as np


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
#     Main testing class
#
#######################################################


class Test01_DataStock():

    def setup(self):
        self.st = DataStock()
        self.nc = 5
        self.nx = 80
        self.lnt = [100, 90, 80, 120, 110]

    # ------------------------
    #   Populating
    # ------------------------

    def test01_add_ref(self):

        # ------------------
        # Populate DataStock

        # add references (i.e.: store size of each dimension under a unique key)
        self.st.add_ref(key='nc', size=self.nc)
        self.st.add_ref(key='nx', size=self.nx)
        for ii, nt in enumerate(self.lnt):
            self.st.add_ref(key=f'nt{ii}', size=nt)

    def test02_add_data(self):

        x = np.linspace(1, 2, self.nx)
        lt = [np.linspace(0, 10, nt) for nt in self.lnt]
        lprof = [(1 + np.cos(t)[:, None]) * x[None, :] for t in lt]

        # add data dependening on these references
        # you can, optionally, specify units, physical dimensionality (ex: distance, time...), quantity (ex: radius, height, ...) and name (to your liking)

        self.st.add_data(
            key='x',
            data=x,
            dimension='distance',
            quant='radius',
            units='m',
            ref='nx',
        )

        for ii, nt in enumerate(self.lnt):
            self.st.add_data(
                key=f't{ii}',
                data=lt[ii],
                dimension='time',
                units='s',
                ref=f'nt{ii}',
            )
            self.st.add_data(
                key=f'prof{ii}',
                data=lprof[ii],
                dimension='velocity',
                units='m/s',
                ref=(f'nt{ii}', 'x'),
            )

    def test03_add_obj(self):
        pass

    # ------------------------
    #   Add / remove
    # ------------------------

    # ------------------------
    #   Selection / sorting
    # ------------------------

    def test10_select(self):
        key = data.select(which='data', units='s', returnas=str)
        assert key == ['trace10']

        out = data.select(units='a.u.', returnas=int)
        assert len(out) == 12, out

        # test quantitative param selection
        out = self.lobj[1].select(which='lines', lambda0=[3.5e-10, 6e-10])
        assert len(out) == 2

        out = self.lobj[1].select(which='lines', lambda0=(3.5e-10, 6e-10))
        assert len(out) == 1

    def test11_sortby(self):
        self.st.sortby(which='data', param='units')

    # ------------------------
    #   show
    # ------------------------

    def test12_show(self):
        self.st.show()

    # ------------------------
    #   Interpolate
    # ------------------------

    def test15_interpolate(self):
        pass

    # ------------------------
    #   Plotting
    # ------------------------

    # ------------------------
    #   File handling
    # ------------------------

    def test20_copy_equal(self):
        st2 = self.st.copy()
        assert st2 == self.st
        assert st2 is not self.st

    def test21_get_nbytes(self):
        nb, dnb = self.st.get_nbytes()

    def test23_saveload(self, verb=False):
            pfe = self.st.save(path=_PATH_OUTPUT, verb=verb, return_pfe=True)
            st2 = load(pfe, verb=verb)
            # Just to check the loaded version works fine
            assert st2 == self.st
            os.remove(pfe)


"""
    def test07_getsetaddremove_param(self):
        data = self.lobj[0]

        out = data.get_param('units')
        data.set_param('units', value='T', key='trace00')
        data.add_param('shot', value=np.arange(0, len(data.ddata)))
        assert np.all(
            data.get_param('shot')['shot'] == np.arange(0, len(data.ddata))
        )
        data.remove_param('shot')
        assert 'shot' not in data.get_lparam(which='data')
"""

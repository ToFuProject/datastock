# ###############
# __version__
# ###############


__version__ = "0.0.50"
# from setuptools_scm import get_version
# __version__ = get_version(root='..', relative_to=__file__)

# from importlib.metadata import version
# __version__ = version(__package__)
# cleanup
# del get_version


# ###############
# sub-packages
# ###############


from . import _generic_check
from ._generic_utils_plot import *
from ._class import DataStock
from ._saveload import load, get_files
from ._direct_calls import *
from . import tests

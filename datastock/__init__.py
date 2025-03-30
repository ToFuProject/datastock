

# ###############
# sub-packages
# ###############


from . import _generic_check
from ._generic_utils_plot import *
from ._class import DataStock
from ._saveload import load, get_files
from ._direct_calls import *
from . import tests


# ###############
# __version__
# ###############


from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("package-name")
except PackageNotFoundError:
    # package is not installed
    pass
#
# cleanup
del version, PackageNotFoundError

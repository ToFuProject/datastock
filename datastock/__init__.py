

from .version import __version__

from . import _generic_check
from ._class import DataStock
from ._saveload import load, get_files
from ._direct_calls import *
from . import tests

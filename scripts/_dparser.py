import sys
import os
import getpass
import argparse


# test if in a git repo
_HERE = os.path.abspath(os.path.dirname(__file__))
_REPOPATH = os.path.dirname(_HERE)
_REPO_NAME = 'datastock'


# #############################################################################
#       utility functions
# #############################################################################


def _str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['yes', 'true', 'y', 't', '1']:
        return True
    elif v.lower() in ['no', 'false', 'n', 'f', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected!')


def _str2boolstr(v):
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ['yes', 'true', 'y', 't', '1']:
            return True
        elif v.lower() in ['no', 'false', 'n', 'f', '0']:
            return False
        elif v.lower() == 'none':
            return None
        else:
            return v
    else:
        raise argparse.ArgumentTypeError('Boolean, None or str expected!')


def _str2tlim(v):
    c0 = (v.isdigit()
          or ('.' in v
              and len(v.split('.')) == 2
              and all([vv.isdigit() for vv in v.split('.')])))
    if c0 is True:
        v = float(v)
    elif v.lower() == 'none':
        v = None
    return v


# #############################################################################
#       Parser for version
# #############################################################################


def parser_version():
    msg = f""" Get {_REPO_NAME} version from bash

    If run from a git repo containing {_REPO_NAME}, just returns git describe
    Otherwise reads the version stored in {_REPO_NAME}/version.py

    """
    ddef = {
        'path': os.path.join(_REPOPATH, _REPO_NAME),
        'envvar': False,
        'verb': True,
        'warn': True,
        'force': False,
        'name': f'{_REPO_NAME.upper()}_VERSION',
    }

    # Instanciate parser
    parser = argparse.ArgumentParser(description=msg)

    # optional path
    parser.add_argument(
        '-p', '--path',
        type=str,
        help='source directory where version.py is found',
        required=False,
        default=ddef['path'],
    )

    # verb
    parser.add_argument(
        '-v', '--verb',
        type=_str2bool,
        help='flag indicating whether to print the version',
        required=False,
        default=ddef['verb'],
    )

    return ddef, parser


# #############################################################################
#       Parser dict
# #############################################################################


_DPARSER = {
    'version': parser_version,
}

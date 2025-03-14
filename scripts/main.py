#!/usr/bin/env python

# Built-in
import sys
import os
import argparse


# import parser dicti
from . import _dparser
from . import _bash_version


###################################################
###################################################
#       default values
###################################################


_PATH_HERE = os.path.abspath(os.path.dirname(__file__))


_LOPTIONS = ['--version']
_LOPSTRIP = [ss.strip('--') for ss in _LOPTIONS]


###################################################
###################################################
#       function
###################################################


def datastock_bash(option=None, ddef=None, **kwdargs):
    """ Print tofu version and / or store in environment variable """

    # --------------
    # Check inputs
    # --------------

    if option not in _LOPSTRIP:
        msg = (
            "Provided option is not acceptable:\n"
            f"\t- available: {_LOPSTRIP}\n"
            f"\t- provided:  {option}"
        )
        raise Exception(msg)

    # --------------
    # call corresponding bash command
    # --------------

    if option == 'version':
        _bash_version.main(
            ddef=ddef,
            **kwdargs,
        )


###################################################
###################################################
#          main
###################################################


def main():
    # Parse input arguments
    msg = """ Get tofu version from bash optionally set an enviroment variable

    If run from a git repo containing tofu, simply returns git describe
    Otherwise reads the tofu version stored in tofu/version.py

    """

    # ------------------
    # Instanciate parser
    # ------------------

    parser = argparse.ArgumentParser(description=msg)

    # ---------------------
    # which script to call
    # ---------------------

    parser.add_argument(
        'option',
        nargs='?',
        type=str,
        default='None',
    )

    #
    parser.add_argument(
        '-v', '--version',
        help='get tofu current version',
        required=False,
        action='store_true',
    )

    # Others
    # parser.add_argument('kwd', nargs='?', type=str, default='None')

    # -------------------
    # check options
    # -------------------

    if sys.argv[1] not in _LOPTIONS:
        msg = (
            "Provided option is not acceptable:\n"
            f"\t- available: {_LOPTIONS}\n"
            f"\t- provided:  {sys.argv[1]}\n"
        )
        raise Exception(msg)

    if len(sys.argv) > 2:
        if any([ss in sys.argv[2:] for ss in _LOPTIONS]):
            lopt = [ss for ss in sys.argv[1:] if ss in _LOPTIONS]
            msg = (
                "Only one option can be provided!\n"
                f"\t- provided: {lopt}"
            )
            raise Exception(msg)

    # ----------------------
    # def values and parser
    # ----------------------

    option = sys.argv[1].strip('--')
    ddef, parser = _dparser._DPARSER[option]()
    if len(sys.argv) > 2:
        kwdargs = dict(parser.parse_args(sys.argv[2:])._get_kwargs())
    else:
        kwdargs = {}

    # ----------------------
    # Call function
    # ----------------------

    datastock_bash(option=option, ddef=ddef, **kwdargs)


###################################################
###################################################
#                   __main__
###################################################

if __name__ == '__main__':
    main()

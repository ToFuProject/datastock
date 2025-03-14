#!/usr/bin/env python

# Built-in
import os


###################################################
###################################################
#       DEFAULTS
###################################################


_PATH_HERE = os.path.dirname(__file__)


###################################################
###################################################
#       function
###################################################


def main(
    verb=None,
    envvar=None,
    path=None,
    warn=None,
    force=None,
    ddef=None,
):
    """ Print version """

    # --------------
    # Check inputs
    # --------------

    kwd = locals()
    for k0 in set(ddef.keys()).intersection(kwd.keys()):
        if kwd[k0] is None:
            kwd[k0] = ddef[k0]
    verb, path = kwd['verb'], kwd['path']

    # verb, warn, force
    dbool = {'verb': verb}
    for k0, v0 in dbool.items():
        if v0 is None:
            dbool[k0] = ddef[k0]
        if not isinstance(dbool[k0], bool):
            msg = (
                f"Arg '{k0}' must be a bool\n"
                f"\t- provided: {dbool[k0]}\n"
            )
            raise Exception(msg)

    # --------------
    # Fetch version from git tags, and write to version.py
    # Also, when git is not available (PyPi package), use stored version.py

    pfe = os.path.join(path, 'version.py')
    if not os.path.isfile(pfe):
        msg = (
            "It seems your current install has no version.py:\n"
            f"\t- looked for: {pfe}"
        )
        raise Exception(msg)

    # --------------
    # Read file

    with open(pfe, 'r') as fh:
        version = fh.read().strip().split("=")[-1].replace("'", '')
    version = version.lower().replace('v', '').replace(' ', '')

    # --------------
    # Outputs

    if dbool['verb'] is True:
        print(version)

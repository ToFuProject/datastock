

# Built-in
import os


# #############################################################################
# #############################################################################
#               Main routine
# #############################################################################


def find_plateau(
    coll=None,
    keys=None,
    ref=None,
):

    # to be removed
    msg = (
        "\nFind_plateau is not implemented yet!\n"
        "It would nice right?\n"
        "...and probably not too hard to implement"
        " since another library already exists (by Jorge MORALES) for this task...\n\n"
        "=> you can volunteer to make it happen here:\n"
        "https://github.com/ToFuProject/datastock/issues/47\n\n"
        "estimated difficulty level: 3-4/5"
    )
    raise NotImplementedError(msg)

    # ------------
    # check inputs

    # check the conformity of all inputs
    _find_plateau_check(
        coll=coll,
        keys=keys,
        ref=ref,
    )


    # -------------
    # import the library (optional dependency)
    try:
        # import as
        pass

    except Exception as err:
        msg = str(err) + (
            "\nImpossible to import ..., it might not be installed?"
        )
        raise Exception(msg)

    # ----------------------------------
    # interface with the library

    # the current library decomposes time trace into a series of plateau
    # a plateau is a time phase during which the data change rate is close to 0
    # Assuming it is possible to decompose time trace into n phases with fixed
    # rates of change (0 being a plateau), it would be nice to store, for each
    # point, the associated rate of change, and the time since the last phase
    # => see with Jorge if that is possible
    # alternatively we can just store the plateaus (rate = 0) and time since
    # the last plateau

    # ----------------
    # return

    # return, a dict, with, for each data key, 2 arrays
    # - rate: rate of change
    # - dt: for each point in phase n, the time since last phase n-1

    # dout = {
        # k0: {
            # 'rate': (nt,) array of floats,
            # 'dt': (nt,) array of floats,
        # },
        # for k0 in keys
    # }
    dout = None

    return dout


# #############################################################################
# #############################################################################
#               sub-routine
# #############################################################################


def _find_plateau_check(
    coll=None,
    keys=None,
    ref=None,
):

    # ------
    # which

    # -----
    # keys


    return keys, ref

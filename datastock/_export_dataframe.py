

# Built-in
import os





# #############################################################################
# #############################################################################
#               Main routine
# #############################################################################


def to_dataframe(
    coll=None,
    which=None,
    keys=None,
):

    # to be removed
    msg = (
        "\nExport to pandas DataFrame is not implemented yet!\n"
        "It would nice right?  and probably not too hard to implement...\n\n"
        "=> you can volunteer to make it happen here:\n"
        "https://github.com/ToFuProject/datastock/issues/46\n\n"
        "estimated difficulty level: 2/5"
    )
    raise NotImplementedError(msg)

    # ------------
    # check inputs

    # check the conformity of all inputs
    _to_dataframe_check(
        coll=coll,
        which=which,
        keys=keys,
    )


    # ----------------------------------
    # instanciate and populate dataframe

    # import optional dependency
    import pandas

    # instanciate
    df = pandas.DataFrame()


    # ----------------
    # return


    return df


# #############################################################################
# #############################################################################
#               sub-routine
# #############################################################################


def _to_dataframe_check(
    coll=None,
    which=None,
    keys=None,
):

    # ------
    # which

    # -----
    # keys


    return which, keys

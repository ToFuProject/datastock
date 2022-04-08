

import nupy as np


from . import _generic_check
from . import _DataCollection_check_inputs


# #############################################################################
# #############################################################################
#               Identify references
# #############################################################################


# def _get_keyingroup_ddata(
    # dd=None, dd_name='data',
    # key=None, monot=None,
    # msgstr=None, raise_=False,
# ):
    # """ Return the unique data key matching key

    # Here, key can be interpreted as name / source / units / quant...
    # All are tested using select() and a unique match is returned
    # If not unique match an error message is either returned or raised

    # """

    # # ------------------------
    # # Trivial case: key is actually a ddata key

    # if key in dd.keys():
        # return key, None

    # # ------------------------
    # # Non-trivial: check for a unique match on other params

    # dind = _select(
        # dd=dd, dd_name=dd_name,
        # dim=key, quant=key, name=key, units=key, source=key,
        # monot=monot,
        # log='raw',
        # returnas=bool,
    # )
    # ind = np.array([ind for kk, ind in dind.items()])

    # # Any perfect match ?
    # nind = np.sum(ind, axis=1)
    # sol = (nind == 1).nonzero()[0]
    # key_out, msg = None, None
    # if sol.size > 0:
        # if np.unique(sol).size == 1:
            # indkey = ind[sol[0], :].nonzero()[0]
            # key_out = list(dd.keys())[indkey]
        # else:
            # lstr = "[dim, quant, name, units, source]"
            # msg = "Several possible matches in {} for {}".format(lstr, key)
    # else:
        # lstr = "[dim, quant, name, units, source]"
        # msg = "No match in {} for {}".format(lstr, key)

    # # Complement error msg and optionally raise
    # if msg is not None:
        # lk = ['dim', 'quant', 'name', 'units', 'source']
        # dk = {
            # kk: (
                # dind[kk].sum(),
                # sorted(set([vv[kk] for vv in dd.values()]))
            # ) for kk in lk
        # }
        # msg += (
            # "\n\nRequested {} could not be identified!\n".format(msgstr)
            # + "Please provide a valid (unique) key/name/dim/quant/units:\n\n"
            # + '\n'.join([
                # '\t- {} ({} matches): {}'.format(kk, dk[kk][0], dk[kk][1])
                # for kk in lk
            # ])
            # + "\nProvided:\n\t'{}'".format(key)
        # )
        # if raise_:
            # raise Exception(msg)
    # return key_out, msg


# def _get_possible_ref12d(
    # dd=None,
    # key=None, ref1d=None, ref2d=None,
    # group1d='radius',
    # group2d='mesh2d',
# ):

    # # Get relevant lists
    # kq, msg = _get_keyingroup_ddata(
        # dd=dd,
        # key=key, group=group2d, msgstr='quant', raise_=False,
    # )

    # if kq is not None:
        # # The desired quantity is already 2d
        # k1d, k2d = None, None

    # else:
        # # Check if the desired quantity is 1d
        # kq, msg = _get_keyingroup_ddata(
            # dd=dd,
            # key=key, group=group1d,
            # msgstr='quant', raise_=True,
        # )

        # # Get dict of possible {ref1d: lref2d}
        # ref = [rr for rr in dd[kq]['ref'] if dd[rr]['group'] == (group1d,)][0]
        # lref1d = [
            # k0 for k0, v0 in dd.items()
            # if ref in v0['ref'] and v0['monot'][v0['ref'].index(ref)] is True
        # ]

        # # Get matching ref2d with same quant and good group
        # lquant = list(set([dd[kk]['quant'] for kk in lref1d]))
        # dref2d = {
            # k0: [
                # kk for kk in _select(
                    # dd=dd, quant=dd[k0]['quant'],
                    # log='all', returnas=str,
                # )
                # if group2d in dd[kk]['group']
                # and not isinstance(dd[kk]['data'], dict)
            # ]
            # for k0 in lref1d
        # }
        # dref2d = {k0: v0 for k0, v0 in dref2d.items() if len(v0) > 0}

        # if len(dref2d) == 0:
            # msg = (
                # "No match for (ref1d, ref2d) for ddata['{}']".format(kq)
            # )
            # raise Exception(msg)

        # # check ref1d
        # if ref1d is None:
            # if ref2d is not None:
                # lk = [k0 for k0, v0 in dref2d.items() if ref2d in v0]
                # if len(lk) == 0:
                    # msg = (
                        # "\nNon-valid interpolation intermediate\n"
                        # + "\t- provided:\n"
                        # + "\t\t- ref1d = {}, ref2d = {}\n".format(ref1d, ref2d)
                        # + "\t- valid:\n{}".format(
                            # '\n'.join([
                                # '\t\t- ref1d = {}  =>  ref2d in {}'.format(
                                    # k0, v0
                                # )
                                # for k0, v0 in dref2d.items()
                            # ])
                        # )
                    # )
                    # raise Exception(msg)
                # if kq in lk:
                    # ref1d = kq
                # else:
                    # ref1d = lk[0]
            # else:
                # if kq in dref2d.keys():
                    # ref1d = kq
                # else:
                    # ref1d = list(dref2d.keys())[0]
        # else:
            # ref1d, msg = _get_keyingroup_ddata(
                # dd=dd,
                # key=ref1d, group=group1d,
                # msgstr='ref1d', raise_=False,
            # )
        # if ref1d not in dref2d.keys():
            # msg = (
                # "\nNon-valid interpolation intermediate\n"
                # + "\t- provided:\n"
                # + "\t\t- ref1d = {}, ref2d = {}\n".format(ref1d, ref2d)
                # + "\t- valid:\n{}".format(
                    # '\n'.join([
                        # '\t\t- ref1d = {}  =>  ref2d in {}'.format(
                            # k0, v0
                        # )
                        # for k0, v0 in dref2d.items()
                    # ])
                # )
            # )
            # raise Exception(msg)

        # # check ref2d
        # if ref2d is None:
            # ref2d = dref2d[ref1d][0]
        # else:
            # ref2d, msg = _get_keyingroup_ddata(
                # dd=dd,
                # key=ref2d, group=group2d,
                # msgstr='ref2d', raise_=False,
            # )
        # if ref2d not in dref2d[ref1d]:
            # msg = (
                # "\nNon-valid interpolation intermediate\n"
                # + "\t- provided:\n"
                # + "\t\t- ref1d = {}, ref2d = {}\n".format(ref1d, ref2d)
                # + "\t- valid:\n{}".format(
                    # '\n'.join([
                        # '\t\t- ref1d = {}  =>  ref2d in {}'.format(
                            # k0, v0
                        # )
                        # for k0, v0 in dref2d.items()
                    # ])
                # )
            # )
            # raise Exception(msg)

    # return kq, ref1d, ref2d

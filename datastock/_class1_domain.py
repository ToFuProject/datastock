# -*- coding: utf-8 -*-


import copy


import numpy as np


# ########################################################
# ########################################################
#           Main
# ########################################################


def domain_ref(
    coll=None,
    domain=None,
):
    """ Return dict of indices matching desired domain """

    # ---------------
    # check inputs

    if domain is None:
        return

    domain = _check(coll=coll, domain=domain)

    # -----------
    # get indices

    lvectu = sorted({v0['vect'] for v0 in domain.values()})

    for vv in lvectu:

        lk0 = [k0 for k0, v0 in domain.items() if v0['vect'] == vv]
        for k0 in lk0:

            if domain[k0].get('domain') is None:
                continue

            domain[k0]['ind'] = _set_ind_from_domain(
                vect=coll.ddata[domain[k0]['vect']]['data'],
                domain=domain[k0]['domain'],
            )

    return domain


# ########################################################
# ########################################################
#           checks
# ########################################################


def _check(
    coll=None,
    domain=None,
):

    # ---------
    # prepare

    ldata = list(coll.ddata.keys())
    lref = list(coll.dref.keys())

    # ------------
    # domain

    c0 = (
        isinstance(domain, dict)
        and all(k0 in lref + ldata for k0, v0 in domain.items())
    )

    if not c0:
        msg = (
            "Arg domain mut be a dict with keys as ref or data\n"
            f"Provided: {domain}"
        )
        raise Exception(msg)

    # ------------
    # check each key

    dfail = {}
    domain = copy.deepcopy(domain)
    for k0, v0 in domain.items():

        # check ref vector
        kwd = {'ref': k0} if k0 in lref else {'key': k0}
        hasref, hasvect, ref, vect = coll.get_ref_vector(**kwd)[:4]
        if not (hasref and ref is not None):
            dfail[k0] = "No associated ref identified!"
            continue
        if not (hasvect and vect is not None):
            dfail[k0] = "No associated ref vector identified!"
            continue

        # v0 is dict
        ltyp = (list, tuple, np.ndarray)
        if isinstance(v0, ltyp):
            domain[k0] = {'domain': v0}
        elif np.isscalar(v0):
            domain[k0] = {'domain': v0}

        c0 = (
            isinstance(domain[k0], dict)
            and any(ss in ['ind', 'domain'] for ss in domain[k0].keys())
            and (
                isinstance(domain[k0].get('domain'), ltyp)
                or np.isscalar(domain[k0].get('domain', 0))
            )
            and isinstance(domain[k0].get('ind', np.r_[0]), np.ndarray)
        )
        if not c0:
            dfail[k0] = "must be a dict with keys ['ind', 'domain']"
            continue

        # vect
        domain[k0]['vect'] = vect

        # domain
        dom = domain[k0].get('domain')
        if dom is not None:
            dom, err = _check_domain(dom)
            if err is not None:
                dfail[k0] = f"value 'domain' must {err}"
                continue
            domain[k0]['domain'] = dom

        # ind
        ind = domain[k0].get('ind')
        if ind is not None:
            vsize = coll.ddata[vect]['data'].size
            if ind.dtype == bool:
                pass
            elif 'int' in ind.dtype.name:
                ind2 = np.zeros((vsize,), dtype=bool)
                ind2[ind] = True
                domain[k0]['ind'] = ind2

            if domain[k0]['ind'].size != vsize:
                msg = (
                    f"Wrong size for domain['{k0}']['ind']:\n"
                    f"\t- expected: {vsize}\n"
                    f"\t- provided: {domain[k0]['ind'].size}\n"
                    f"\n ind.dtype = {ind.dtype}"
                )
                raise Exception(msg)

    # -----------
    # errors

    if len(dfail) > 0:
        lstr = [f"\t- '{k0}': {v0}" for k0, v0 in dfail.items()]
        msg = (
            "The following domain keys / values are not conform:\n"
             + "\n".join(lstr)
        )
        raise Exception(msg)

    return domain


def _check_domain(dom=None):

    # 3 possibilities
    lc = [
        isinstance(dom, (list, tuple))
        and len(dom) == 2
        and all(np.isscalar(dd) for dd in dom)
        and dom[0] <= dom[1],
        hasattr(dom, '__iter__')
        and all(
            isinstance(dd, (list, tuple))
            and len(dd) == 2
            and all(np.isscalar(di) for di in dd)
            and dd[0] <= dd[1]
            for dd in dom
        ),
        np.isscalar(dom) or np.array(dom).size == 1,
    ]

    # adjust
    if lc[0]:
        dom = [dom]
    elif lc[2]:
        if not isinstance(dom, (float, int)):
            dom = np.array(dom).ravel()[0]
    elif not lc[1]:
        msg = "be a list of tuples or lists of len() = 2!"
        return None, msg

    return dom, None


# ########################################################
# ########################################################
#           apply domain
# ########################################################


def _set_ind_from_domain(
    vect=None,
    domain=None,
):

    # ------------
    # scalar

    if np.isscalar(domain):
        indi = np.nanargmin(np.abs(vect - domain))
        ind = np.zeros(vect.shape, dtype=bool)
        ind[indi] = True
        return ind

    # -----------------
    # sort intervals

    lin = [dd for dd in domain if isinstance(dd, list)]
    lout = [dd for dd in domain if isinstance(dd, tuple)]

    # ------------------------
    # get in for each interval

    shape_in = tuple(np.r_[len(lin), vect.shape])
    shape_out = tuple(np.r_[len(lout), vect.shape])

    # in
    ind_in = np.zeros(shape_in, dtype=bool)
    for ii, ddi in enumerate(lin):
        ind_in[ii, ...] = (vect >= ddi[0]) & (vect < ddi[1])
    ind_in = np.any(ind_in, axis=0)

    # out
    ind_out = np.zeros(shape_out, dtype=bool)
    for ii, ddi in enumerate(lout):
        ind_out[ii, ...] = (vect >= ddi[0]) & (vect < ddi[1])
    ind_out = np.any(ind_out, axis=0)

    # ------------------------
    # get overall indices

    ind = ind_in & (~ind_out)

    return ind

from pasteur.attribute import SeqAttributes
import logging

logger = logging.getLogger(__name__)


def calc_table_complexity(ver, attrs, params={}):
    c = len(attrs[None])
    h = 0

    if not params.get("no_hist", False):
        for table_sel, tattrs in attrs.items():
            if table_sel is None:
                continue

            if isinstance(tattrs, SeqAttributes) and not params.get("no_seq", False):
                if tattrs.attrs:
                    h += len(tattrs.attrs)
                if tattrs.hist:
                    if not params.get("rake", False):
                        for ha in tattrs.hist.values():
                            h += len(ha)
            else:
                h += len(tattrs)

    n = ver.rows
    return c * ((c + h) ** (1 / 3)) * (n ** (1 / 3))


def get_table_sens(ver, skip_sens=False):
    # unwrap table partition
    if hasattr(ver, "table"):
        ver = getattr(ver, "table")

    if skip_sens:
        return 1
    elif ver.max_len:
        return ver.max_len
    elif ver.children:
        return ver.children
    else:
        return 1


def calc_sens(ver, skip_sens=False, ctx=False):
    # unwrap table partition
    if hasattr(ver, "table"):
        ver = getattr(ver, "table")

    sens = get_table_sens(ver, skip_sens or ctx)

    # If we repeat the unroll, we only use 1 row per grandparent, so we
    # have to skip the sensitivity of the parent
    skip_sens_next = ctx and ver.seq_repeat

    if ver.parents:
        sens *= max([calc_sens(p, skip_sens=skip_sens_next) for p in ver.parents])

    return sens


def calc_privacy_budgets(total: float, mvers, params={}):
    complexities = {
        mver: calc_table_complexity(mver.ver, attrs, params)
        for mver, (attrs, _) in mvers.items()
    }

    MAX_COMPLEXITY = 3

    # FIXME: Dirty heuristic to avoid the medicine table having 80% of the privacy budget
    for i in range(100):
        initial_sum = sum(complexities.values())
        for mver, compl in list(complexities.items()):
            max_complexity = MAX_COMPLEXITY * (initial_sum - compl) / len(mvers)
            if compl > max_complexity:
                complexities[mver] = max_complexity
        # print(i, list(complexities.values()))
        total_complexity = sum(complexities.values())

    budgets = {}
    sensitivities = {}
    for mver, (attrs, _) in mvers.items():
        sens = calc_sens(mver.ver, ctx=mver.ctx)
        if smax := params.get("max_sens", None):
            sens = min(sens, smax)
        budget = total * (complexities[mver] / total_complexity)
        budgets[mver] = budget
        sensitivities[mver] = sens

    return budgets, sensitivities

from pasteur.attribute import SeqAttributes
import logging

logger = logging.getLogger(__name__)

def calc_table_complexity(ver, attrs, params = {}):
    c = len(attrs[None])
    h = 0

    if not params.get('no_hist', False):
        for table_sel, tattrs in attrs.items():
            if table_sel is None:
                continue

            if isinstance(tattrs, SeqAttributes) and not params.get("no_seq", False):
                if tattrs.attrs:
                    h += len(tattrs.attrs)
                if tattrs.hist:
                    if not params.get('rake', False):
                        for ha in tattrs.hist.values():
                            h += len(ha)
            else:
                h += len(tattrs)

    n = ver.rows
    return c * ((c + h) ** (1/3)) * (n ** (1/3))


def calc_sens(ver):
    # unwrap table partition
    if hasattr(ver, 'table'):
        ver = getattr(ver, 'table')

    if ver.max_len:
        sens = ver.max_len
    elif ver.children:
        sens = ver.children
    else:
        sens = 1

    if ver.parents:
        sens *= max([calc_sens(p) for p in ver.parents])

    return sens


def calc_privacy_budgets(total: float, mvers, params = {}):
    complexities = {mver: calc_table_complexity(mver.ver, attrs, params) for mver, (attrs, _) in mvers.items()}

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
        sens = calc_sens(mver.ver)
        if smax := params.get("max_sens", None):
            sens = min(sens, smax)
        budget = total * (complexities[mver] / total_complexity)
        budgets[mver] = budget
        sensitivities[mver] = sens

    return budgets, sensitivities

from pasteur.attribute import SeqAttributes


def calc_table_complexity(ver, attrs):
    c = len(attrs[None])
    h = 0
    for table_sel, tattrs in attrs.items():
        if table_sel is None:
            continue

        if isinstance(tattrs, SeqAttributes):
            if tattrs.attrs:
                h += len(tattrs.attrs)
            if tattrs.hist:
                for ha in tattrs.hist.values():
                    h += len(ha)
        else:
            h += len(tattrs)
    n = ver.rows

    return c * ((c + h) ** (1/3)) * (n ** (1/3))


def calc_sens(ver):
    if ver.max_len:
        sens = ver.max_len
    elif ver.children:
        sens = ver.children
    else:
        sens = 1

    if ver.parents:
        sens *= max([calc_sens(p) for p in ver.parents])

    return sens


def calc_privacy_budgets(total: float, mvers):
    total_complexity = sum(
        [calc_table_complexity(mver.ver, attrs) for mver, (attrs, _) in mvers.items()]
    )

    budgets = {}
    sensitivities = {}
    for mver, (attrs, _) in mvers.items():
        sens = calc_sens(mver.ver)
        complexity = calc_table_complexity(mver.ver, attrs)
        budget = total * (complexity / total_complexity)
        budgets[mver] = budget
        sensitivities[mver] = sens

    return budgets, sensitivities

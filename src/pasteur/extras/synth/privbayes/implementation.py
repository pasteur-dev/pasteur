import logging
import random
from functools import partial as ft_partial
from itertools import combinations, product
from typing import Any, NamedTuple, Sequence, cast

import numpy as np
import pandas as pd

from ....attribute import (
    Attribute,
    Attributes,
    CatValue,
    DatasetAttributes,
    Grouping,
    SeqAttributes,
    get_dtype,
)
from ....marginal import (
    ZERO_FILL,
    MarginalOracle,
    MarginalRequest,
    normalize,
    two_way_normalize,
    unpack,
)
from ....marginal.numpy import (
    AttrSelector,
    AttrSelectors,
    CalculationInfo,
    TableSelector,
)
from ....utils.progress import piter, prange, process_in_parallel

logger = logging.getLogger(__name__)

MAX_EPSILON = 1e3 - 10
MAX_T = 1e5


class Node(NamedTuple):
    attr: str
    value: str
    p: MarginalRequest
    domain: int
    partial: bool


Nodes = list[Node]


def sens_mutual_info(n: int):
    """Provides the the log2 sensitivity of the mutual information function for a given
    dataset size (n)."""
    return 2 / n * np.log2((n + 1) / 2) + (n - 1) / n * np.log2((n + 1) / (n - 1))


def calc_mutual_info(req: AttrSelectors, mar: np.ndarray, info: CalculationInfo):
    """Calculates mutual information I(X,P) for the provided data using log2."""
    j_mar, x_mar, p_mar = two_way_normalize(req, mar, info)
    mi = np.sum(j_mar * np.log2(j_mar / (np.outer(x_mar, p_mar) + ZERO_FILL)))
    return mi


def sens_r_function(n: int):
    """Provides the the R function sensitivity for a given dataset size (n)."""
    return 3 / n + 2 / (n**2)


def calc_r_function(req: AttrSelectors, mar: np.ndarray, info: CalculationInfo):
    """Calculates the R(X,P) function for the provided data."""
    j_mar, x_mar, p_mar = two_way_normalize(req, mar, info)
    r = np.sum(np.abs(j_mar - np.outer(x_mar, p_mar))) / 2
    return r


def calc_entropy(req: AttrSelectors, mar: np.ndarray, info: CalculationInfo):
    """Calculates the entropy for the provided data."""
    mar = normalize(req, mar, info)
    # TODO: check this is correct using scipy
    return -np.sum(mar * np.log2(mar))  # type: ignore


def sens_entropy(n: int):
    """Provides the sensitivity for the entropy function for a given dataset size (n)."""
    # TODO: Mathematically prove this is correct.
    return np.log2(n) / n - (n - 1) / n * np.log2((n - 1) / n)


def add_to_pset(s: tuple, x: int, h: int):
    """Given parent set `s`, adds attribute `x` with height `h`.

    Making a list is faster than ex. using an iterator."""
    y = list(s)
    y[x] = h
    return tuple(y)


def add_multiple_to_pset(s: tuple, x: Sequence[int], h: list[int]):
    """Given parent set `s`, adds attributes `x` with heights `h`.

    `x` is checked for length, so `h` may be a larger array."""

    y = list(s)

    for i in range(len(x)):
        y[x[i]] = h[i]
    return tuple(s)


def maximal_parents(domains: list[list[int]], tau: float) -> list[tuple[int, ...]]:
    """Given a set V containing hierarchical attributes (by int) and a tau
    score that is divided by the size of the domain, return a set of all
    possible combinations of attributes, such that if t > 1 there isn't an
    attribute that can be indexed in a higher level

    This is a modification of the original maximal_parents algorithm,
    that uses an unrolled counter to calculate the combinations, which
    avoids unecessary overhead incurred by the original version
    (set allocation, recursion, hashing)."""

    MULT = (1 << 16) - 1
    to_log = lambda a: int(np.log(a) * MULT)

    out = []
    heights = [len(d) for d in domains]
    length = len(domains)

    # Use log integer for tau to remove error accumulation
    ltau = to_log(tau)
    ldom = [[to_log(d) for d in ds] for ds in domains]
    ctau = ltau

    # When h is max, attr is not included, otherwise, it is included
    # with its maximum domain
    counter = list(heights)
    not_overflow = True
    is_maximal = True
    while not_overflow:
        if is_maximal:
            out.append(tuple(counter))

        is_maximal = False
        for i in range(length):
            c_i = counter[i]
            if c_i <= 0:
                # If overflow, reset to not including the height
                counter[i] = heights[i]
                ctau += ldom[i][c_i]
                # Detect overflow condition once last count goes to -1
                if i == length - 1:
                    not_overflow = False
            else:
                # Attribute was used before, so we add its tau value
                if c_i < heights[i]:
                    ctau += ldom[i][c_i]
                else:
                    is_maximal = True
                # Lower by one at least
                c_i -= 1

                # It might be the case that the domain is too high
                # for it to be included, so lower complexity until it is feasible
                while c_i >= 0 and ctau < ldom[i][c_i]:
                    c_i -= 1

                if c_i >= 0:
                    # If it was possible to include a height, update counter[i],
                    # ctau, and break
                    counter[i] = c_i
                    ctau -= ldom[i][c_i]
                    break
                else:
                    # Otherwise, we have an overflow condition and we move on to
                    # lowering the complexity of the next attribute
                    counter[i] = heights[i]
                    # Detect overflow condition once last count goes to -1
                    if i == length - 1:
                        not_overflow = False

    return out


def get_attrs(ds_attrs: DatasetAttributes, sel: TableSelector):
    if isinstance(sel, tuple):
        table_name, order = sel
        hist = cast(SeqAttributes, ds_attrs[table_name]).hist
        return hist[order]
    elif isinstance(sel, str):
        tattrs = ds_attrs[sel]
        if isinstance(tattrs, SeqAttributes):
            tattrs = tattrs.attrs
            assert tattrs is not None
        return tattrs
    else:
        return cast(Attributes, ds_attrs[None])


def calculate_attr_combinations(table: TableSelector, attr: Attribute):
    vers: list[tuple[AttrSelector, int, tuple[str, ...]]] = []
    if attr.common:
        for h in range(attr.common.height):
            sel = (table, attr.name, h)
            dom = attr.common.get_domain(h)
            vers.append((sel, dom, (attr.common.name,)))

    names = list(attr.vals)
    heights = product(
        *[
            range(-1, v.height - (1 if attr.common else 0))
            for v in attr.vals.values()
            if isinstance(v, CatValue)
        ]
    )

    for combo in heights:
        sel = {n: h for n, h in zip(names, combo) if h > -1}
        if not sel:
            continue
        dom = CatValue.get_domain_multiple(
            list(sel.values()),
            [cast(CatValue, attr[n]) for n in sel],
        )
        deps = tuple(sel)
        vers.append(((table, attr.name, sel), dom, deps))

    return sorted(vers, key=lambda c: c[1])


def calculate_attrs_combinations(table: TableSelector, attrs: Attributes):
    combos = {
        (table, n): calculate_attr_combinations(table, a)
        for n, a in attrs.items()
        if a.vals
    }
    val_to_idx = []
    for attr in attrs.values():
        for val in attr.vals:
            val_to_idx.append((table, val))
        if attr.common:
            val_to_idx.append((table, attr.common.name))
    return combos, val_to_idx


def greedy_bayes(
    oracle: MarginalOracle,
    ds_attrs: DatasetAttributes,
    n: int,
    e1: float | None,
    e2: float | None,
    theta: float,
    use_r: bool,
    unbounded_dp: bool,
    random_init: bool,
    prefer_table: str | None = None,
    rake: bool = True
) -> tuple[Nodes, float]:
    """Performs the greedy bayes algorithm for variable domain data.

    Supports variable e1, e2, where in the paper they are defined as
    `e1 = b * e` and `e2 = (1 - b) * e`, variable theta, and both
    mutual information and R functions.

    Binary domains are not supported due to computational intractability."""

    calc_fun = calc_r_function if use_r else calc_mutual_info
    sens_fun = sens_r_function if use_r else sens_mutual_info

    #
    # Set up maximal parents algorithm
    # - with unrolled recursion into while + for loop
    # - unroll attribute combinations into one attribute for improved perf.
    # - sort by domain. Combinations with higher domain more maximal.
    # - Not necessarily true, one value might be higher in one combination, one in other.
    # - Prevents complex logic in maximal parents
    #

    # Find attribute combinations
    combos = {}
    val_idx = []
    for t, table_attrs in ds_attrs.items():
        if isinstance(table_attrs, SeqAttributes):
            if table_attrs.attrs:
                tcombos, tval_to_idx = calculate_attrs_combinations(
                    t, table_attrs.attrs
                )
                combos.update(tcombos)
                val_idx.extend(tval_to_idx)

            if not prefer_table or prefer_table == t:
                for s, hist in table_attrs.hist.items():
                    assert isinstance(t, str)
                    tcombos, tval_to_idx = calculate_attrs_combinations((t, s), hist)
                    combos.update(tcombos)
                    val_idx.extend(tval_to_idx)
        else:
            tcombos, tval_to_idx = calculate_attrs_combinations(t, table_attrs)
            combos.update(tcombos)
            val_idx.extend(tval_to_idx)

    #
    # Implement misc functions for summating the scores
    #
    score_cache = {}

    def calc_candidate_scores(
        candidates: list[tuple[str, MarginalRequest, tuple[str, tuple[int, ...]]]]
    ):
        """Calculates the mutual information approximation score for each candidate
        marginal based on `calc_fun`"""

        # Split marginals into already processed and to be processed
        scores = np.empty((len(candidates)), dtype="float")
        cached = np.zeros((len(candidates)), dtype="bool")

        requests = []
        for i, (x, pset, cand_hash) in enumerate(candidates):
            if cand_hash in score_cache:
                scores[i] = score_cache[cand_hash]
                cached[i] = True
                continue

            # Convert parent selector to marginal by adding x to the end
            mar = list(pset)
            mar.append((None, val_to_attr[x], {x: 0}))
            requests.append(mar)

        # Process new ones
        new_mar = np.sum(~cached)
        all_mar = len(candidates)
        if new_mar > 0:
            new_scores: list[float] = oracle.process(
                requests,
                desc=f"Calculating {new_mar}/{all_mar} ({all_mar/new_mar:.1f}x w/ cache) marginals",
                postprocess=calc_fun,
            )
        else:
            new_scores = []

        # Update cache
        scores[~cached] = new_scores
        for i, (x, pset, cand_hash) in enumerate(candidates):
            if not cached[i]:
                score_cache[cand_hash] = scores[i]

        return scores

    def pick_candidate(
        candidates: list[tuple[str, MarginalRequest, tuple[str, tuple[int, ...]]]]
    ) -> tuple[str, MarginalRequest]:
        """Selects a candidate based on the exponential mechanism by calculating
        all of their scores first."""
        vals = np.array(calc_candidate_scores(candidates))

        # If e1 is bigger than max_epsilon, assume it's infinite.
        if e1 is None or e1 > MAX_EPSILON:
            return candidates[int(np.argmax(vals))][:2]

        # np.exp is unstable for large vals
        # subtract max (taken from original source)
        # doesn't affect probabilities
        vals -= vals.max()

        delta = (d - n_chosen) * sens_fun(n) / e1
        p = np.exp(vals / 2 / delta)
        p /= p.sum()

        choice = np.random.choice(len(candidates), size=1, p=p)[0]

        return candidates[choice][:2]

    #
    # Implement greedy bayes (as shown in the paper)
    #
    attrs = cast(Attributes, ds_attrs[None])
    todo = []
    val_to_attr = {}
    for name, attr in attrs.items():
        todo.extend(list(attr.vals))
        for val in attr.vals:
            val_to_attr[val] = name
        if attr.common:
            val_to_attr[attr.common.name] = name

    d = len(todo)
    EMPTY_HASH = tuple(-1 for _ in range(len(val_idx)))

    if len(ds_attrs) > 1:
        x1 = -1
    elif random_init:
        x1 = random.choice(range(d))
    else:
        # Pick x1 based on entropy
        # consumes some privacy budget, but starting with a bad choice can lead
        # to a bad network.
        reqs: list[MarginalRequest] = [[(None, val_to_attr[v], {v: 0})] for v in todo]

        vals = list(
            oracle.process(
                reqs,
                desc="Choosing first node based on entropy",
                postprocess=calc_entropy,
            ),
        )
        vals = np.array(vals)
        vals -= vals.max()

        delta = d * sens_entropy(n) / (e1 if e1 is not None else 1)
        p = np.exp(vals / 2 / delta)
        p /= p.sum()

        if e1 is None or e1 > MAX_EPSILON:
            x1 = np.array(range(d))[np.argmax(p)]
        else:
            x1: int = np.random.choice(range(d), size=1, p=p)[0]

    if x1 != -1:
        val = todo.pop(x1)
        generated = [val]
        generated_attrs = set([val_to_attr[val]])
        n_chosen = 1 if not random_init else 0
        nodes = [
            Node(
                val_to_attr[val],
                val,
                [],
                cast(CatValue, attrs[val_to_attr[val]][val]).domain,
                False,
            )
        ]
    else:
        generated = []
        generated_attrs = set()
        n_chosen = 0
        nodes = []

    # Calculate theta
    t = (n * (e2 if e2 is not None else 1)) / ((1 if unbounded_dp else 2) * d * theta)

    # Allow for "unbounded" privacy budget without destroying the computer.
    if e1 is None or e1 > MAX_EPSILON:
        logger.warning("Baking without DP (e1=inf).")
    if t > n / 10 or e2 is None or e2 > MAX_EPSILON or t > MAX_T:
        t = min(n / 10, MAX_T)
        logger.warning(
            f"Considering e2={e2} unbounded, t will be bound to min(n/10, {MAX_T:.0e})={t:.2f} for computational reasons."
        )

    first = True
    for _ in prange(len(todo), desc="Finding Nodes: "):
        candidates: list[tuple[str, MarginalRequest, tuple[str, tuple[int, ...]]]] = []
        if d > 30:
            if first:
                logger.error("Too many columns, disabling parent correlations.")
                first = False

            info = [(x, []) for x in todo]
            node_psets = [[] for _ in range(len(todo))]
        else:
            base_args = {}
            per_call_args = []
            info = []

            for x in todo:
                domains = []
                selectors = []
                x_dom = cast(CatValue, attrs[val_to_attr[x]][x]).domain
                new_tau = t / x_dom

                # Create customized domain, with relevant selectors, by
                # Removing combinations with unmet dependencies
                for name, attr_combos in combos.items():
                    if (
                        rake
                        and name[0]
                        and len(name[0]) == 2
                        and name[1] != val_to_attr[x]
                    ):
                        # Skip seq attrs that are not the same column
                        continue
                    doms = []
                    sels = []
                    for sel, dom, deps in attr_combos:
                        # For each combination, check that its dependencies are met
                        # This is the case when a dependency is not in todo (generated values)
                        # and its attribute has been partially generated (common values)
                        deps_met = True
                        if sel[0] == None:
                            for dep in deps:
                                if dep in todo or not val_to_attr[dep] in generated_attrs:
                                    deps_met = False
                                    break

                        if deps_met:
                            if name[0] is None and name[1] == val_to_attr[x]:
                                val_sel = sel[-1]
                                if isinstance(val_sel, int):
                                    # If val_sel is an int, we're sampling the common value
                                    # on x already, making this an invalid combination
                                    deps_met = False
                                else:
                                    # There needs to be a common value
                                    attr = attrs[name[1]]
                                    cmn = attr.common
                                    # Adjust domain if x is in the same attribute
                                    # Find full domain when including x for attribute
                                    # Divide by x's domain
                                    # This is not a representative domain, but will
                                    # be equivalent in the `maximal_parents` computation
                                    # as tau /= x_dom
                                    if isinstance(cmn, CatValue):
                                        full_dom = cmn.get_domain_multiple(
                                            [*val_sel.values(), 0],
                                            [
                                                *[cast(CatValue, attr[v]) for v in val_sel],
                                                cast(CatValue, attr[x]),
                                            ],
                                        )
                                        dom = full_dom // x_dom

                        if deps_met:
                            doms.append(dom)
                            sels.append(sel)

                    if doms and sels:
                        domains.append(doms)
                        selectors.append(sels)
                per_call_args.append({"tau": new_tau, "domains": domains})
                info.append((x, selectors))

            node_psets = process_in_parallel(
                maximal_parents,
                per_call_args,
                base_args,
                desc="Finding Maximal Parent sets",
            )

        for (val, sels), psets in zip(info, node_psets):
            for pset in psets:
                cand_hash = list(EMPTY_HASH)

                mar: MarginalRequest = []
                for i, h in enumerate(pset):
                    if h < len(sels[i]):
                        sel = sels[i][h]
                        mar.append(sel)

                        # Create custom hash which is a tuple with an integer
                        # indicating which values are used and with what heights
                        table, aname, phs = sel
                        if isinstance(phs, dict):
                            for p, ph in phs.items():
                                cand_hash[val_idx.index((table, p))] = ph
                        else:
                            cmn = get_attrs(ds_attrs, table)[aname].common
                            assert cmn is not None
                            cand_hash[val_idx.index((table, cmn.name))] = phs
                candidates.append((val, mar, (val, tuple(cand_hash))))
            if not psets:
                candidates.append((val, [], (val, EMPTY_HASH)))

        x, pset = pick_candidate(candidates)
        attr = val_to_attr[x]
        generated.append(x)
        todo.remove(x)

        nodes.append(
            Node(
                attr=attr,
                value=x,
                p=pset,
                domain=cast(CatValue, attrs[attr][x]).domain,
                partial=attr in generated_attrs,
            )
        )
        generated_attrs.add(attr)

    return nodes, t


def to_str(a: str | tuple):
    if isinstance(a, str):
        return a

    return "_".join(map(str, a))


def print_tree(
    attrs: DatasetAttributes,
    nodes: Nodes,
    e1: float,
    e2: float,
    theta: float,
    t: float,
):
    s = f"Bayesian Network Tree:\n"
    e1 = e1 or -1
    e2 = e2 or -1
    s += f"(PrivBayes e1={e1:.5f}, e2={e2:.5f}, theta={theta:.2f}, available t={t:.2f})"

    pset_len = 57

    tlen = len(" Multi-Val Attrs")
    for _, x, _, _, _ in nodes:
        if len(x) > tlen:
            tlen = len(x)

    for name in cast(Attributes, attrs[None]).keys():
        al = to_str(name)
        if len(al) > tlen:
            tlen = len(al)

    tlen += 1

    s += f"\n┌{'─'*(tlen+1)}┬──────┬──────────┬{'─'*pset_len}┐"
    s += f"\n│{'Value Nodes'.rjust(tlen)} │  Dom │ Avail. t │ Attribute Parents{' '*(pset_len - 18)}│"
    s += f"\n├{'─'*(tlen+1)}┼──────┼──────────┼{'─'*pset_len}┤"
    for x_attr, x, p, domain, partial in nodes:
        # Show * when using a reduced marginal + correct domain
        # cmn_val = attrs[x_attr].common
        # common = cmn_val.get_domain(0) if cmn_val else 0
        dom = f"{domain:>4d}"

        # Print Line start
        s += f"\n│{x.rjust(tlen)} │ {dom}{'*' if partial else ' '}│ {t/domain:>8.2f} │"

        # Print Parents
        line_str = ""
        for parent in p:
            if len(parent) == 3:
                table, p_name, attr_sel = parent
            else:
                p_name, attr_sel = parent
                table = None

            p_str = " "

            if isinstance(table, tuple):
                table_name, order = table
                hist = cast(SeqAttributes, attrs[table[0]]).hist
                tattrs = hist[order]
                p_str += f"{table_name}[-{1 + order}]."
            elif isinstance(table, str):
                p_str += f"{table}."
                tattrs = attrs[table]
                if isinstance(tattrs, SeqAttributes):
                    tattrs = tattrs.attrs
                    assert tattrs is not None
            else:
                tattrs = cast(Attributes, attrs[None])

            p_str += f"{p_name}["
            if isinstance(attr_sel, dict):
                for col in tattrs[p_name].vals:
                    if col in attr_sel:
                        p_str += str(attr_sel[col])
                    else:
                        p_str += "."
            else:
                p_str += f"c{attr_sel}"
            p_str += "]"

            if len(p_str) + len(line_str) >= pset_len:
                s += f"{line_str:57s}│"
                s += f"\n│{' '*(tlen+1)}│      │          │"
                line_str = f" >{p_str}"
            else:
                line_str += p_str

        s += f"{line_str:57s}│"

    # Skip multi-col attr printing if there aren't any of them.
    tattrs = cast(Attributes, attrs[None])
    if not any(len(attr.vals) > 1 for attr in tattrs.values()):
        s += f"\n└{'─'*(tlen+1)}┴──────┴──────────┴{'─'*pset_len}┘"
        return s

    # Print mutli-column attrs
    s += f"\n├{'─'*(tlen+1)}┼──────┼──────────┴{'─'*pset_len}┤"
    s += f"\n│{'Multi-Val Attrs'.rjust(tlen)} │  Cmn │ Values    {' '*pset_len}│"
    s += f"\n├{'─'*(tlen+1)}┼──────┼───────────{'─'*pset_len}┤"

    for name, attr in tattrs.items():
        cols = attr.vals
        if len(cols) <= 1:
            continue

        if attr.common:
            cmn = str(attr.common.domain)
        else:
            cmn = "NIL"
        s += f"\n│{to_str(name).rjust(tlen)} │ {cmn:>4s} │"
        line_str = ""
        for i, col in enumerate(cols):
            c_str = f" {col}"

            if len(c_str) + len(line_str) >= pset_len + 11:
                s += f"{line_str:57s}│"
                s += f"\n│{' '*(tlen+1)}│     │"
                line_str = f" >{c_str}"
            else:
                line_str += c_str

        s += f"{line_str:68s}│"

    s += f"\n└{'─'*(tlen+1)}┴──────┴───────────{'─'*pset_len}┘"
    return s


def calc_noisy_marginals(
    oracle: MarginalOracle,
    nodes: Nodes,
    noise_scale: float,
    skip_zero_counts: bool,
):
    """Calculates the marginals and adds laplacian noise with scale `noise_scale`."""
    requests = []
    for x_attr, x, p, _, _ in nodes:
        mar = list(p)
        mar.append((None, x_attr, {x: 0}))
        requests.append(mar)

    marginals = oracle.process(
        requests,
        desc="Calculating noisy marginals.",
        postprocess=unpack,
    )

    noised_marginals = []
    for (x_attr, x, p, _, _), marginal in zip(nodes, marginals):
        noise = cast(
            np.ndarray, np.random.laplace(scale=noise_scale, size=marginal.shape)
        )
        marginal /= marginal.sum()

        if skip_zero_counts:
            # Skip adding noise to zero counts
            noise[marginal == 0] = 0

        noised_marginal = (marginal + noise).clip(0)
        noised_marginal /= noised_marginal.sum()
        noised_marginals.append(noised_marginal)

    return noised_marginals


def sample_rows(
    idx: pd.Index,
    attrs: DatasetAttributes,
    hist: dict[TableSelector, pd.DataFrame],
    nodes: list[Node],
    marginals: list[np.ndarray],
) -> pd.DataFrame:
    out_cols = {}
    n = len(idx)

    attr_sampled_cols: dict[str, str] = {}
    for (x_attr, x, p, domain, partial), marginal in piter(
        zip(nodes, marginals),
        total=len(nodes),
        desc="Sampling values sequentially",
        leave=False,
    ):
        if len(p) == 0:
            # No parents = use 1-way marginal
            # Concatenate m to avoid N dimensions and use lookup table to recover
            m = marginal.reshape(-1)
            try:
                out_col = np.random.choice(domain, size=n, p=m)
            except ValueError as e:
                logger.warning(
                    f"Received error when sampling probabilities, picking at random:\n{e}"
                )
                out_col = np.random.randint(domain, size=n)
            out_col = out_col.astype(get_dtype(domain))
        else:
            # Use conditional probability
            # Reshape marginal to one dimension for p, x
            domains = tuple(reversed(marginal.shape[:-1]))
            marginal = marginal.reshape((-1, marginal.shape[-1]))
            # Get groups for marginal
            i = 0
            mul = 1
            dtype = get_dtype(marginal.shape[0] * marginal.shape[1])
            _sum_nd = np.zeros((n,), dtype=dtype)
            _tmp_nd = np.zeros((n,), dtype=dtype)
            for parent in reversed(p):
                if len(parent) == 3:
                    table, attr_name, sel = parent
                else:
                    attr_name, sel = parent
                    table = None
                tattrs = get_attrs(attrs, table)

                if isinstance(sel, dict):
                    l_mul = 1
                    for val, h in reversed(sel.items()):
                        meta = cast(
                            CatValue,
                            tattrs[attr_name].vals[val],
                        )
                        mapping = np.array(meta.get_mapping(h), dtype=dtype)
                        domain = meta.get_domain(h)

                        if table is None:
                            pcol = out_cols[val]
                        else:
                            pcol = hist[table][val]
                        col_lvl = mapping[pcol]
                        np.multiply(col_lvl, mul * l_mul, out=_tmp_nd, dtype=dtype)
                        np.add(_sum_nd, _tmp_nd, out=_sum_nd, dtype=dtype)
                        l_mul *= domain

                        assert domain == domains[i], "Domain mismatch"
                        i += 1
                else:
                    # Find common data
                    attr = tattrs[attr_name]
                    cmn = attr.common
                    assert cmn is not None
                    mapping = np.array(cmn.get_mapping(sel), dtype=dtype)

                    # Derive common column
                    cmn_col = None
                    if table is None:
                        for nc, c in out_cols.items():
                            if nc in attr.vals:
                                meta = attr[nc]
                                assert isinstance(meta, CatValue)
                                cmn_col = meta.get_mapping(meta.height - 1)[c]
                                break
                    else:
                        nc, meta = next(iter(attr.vals.items()))
                        meta = attr[nc]
                        assert isinstance(meta, CatValue)
                        cmn_col = meta.get_mapping(meta.height - 1)[hist[table][nc]]
                        
                    assert cmn_col is not None

                    # Apply common col
                    col_lvl = mapping[cmn_col]
                    np.multiply(col_lvl, mul, out=_tmp_nd, dtype=dtype)
                    np.add(_sum_nd, _tmp_nd, out=_sum_nd, dtype=dtype)

                    l_mul = cmn.get_domain(sel)
                    assert l_mul == domains[i], "Domain mismatch"
                    i += 1
                mul *= l_mul

            # Add common group from sampled parent if exists
            attr = cast(Attributes, attrs[None])[x_attr]
            cmn = attr.common
            if partial and cmn:
                # Find common data
                mapping = np.array(cmn.get_mapping(0), dtype=dtype)

                # Grab parent col
                cmn_col = None
                for nc, c in out_cols.items():
                    if nc in attr.vals:
                        meta = attr[nc]
                        assert isinstance(meta, CatValue)
                        cmn_col = meta.get_mapping(meta.height - 1)[c]
                        break
                assert cmn_col is not None

                # Apply to sum
                col_lvl = mapping[cmn_col]
                np.multiply(col_lvl, mul, out=_tmp_nd, dtype=dtype)
                np.add(_sum_nd, _tmp_nd, out=_sum_nd, dtype=dtype)

            # Sample groups
            tattrs = cast(Attributes, attrs[None])
            x_domain = cast(CatValue, tattrs[x_attr][x]).domain
            out_col = np.zeros((n,), dtype=get_dtype(x_domain))
            groups = _sum_nd

            # Group rows based on parent values + co-dependent common value
            for group in np.unique(groups):
                # When common was applied `mul` was not modified
                # So the parent and common groups can be separated with modulo
                parent_group = group % mul
                common_group = group // mul

                m = marginal
                m_g = m[parent_group, :]  # m[:, group]

                # If we have sampled the common value, mask the marginal to allow only common values
                if partial and cmn:
                    cv = cast(CatValue, tattrs[x_attr][x])
                    m_g = m_g * (cv.get_mapping(cv.height - 1) == common_group)

                # FIXME: find sampling strategy for this
                m_sum = m_g.sum()
                if m_sum < 1e-6:
                    # If the sum of the group is zero, there are no samples
                    # Use the average probability of the variable
                    m_avg = (
                        marginal.sum(axis=0) / marginal.sum()
                    )  # marginal.sum(axis=1) / marginal.sum()
                    m_g = m_avg
                else:
                    # Otherwise normalize
                    m_g = m_g / m_sum

                size = np.count_nonzero(groups == group)
                out_col[groups == group] = np.random.choice(x_domain, size=size, p=m_g)

        # Output column
        out_cols[x] = out_col
        attr_sampled_cols[x_attr] = x

    return pd.DataFrame(out_cols, index=idx)

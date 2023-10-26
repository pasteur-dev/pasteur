""" This module contains functions for post-processing marginals.
"""

import numpy as np
from .numpy import AttrSelectors, CalculationInfo

ZERO_FILL = 1e-24


def unpack(req: AttrSelectors, mar: np.ndarray, info: CalculationInfo):
    """Undoes the marginal packing done by the marginal calculation algorithm to save space."""
    packed_doms = []
    full_doms = []
    val_doms = []
    commons = []

    for (table, attr, sel) in req:
        packed_dom = 1
        full_dom = 1
        attr_dom = []

        if isinstance(sel, dict):
            common = info.common[(table, attr)]

            for n, h in sel.items():
                v_dom = info.domains[(table, n)][h]
                full_dom *= v_dom
                packed_dom *= v_dom - common
                attr_dom.append(v_dom)

            commons.append(common)
            full_doms.append(full_dom)
            packed_doms.append(packed_dom + common)
            val_doms.append(attr_dom)
        else:
            commons.append(0)
            v_dom = info.domains[(table, info.common_names[(table, attr)])][sel]
            packed_doms.append(v_dom)
            full_doms.append(v_dom)
            val_doms.append(None)

    out = np.zeros(full_doms)
    slices = [slice(d) for d in packed_doms]
    out[slices] = mar.reshape(packed_doms)
    del mar

    for i, (common, val, packed) in enumerate(zip(commons, val_doms, packed_doms)):
        if not common:
            continue

        write_slices = tuple([
            slice(-packed + common, None) if j == i else slice(None)
            for j in range(len(packed_doms))
        ])
        read_slices = tuple([
            slice(common, packed) if j == i else slice(None)
            for j in range(len(packed_doms))
        ])
        out[write_slices] = out[read_slices]

        ofs = sum(val[:-1])
        for k in reversed(range(1, common)):
            write_slices = tuple([
                slice(ofs * k, ofs * k + 1) if j == i else slice(None)
                for j in range(len(packed_doms))
            ])
            read_slices = tuple([
                slice(k, k + 1) if j == i else slice(None)
                for j in range(len(packed_doms))
            ])
            out[write_slices] = out[read_slices]

    expanded_doms = []
    for v in val_doms:
        expanded_doms.extend(v)
    return out.reshape(expanded_doms)


def two_way_normalize(
    req: AttrSelectors,
    mar: np.ndarray,
    info: CalculationInfo,
    zero_fill: float | None = ZERO_FILL,
):
    table, attr, sel = req[-1]

    if isinstance(sel, dict):
        dom = 1
        common = info.common[(table, attr)]

        for n, h in sel.items():
            v_dom = info.domains[(table, n)][h]
            dom *= v_dom - common

        dom += common
    else:
        dom = info.domains[(table, info.common_names[(table, attr)])][sel]

    margin = mar.reshape((-1, dom)).astype("float32")

    margin /= margin.sum()
    if zero_fill is not None:
        # Mutual info turns into NaN without this
        margin += zero_fill

    j_mar = margin
    x_mar = np.sum(margin, axis=1)
    p_mar = np.sum(margin, axis=0)

    return j_mar, x_mar, p_mar


def normalize(
    req: AttrSelectors,
    mar: np.ndarray,
    info: CalculationInfo,
    zero_fill: float | None = ZERO_FILL,
):
    margin = mar.astype("float32")

    margin /= margin.sum()
    if zero_fill is not None:
        # Mutual info turns into NaN without this
        margin += zero_fill

    return margin

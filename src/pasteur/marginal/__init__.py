# from .numpy import *

from typing import NamedTuple

import numpy as np

from ..attribute import Attributes
from ..utils import LazyChunk, LazyFrame
from ..utils.progress import piter, process_in_parallel
from .memory import load_from_memory, map_to_memory
from .numpy import (
    ZERO_FILL,
    AttrSelector,
    AttrSelectors,
    expand_table,
    get_domains,
    postprocess,
    postprocess_1way,
)

try:
    from .native_py import calc_marginal, calc_marginal_1way

except Exception as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.error(
        f"Failed importing native marginal implementation, using numpy instead (2-8x slower). Error:\n{e}"
    )

    from .numpy import calc_marginal, calc_marginal_1way


class MarginalRequest(NamedTuple):
    x: AttrSelector | None
    p: AttrSelectors
    partial: bool


def _marginal_worker(
    mem_cols,
    info_cols,
    mem_noncommon,
    info_noncommon,
    domains,
    req: MarginalRequest,
    data: np.ndarray | None = None,
):
    cols = load_from_memory(mem_cols, info_cols)
    cols_noncommon = load_from_memory(mem_noncommon, info_noncommon)

    x, p, partial = req

    if x is None:
        return calc_marginal_1way(cols, cols_noncommon, domains, p, data)
    else:
        return calc_marginal(cols, cols_noncommon, domains, x, p, partial, data)


class MarginalOracle:
    def __init__(
        self, attrs: Attributes, data: LazyFrame, batched: bool = True
    ) -> None:
        self.attrs = attrs
        self.data = data
        self.batched = batched and data.partitioned

        self._loaded = False

    def get_domains(self):
        return get_domains(self.attrs)

    def get_shape(self):
        return self.data.shape

    def _process_batch(
        self,
        data: LazyChunk,
        requests: list[MarginalRequest],
        old_data: list[np.ndarray] | None = None,
    ):
        # Load data
        if self.batched or not self._loaded:
            df = data()
            cols, cols_noncommon, domains = expand_table(self.attrs, df)
            del df

            mem_cols, info_cols = map_to_memory(cols)
            del cols
            mem_noncommon, info_noncommon = map_to_memory(cols_noncommon)
            del cols_noncommon

            if not self.batched:
                self._cache = (
                    mem_cols,
                    info_cols,
                    mem_noncommon,
                    info_noncommon,
                    domains,
                )
                self._loaded = True
        else:
            # Allow using cached data
            mem_cols, info_cols, mem_noncommon, info_noncommon, domains = self._cache

        base_args = {
            "mem_cols": mem_cols,
            "info_cols": info_cols,
            "mem_noncommon": mem_noncommon,
            "info_noncommon": info_noncommon,
            "domains": domains,
        }

        if old_data:
            per_call_args = [
                {"req": req, "data": old} for req, old in zip(requests, old_data)
            ]
        else:
            per_call_args = [{"req": req} for req in requests]

        res = process_in_parallel(
            _marginal_worker, per_call_args, base_args, 100, "Calculating marginals"
        )

        if not self.batched:
            mem_cols.close()
            mem_cols.unlink()

            mem_noncommon.close()
            mem_noncommon.unlink()

        return res

    def _postprocess(self, requests: list[MarginalRequest], data: list[np.ndarray]):
        out = []
        for req, res in zip(requests, data):
            if req.x is not None:
                out.append(postprocess(res))
            else:
                out.append(postprocess_1way(res))
        return out

    def process(
        self, requests: list[MarginalRequest], desc: str = "Processing partition"
    ):
        if not self.batched:
            res = self._process_batch(self.data, requests)
            return self._postprocess(requests, res)

        old = None
        for batch in piter(self.data.values(), total=len(self.data), desc=desc, leave=False):
            old = self._process_batch(batch, requests, old)

        assert old is not None
        return self._postprocess(requests, old)

    def close(self):
        if self.batched or not self._loaded:
            return

        mem_cols, _, mem_noncommon, _, _ = self._cache
        mem_cols.close()
        mem_cols.unlink()

        mem_noncommon.close()
        mem_noncommon.unlink()


__all__ = [
    "AttrSelector",
    "AttrSelectors",
    "MarginalOracle",
    "MarginalRequest",
    "ZERO_FILL",
]

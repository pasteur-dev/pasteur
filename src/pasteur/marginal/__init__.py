# from .numpy import *

from typing import NamedTuple
from threading import Thread
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


def _marginal_initializer(base_args, per_call_args):
    cols = load_from_memory(base_args["mem_cols"], base_args["info_cols"])
    cols_noncommon = load_from_memory(
        base_args["mem_noncommon"], base_args["info_noncommon"]
    )

    new_base_args = base_args.copy()
    new_base_args["cols"] = cols
    new_base_args["cols_noncommon"] = cols_noncommon

    return new_base_args, per_call_args


def _marginal_worker(
    cols,
    cols_noncommon,
    domains,
    req: MarginalRequest,
    data: np.ndarray | None = None,
    **_,
) -> np.ndarray:
    x, p, partial = req

    if x is None:
        return calc_marginal_1way(cols, cols_noncommon, domains, p, data)
    else:
        return calc_marginal(cols, cols_noncommon, domains, x, p, partial, data)


def _marginal_finalizer(base_args, per_call_args):
    base_args["mem_cols"].close()
    base_args["mem_noncommon"].close()


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

    def _load_batch(self, data: LazyChunk):
        df = data()
        cols, cols_noncommon, domains = expand_table(self.attrs, df)
        del df

        self.new_domains = domains
        self.new_mem_cols, self.new_info_cols = map_to_memory(cols)
        del cols
        self.new_mem_noncommon, self.new_info_noncommon = map_to_memory(cols_noncommon)
        del cols_noncommon

    def _swap_batch(self):
        self.domains = self.new_domains
        self.mem_cols = self.new_mem_cols
        self.info_cols = self.new_info_cols
        self.mem_noncommon = self.new_mem_noncommon
        self.info_noncommon = self.new_info_noncommon

        del self.new_domains
        del self.new_mem_cols
        del self.new_info_cols
        del self.new_mem_noncommon
        del self.new_info_noncommon

    def _process_batch(
        self,
        requests: list[MarginalRequest],
        old_data: list[np.ndarray] | None = None,
    ):
        base_args = {
            "mem_cols": self.mem_cols,
            "info_cols": self.info_cols,
            "mem_noncommon": self.mem_noncommon,
            "info_noncommon": self.info_noncommon,
            "domains": self.domains,
        }

        if old_data:
            per_call_args = [
                {"req": req, "data": old} for req, old in zip(requests, old_data)
            ]
        else:
            per_call_args = [{"req": req} for req in requests]

        res = process_in_parallel(
            _marginal_worker,
            per_call_args,
            base_args,
            5,
            desc="Calculating marginals",
            initializer=_marginal_initializer,
            finalizer=_marginal_finalizer,
        )

        return res

    def _unload_batch(self):
        if not self.batched:
            return

        self.mem_cols.close()
        self.mem_cols.unlink()

        self.mem_noncommon.close()
        self.mem_noncommon.unlink()

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
            if not self._loaded:
                self._load_batch(self.data)
                self._swap_batch()
                self._loaded = True

            res = self._process_batch(requests)
            return self._postprocess(requests, res)

        old = None
        partitions = list(self.data.values())
        t = Thread(target=self._load_batch, args=(partitions[0],))
        t.start()

        for i in piter(range(len(partitions)), desc=desc, leave=False):
            t.join()
            self._swap_batch()
            if i < len(partitions) - 1:
                t = Thread(target=self._load_batch, args=(partitions[i + 1],))
                t.start()

            old = self._process_batch(
                requests,
                old,
            )
            self._unload_batch()

        assert old is not None
        return self._postprocess(requests, old)

    def close(self):
        if self.batched or not self._loaded:
            return

        self._unload_batch()


__all__ = [
    "AttrSelector",
    "AttrSelectors",
    "MarginalOracle",
    "MarginalRequest",
    "ZERO_FILL",
]

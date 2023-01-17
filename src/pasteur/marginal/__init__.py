# from .numpy import *

import logging
from typing import NamedTuple

import numpy as np

from ..attribute import Attributes
from ..utils import LazyChunk, LazyFrame
from ..utils.progress import piter, process_in_parallel, process_async, IS_SUBPROCESS
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

logger = logging.getLogger(__name__)

from .numpy import calc_marginal as calc_marginal_np

try:
    from .native_py import calc_marginal, calc_marginal_1way
except Exception as e:
    logger.error(
        f"Failed importing native marginal implementation, using numpy instead (2-8x slower). Error:\n{e}"
    )
    from .numpy import calc_marginal, calc_marginal_1way


class MarginalRequest(NamedTuple):
    x: AttrSelector | None
    p: AttrSelectors
    partial: bool
    zero_fill: float = ZERO_FILL


def _marginal_initializer(base_args, per_call_args):
    cols = load_from_memory(base_args["mem_cols"], base_args["info_cols"])
    cols_noncommon = load_from_memory(
        base_args["mem_noncommon"], base_args["info_noncommon"]
    )

    new_base_args = base_args.copy()
    new_base_args["cols"] = cols
    new_base_args["cols_noncommon"] = cols_noncommon

    if base_args["mem_marginals"] is not None:
        marginals = load_from_memory(
            base_args["mem_marginals"], base_args["info_marginals"]
        )
        new_base_args["marginals"] = marginals["data"]
    else:
        new_base_args["marginals"] = None

    return new_base_args, per_call_args


def _marginal_worker(
    cols,
    cols_noncommon,
    domains,
    req: MarginalRequest,
    i: int,
    marginals: list[np.ndarray] | None,
    **_,
) -> np.ndarray | None:
    x, p, partial, _ = req

    if marginals is not None:
        data = marginals[i]
    else:
        data = None

    if x is None:
        data = calc_marginal_1way(cols, cols_noncommon, domains, p, data)
    else:
        if partial:
            # Native implementation is unfinished
            data = calc_marginal_np(cols, cols_noncommon, domains, x, p, partial, data)
        else:
            data = calc_marginal(cols, cols_noncommon, domains, x, p, partial, data)

    if marginals is None:
        return data


def _marginal_finalizer(base_args, per_call_args):
    if not IS_SUBPROCESS:
        return

    base_args["mem_cols"].close()
    base_args["mem_noncommon"].close()
    if base_args["marginals"] is not None:
        base_args["marginals"].close()


def _marginal_parallel_worker(
    requests: list[MarginalRequest], attrs: Attributes, chunk: LazyChunk
):
    cols, cols_noncommon, domains = expand_table(attrs, chunk())

    out = []
    for x, p, partial, _ in requests:
        if x is None:
            out.append(calc_marginal_1way(cols, cols_noncommon, domains, p))
        else:
            if partial:
                # Native implementation is unfinished
                out.append(
                    calc_marginal_np(cols, cols_noncommon, domains, x, p, partial)
                )
            else:
                out.append(calc_marginal(cols, cols_noncommon, domains, x, p, partial))

    return out


def _marginal_loader(attrs, data):
    import pandas as pd

    if isinstance(data, list):
        df = pd.concat([d() for d in data])
    else:
        df = data()
    cols, cols_noncommon, domains = expand_table(attrs, df)
    del df

    domains = domains
    mem_cols, info_cols = map_to_memory(cols)
    del cols
    mem_noncommon, info_noncommon = map_to_memory(cols_noncommon)
    del cols_noncommon

    return domains, mem_cols, info_cols, mem_noncommon, info_noncommon


class MarginalOracle:
    def __init__(
        self,
        attrs: Attributes,
        data: LazyFrame,
        batched: bool = True,
        sequential_min: int = 1000,
        sequential_chunks: int = 1,
        min_chunk_size: int = 100,
        max_worker_mult: int = 3,
    ) -> None:
        self.attrs = attrs
        self.data = data
        self.batched = batched and data.partitioned
        self.sequential_min = sequential_min
        self.sequential_chunks = sequential_chunks
        self.min_chunk_size = min_chunk_size
        self.max_worker_mult = max_worker_mult

        self._loaded = False
        self.counts = None

        self.mem_marginals = None
        self.info_marginals = None

    def get_domains(self):
        return get_domains(self.attrs)

    def get_shape(self):
        return self.data.shape

    def _load_batch(self, data: LazyChunk):
        (
            self.domains,
            self.mem_cols,
            self.info_cols,
            self.mem_noncommon,
            self.info_noncommon,
        ) = _marginal_loader(self.attrs, data)

    def _process_batch(
        self,
        requests: list[MarginalRequest],
        desc: str = "Calculating marginals",
    ):
        base_args = {
            "mem_cols": self.mem_cols,
            "info_cols": self.info_cols,
            "mem_noncommon": self.mem_noncommon,
            "info_noncommon": self.info_noncommon,
            "domains": self.domains,
            "mem_marginals": self.mem_marginals,
            "info_marginals": self.info_marginals,
        }

        per_call_args = [{"i": i, "req": req} for i, req in enumerate(requests)]

        res = process_in_parallel(
            _marginal_worker,
            per_call_args,
            base_args,
            min_chunk_size=self.min_chunk_size,
            max_worker_mult=self.max_worker_mult,
            desc=desc,
            initializer=_marginal_initializer,
            finalizer=_marginal_finalizer,
        )

        return res

    def _unload_batch(self):
        self.mem_cols.close()
        self.mem_cols.unlink()

        self.mem_noncommon.close()
        self.mem_noncommon.unlink()

    def _load_marginals(self, data: list[np.ndarray]):
        self.mem_marginals, self.info_marginals = map_to_memory({"data": data})

    def _unload_marginals(self):
        if self.mem_marginals is not None:
            self.mem_marginals.close()
            self.mem_marginals.unlink()

        self.mem_marginals = None
        self.info_marginals = None

    def _postprocess(self, requests: list[MarginalRequest], data: list[np.ndarray]):
        out = []
        for req, res in zip(requests, data):
            if req.x is not None:
                out.append(postprocess(res, req.zero_fill))
            else:
                out.append(postprocess_1way(res, req.zero_fill))
        return out

    def _process_merged(self, requests: list[MarginalRequest], desc: str):
        """Loads dataset into RAM and calculates marginals by placing dataset
        in shared memory and partitioning marginals into parallel processes.

        Requires being able to fit dataset in RAM."""
        if not self._loaded:
            self._load_batch(self.data)
            self._loaded = True

        self._unload_marginals()
        res = self._process_batch(requests, desc=desc)
        return self._postprocess(requests, res)  # type: ignore

    def _process_batched_sequential(self, requests: list[MarginalRequest], desc: str):
        """Loads partitions sequentially into main memory and partitions marginals
        into workers to process them. The next partition is loaded in parallel.

        Only requires one partition and one running copy of marginal counts in memory.

        Most memory efficient. If marginal calculations require more than the time
        to load a partition also most time efficient.

        However, for small batches of marginals, incurs having to load the dataset
        in a non-parallelized manner, which takes minutes."""

        chunks = list(self.data.values())
        partition_n = (len(chunks) - 1) // self.sequential_chunks + 1
        print(partition_n)
        partitions = [
            [chunks[i] for i in range(self.sequential_chunks * j, min(self.sequential_chunks * (j + 1), len(chunks)))]
            for j in range(partition_n)
        ]
        print([len(p) for p in partitions])
        assert len(partitions) >= 2

        res = process_async(_marginal_loader, self.attrs, partitions[-1])
        (
            self.domains,
            self.mem_cols,
            self.info_cols,
            self.mem_noncommon,
            self.info_noncommon,
        ) = res.get()

        res = process_async(_marginal_loader, self.attrs, partitions[0])
        self._unload_marginals()
        out = self._process_batch(requests)
        self._load_marginals(out)  # type: ignore

        for i in piter(range(len(partitions) - 1), desc=desc, leave=False):
            (
                self.domains,
                self.mem_cols,
                self.info_cols,
                self.mem_noncommon,
                self.info_noncommon,
            ) = res.get()

            if i < len(partitions) - 2:
                res = process_async(_marginal_loader, self.attrs, partitions[i + 1])

            self._process_batch(requests)
            self._unload_batch()

        assert self.mem_marginals is not None and self.info_marginals is not None
        raw = load_from_memory(self.mem_marginals, self.info_marginals)
        out = self._postprocess(requests, raw["data"])
        self._unload_marginals()
        return out

    def _process_batched_parallel(self, requests: list[MarginalRequest], desc: str):
        """Each worker is given one partition, which it loads, and calculates
        all the marginals for it. The marginals for each partition are merged in
        the end.

        Requires keeping a copy of marginals in memory, which makes it less
        efficient than `_process_batched_sequential()`. However, for low numbers
        of marginals, loading is the main bottleneck and can be linearly
        parallelized (using fast storage) this way."""

        base_args = {"requests": requests, "attrs": self.attrs}
        per_call_args = [{"chunk": chunk} for chunk in self.data.values()]

        res = process_in_parallel(
            _marginal_parallel_worker,
            per_call_args,
            base_args,
            desc=desc,
            max_worker_mult=self.max_worker_mult,
        )

        assert len(res) > 0
        out = res[0]

        for sub_arrs in piter(
            res[1:], desc="Summing partitioned marginals", leave=False
        ):
            for i in range(len(out)):
                out[i] += sub_arrs[i]

        return self._postprocess(requests, out)

    def process(
        self, requests: list[MarginalRequest], desc: str = "Processing partition"
    ):
        if not self.batched:
            logger.debug(
                f"Processing {len(requests)} marginals by loading dataset in memory."
            )
            return self._process_merged(requests, desc)

        if self.sequential_min < len(requests):
            logger.debug(
                f"Processing {len(requests)} marginals by sequential loading of partitions into shared memory."
            )
            return self._process_batched_sequential(requests, desc)

        logger.debug(
            f"Processing {len(requests)} marginals by loading partitions in parallel."
        )
        return self._process_batched_parallel(requests, desc)

    def get_counts(self, desc: str = "Calculating counts"):
        if self.counts:
            return self.counts

        cols = []
        reqs = []
        for name, attr in self.attrs.items():
            for val in attr.vals:
                cols.append(val)
                reqs.append(
                    MarginalRequest(
                        None,
                        {name: AttrSelector(name, attr.common, {val: 0})},
                        False,
                        zero_fill=0,
                    )
                )
        count_arr = self.process(reqs, desc=desc)
        self.counts = {name: count for name, count in zip(cols, count_arr)}
        return self.counts

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

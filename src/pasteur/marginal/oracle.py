import logging
from typing import Any, Callable, Literal, NamedTuple, overload

import numpy as np
import pandas as pd

from ..attribute import Attributes, CatValue, SeqAttributes
from ..utils import LazyChunk, LazyFrame
from ..utils.progress import PROGRESS_STEP_NS, piter, process_in_parallel
from .memory import load_from_memory, map_to_memory, merge_memory
from .numpy import (
    ZERO_FILL,
    AttrSelector,
    AttrSelectors,
    TableSelector,
    expand_table,
    get_domains,
)
from .numpy import normalize as normalize_2way
from .numpy import normalize_1way

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
    x: AttrSelector
    p: AttrSelectors


PreprocessFun = Callable[
    [dict[str, Attributes], dict[str, LazyChunk], dict[str, LazyChunk]],
    tuple[Attributes, pd.DataFrame]
    | tuple[
        Attributes,
        pd.DataFrame,
        dict[str, Attributes | SeqAttributes],
        dict[str | tuple[str, int], pd.DataFrame],
    ],
]


def _tabular_load(
    attrs: dict[str, Attributes],
    tables: dict[str, LazyChunk],
    ids: dict[str, LazyChunk],
):
    assert len(tables) == 1 and len(attrs) == 1
    return next(iter(attrs.values())), next(iter(tables.values()))()

def _counts_load(
    attrs: dict[str, Attributes],
    tables: dict[str, LazyChunk],
    ids: dict[str, LazyChunk],
):
    return {}, pd.DataFrame(), attrs, {k: c() for k, c in tables.items()}

def sequential_load(
    attrs: dict[str, Attributes],
    tables: dict[str, LazyChunk],
    ids: dict[str, LazyChunk],
    preprocess: PreprocessFun,
):
    out = preprocess(attrs, tables, ids)
    if len(out) == 2:
        new_attrs, table = out
        hist_attrs = {}
        hist_tables = {}
    else:
        new_attrs, table, hist_attrs, hist_tables = out

    data, info = expand_table(new_attrs, table, hist_attrs, hist_tables)

    mem_arr, mem_info = map_to_memory(data)
    del data
    return mem_arr, mem_info, info


def _parallel_load_worker(
    attrs: dict[str, Attributes],
    tables: dict[str, LazyChunk],
    ids: dict[str, LazyChunk],
    preprocess: PreprocessFun,
):
    out = preprocess(attrs, tables, ids)
    if len(out) == 2:
        new_attrs, table = out
        hist_attrs = {}
        hist_tables = {}
    else:
        new_attrs, table, hist_attrs, hist_tables = out

    data, info = expand_table(new_attrs, table, hist_attrs, hist_tables)
    mem_arr, mem_info = map_to_memory(data)
    return mem_arr, mem_info, info


def parallel_load(
    attrs: dict[str, Attributes],
    tables: dict[str, LazyFrame],
    ids: dict[str, LazyFrame],
    preprocess: PreprocessFun,
):

    base_args = {
        "attrs": attrs,
        "preprocess": preprocess,
    }
    per_call_args = [
        {"tables": chunks, "ids": ids_chunks}
        for chunks, ids_chunks in LazyFrame.zip_values([tables, ids])
    ]
    out = process_in_parallel(
        _parallel_load_worker, per_call_args, base_args, desc="Loading data"
    )
    info = out[0][-1]
    mem_arr, mem_info = merge_memory(out)

    return mem_arr, mem_info, info


def _marginal_initializer(base_args, per_call_args):
    copy = base_args["copy"]
    data = load_from_memory(base_args["mem_arr"], base_args["mem_info"], copy)

    new_base_args = base_args.copy()
    new_base_args["data"] = data
    return new_base_args, per_call_args


def _marginal_worker(
    data,
    info,
    req: MarginalRequest,
    normalize: bool,
    zero_fill: float,
    postprocess: Callable | None,
    **_,
):
    x, p = req
    res = calc_marginal(data, info, x, p)

    if normalize:
        if postprocess is not None:
            return postprocess(*normalize_2way(res, zero_fill))
        else:
            return normalize_2way(res, zero_fill)

    if postprocess is not None:
        return postprocess(res)
    return res


def _marginal_worker_1way(
    data,
    info,
    req: AttrSelectors,
    normalize: bool,
    zero_fill: float,
    postprocess: Callable | None,
    **_,
) -> np.ndarray:
    res = calc_marginal_1way(data, info, req)

    if normalize:
        if postprocess is not None:
            return postprocess(normalize_1way(res, zero_fill))
        else:
            return normalize_1way(res, zero_fill)

    if postprocess is not None:
        return postprocess(res)
    return res


def _marginal_batch_worker_inmem(
    mem_arr,
    mem_info,
    info,
    arange,
    requests: list,
    one_way,
    progress_lock,
    progress_send,
) -> list[np.ndarray]:
    from time import time_ns

    data = load_from_memory(mem_arr, mem_info, range=arange, copy=True)
    out = []
    u = 0
    last_updated = time_ns()
    for x, p in requests:
        # @TODO: Add partial
        if one_way:
            out.append(calc_marginal_1way(data, info, x))
        else:
            out.append(calc_marginal(data, info, x, p))

        u += 1
        if (curr_time := time_ns()) - last_updated > PROGRESS_STEP_NS:
            last_updated = curr_time
            with progress_lock:
                progress_send.send(u)
            u = 0

    if u > 0:
        with progress_lock:
            progress_send.send(u)
    return out


def _marginal_batch_worker_load(
    attrs: dict[str, Attributes],
    tables: dict[str, LazyChunk],
    ids: dict[str, LazyChunk],
    preprocess: Callable[
        [dict[str, Attributes], dict[str, LazyChunk], dict[str, LazyChunk]],
        tuple[Attributes, pd.DataFrame]
        | tuple[
            Attributes,
            pd.DataFrame,
            dict[str, Attributes | SeqAttributes],
            dict[str | tuple[str, int], pd.DataFrame],
        ],
    ],
    requests: list,
    one_way: bool,
    progress_lock,
    progress_send,
) -> list[np.ndarray]:
    from time import time_ns

    out = preprocess(attrs, tables, ids)
    if len(out) == 2:
        new_attrs, table = out
        hist_attrs = {}
        hist_tables = {}
    else:
        new_attrs, table, hist_attrs, hist_tables = out

    data, info = expand_table(new_attrs, table, hist_attrs, hist_tables)

    out = []
    u = 0
    last_updated = time_ns()
    for x, p in requests:
        # @TODO: Add partial
        if one_way:
            out.append(calc_marginal_1way(data, info, x))
        else:
            out.append(calc_marginal(data, info, x, p))

        u += 1
        if (curr_time := time_ns()) - last_updated > PROGRESS_STEP_NS:
            last_updated = curr_time
            with progress_lock:
                progress_send.send(u)
            u = 0

    if u > 0:
        with progress_lock:
            progress_send.send(u)
    return out


class MarginalOracle:
    MODES = Literal[
        "out_of_core",
        "inmemory",
        "inmemory_shared",
        "inmemory_copy",
        "inmemory_batched",
    ]

    def __init__(
        self,
        attrs: dict[str, Attributes],
        tables: dict[str, LazyFrame],
        ids: dict[str, LazyFrame],
        preprocess: PreprocessFun = _tabular_load,
        mode: "MarginalOracle.MODES" = "out_of_core",
        *,
        min_chunk_size: int = 1,
        max_worker_mult: int = 1,
        repartitions: int | None = None,
        log: bool = True,
    ) -> None:
        self.attrs = attrs
        self.tables = tables
        self.ids = ids
        self.preprocess = preprocess

        if mode == "out_of_core" and not LazyFrame.are_partitioned([tables, ids]):
            logger.info("Data is not partitioned, switching to mode `inmemory_copy`.")
            self.mode = "inmemory_copy"
        elif mode == "inmemory":
            # inmemory is an alias for inmemory_copy
            self.mode = "inmemory_copy"
        else:
            self.mode = mode

        data_partitions = 1
        if LazyFrame.are_partitioned([tables, ids]):
            data_partitions = len(LazyFrame.zip([tables, ids]))
        self.repartitions = repartitions or data_partitions

        if self.repartitions == 1 and mode == "inmemory_batched":
            logger.info(
                "Data is not partitioned and `repartitions` is not provided. Can't infer partition number, switching to mode `inmemory_copy`."
            )
            self.mode = "inmemory_copy"

        self.min_chunk_size = min_chunk_size
        self.max_worker_mult = max_worker_mult
        self.counts = None
        self.log = log
        self._load_id = None

        self.marginal_count = 0

    def load_data(self, preprocess: PreprocessFun):
        if self._load_id:
            if self._load_id == id(preprocess):
                return
            else:
                self.unload_data()

        if LazyFrame.are_partitioned([self.tables, self.ids]):
            # Load data in parallel
            (self.mem_arr, self.mem_info, self.info) = parallel_load(
                self.attrs, self.tables, self.ids, preprocess
            )
        else:
            # Load data sequentially
            (self.mem_arr, self.mem_info, self.info) = sequential_load(
                self.attrs, self.tables, self.ids, preprocess # type: ignore
            ) 

        self._load_id = id(preprocess)

    def unload_data(self):
        if not self._load_id:
            return

        self.mem_arr.close()
        self.mem_arr.unlink()
        self._load_id = None

    def _process_inmemory(
        self,
        requests: list[MarginalRequest] | list[AttrSelectors],
        desc: str,
        normalize: bool,
        zero_fill: float | None,
        preprocess: PreprocessFun,
        postprocess: Callable | None,
    ):
        assert self.mode in ("inmemory_shared", "inmemory_copy")

        if len(requests) == 0:
            return []
        is_1way = not isinstance(requests[0], MarginalRequest)

        self.load_data(preprocess)
        base_args = {
            "mem_arr": self.mem_arr,
            "mem_info": self.mem_info,
            "info": self.info,
            "copy": self.mode == "inmemory_copy",
            "normalize": normalize,
            "zero_fill": zero_fill,
            "postprocess": postprocess,
        }

        per_call_args = [{"req": req} for req in requests]

        res = process_in_parallel(
            _marginal_worker_1way if is_1way else _marginal_worker,
            per_call_args,
            base_args,
            min_chunk_size=self.min_chunk_size,
            max_worker_mult=self.max_worker_mult,
            desc=desc,
            initializer=_marginal_initializer,
        )

        return res

    def _process_batched(
        self,
        requests: list[MarginalRequest] | list[AttrSelectors],
        desc: str,
        normalize: bool,
        zero_fill: float | None,
        preprocess: PreprocessFun,
        postprocess: Callable | None,
    ):
        assert self.mode in ("inmemory_batched", "out_of_core")

        from multiprocessing import Pipe
        from threading import Lock, Thread

        from ..utils.progress import MULTIPROCESS_ENABLE, get_manager

        if len(requests) == 0:
            return []
        is_1way = not isinstance(requests[0], MarginalRequest)

        progress_recv, progress_send = Pipe(duplex=False)
        if MULTIPROCESS_ENABLE:
            progress_lock = get_manager().Lock()
        else:
            # Use a thread lock to prevent launching a pool with multiprocess
            # disabled
            progress_lock = Lock()

        base_args = {
            "preprocess": preprocess,
            "progress_send": progress_send,
            "progress_lock": progress_lock,
            "requests": requests,
            "one_way": is_1way,
        }

        if self.mode == "out_of_core":
            base_args.update({"attrs": self.attrs})
            per_call_args = [
                {"ids": ids_chunk, "tables": chunks}
                for chunks, ids_chunk in LazyFrame.zip_values([self.tables, self.ids])
            ]
            l = len(requests) * len(LazyFrame.zip_values([self.tables, self.ids]))
            fun = _marginal_batch_worker_load
        else:
            self.load_data(preprocess)
            base_args.update(
                {"mem_arr": self.mem_arr, "mem_info": self.mem_info, "info": self.info}
            )
            n = next(iter(self.mem_info.values()))[0].shape[0]
            chunk_n_suggestion = min(n, self.repartitions)
            chunk_len = (n - 1) // chunk_n_suggestion + 1
            chunk_n = (n - 1) // chunk_len + 1
            chunk_ranges = [
                (chunk_len * j, min(chunk_len * (j + 1), n)) for j in range(chunk_n)
            ]
            per_call_args = [{"arange": chunk_range} for chunk_range in chunk_ranges]
            l = len(requests) * chunk_n
            fun = _marginal_batch_worker_inmem

        def track_progress():
            pbar = None
            n = 0
            while n < l and (u := progress_recv.recv()) is not None:
                if pbar is None:
                    # Start pbar after the partition pbar has started
                    pbar = piter(desc="Calculating submarginals", total=l, leave=False)
                n += u
                pbar.update(u)

        t = Thread(target=track_progress)
        try:
            t.start()

            res = process_in_parallel(
                fun,
                per_call_args,
                base_args,
                desc=desc,
                max_worker_mult=self.max_worker_mult,
            )
        finally:
            progress_send.send(None)
            progress_send.close()
            progress_recv.close()
            t.join()

        if len(res) == 0:
            return []

        out = []
        for i in piter(
            range(len(requests)),
            desc="Postprocessing partitioned marginals",
            leave=False,
        ):
            mar = np.sum([batch[i] for batch in res], axis=0)

            if postprocess is not None:
                if not normalize:
                    out.append(postprocess(mar))
                elif is_1way:
                    out.append(postprocess(normalize_1way(mar, zero_fill)))
                else:
                    out.append(postprocess(*normalize_2way(mar, zero_fill)))
            else:
                if not normalize:
                    out.append(mar)
                elif is_1way:
                    out.append(normalize_1way(mar, zero_fill))
                else:
                    out.append(normalize_2way(mar, zero_fill))

        return out

    @overload
    def process(
        self,
        requests: list[MarginalRequest],
        desc: str = ...,
        normalize: Literal[False] = ...,
        zero_fill: float | None = ...,
        preprocess: PreprocessFun | None = ...,
        postprocess: None = ...,
    ) -> list[np.ndarray]:
        ...

    @overload
    def process(
        self,
        requests: list[MarginalRequest],
        desc: str = ...,
        normalize: Literal[True] = ...,
        zero_fill: float | None = ...,
        preprocess: PreprocessFun | None = ...,
        postprocess: None = ...,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        ...

    @overload
    def process(
        self,
        requests: list[AttrSelectors],
        desc: str = ...,
        normalize: bool = ...,
        zero_fill: float | None = ...,
        preprocess: PreprocessFun | None = ...,
        postprocess: None = ...,
    ) -> list[np.ndarray]:
        ...

    @overload
    def process(
        self,
        requests: list[AttrSelectors],
        desc: str = ...,
        normalize: bool = ...,
        zero_fill: float | None = ...,
        preprocess: PreprocessFun | None = ...,
        postprocess: Callable = ...,
    ) -> list[Any]:
        ...

    def process(
        self,
        requests: list[MarginalRequest] | list[AttrSelectors],
        desc: str = "Processing partition",
        normalize: bool = True,
        zero_fill: float | None = ZERO_FILL,
        preprocess: PreprocessFun | None = None,
        postprocess: Callable | None = None,
    ) -> list[np.ndarray] | list[tuple[np.ndarray, np.ndarray, np.ndarray]] | list[Any]:
        self.marginal_count += len(requests)

        if not preprocess:
            preprocess = self.preprocess

        if self.mode in ("inmemory_batched", "out_of_core"):
            logger.debug(
                f"Processing {len(requests)} marginals by loading partitions in parallel."
            )
            return self._process_batched(
                requests, desc, normalize, zero_fill, preprocess, postprocess
            )
        else:
            logger.debug(
                f"Processing {len(requests)} marginals by loading dataset in memory."
            )
            return self._process_inmemory(requests, desc, normalize, zero_fill, preprocess, postprocess)

    def get_counts(self, desc: str = "Calculating counts"):
        if self.counts:
            return self.counts

        cols = []
        reqs: list[AttrSelectors] = []
        for table_name, table_attrs in self.attrs.items():
            for attr in table_attrs.values():
                if attr.common:
                    reqs.append([(table_name, attr.name, 0)])
                    cols.append((table_name, attr.common.name))
                for val_name, val in attr.vals.items():
                    if isinstance(val, CatValue):
                        reqs.append([(table_name, attr.name, {val_name: 0})])
                        cols.append((table_name, val_name))

        count_arr = self.process(reqs, preprocess=_counts_load, desc=desc) # type: ignore
        self.counts = {name: count for name, count in zip(cols, count_arr)}

        return self.counts

    def close(self):
        if self.log:
            logger.info(f"Processed {self.marginal_count} marginals.")
        self.unload_data()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

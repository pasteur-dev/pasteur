import logging
from typing import Any, Callable, Literal, NamedTuple, overload

import numpy as np

from ..attribute import Attributes
from ..utils import LazyChunk, LazyFrame
from ..utils.progress import PROGRESS_STEP_NS, piter, process_in_parallel
from .numpy import ZERO_FILL, AttrSelector, AttrSelectors, expand_table, get_domains
from .numpy import normalize as normalize_2way
from .numpy import normalize_1way
from .memory import load_from_memory, map_to_memory, allocate_memory

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
    partial: bool


def _find_columns(reqs: list[MarginalRequest]) -> list[str] | None:
    cols = set()
    for req in reqs:
        if req.x is not None:
            cols.update(req.x.cols.keys())
        for p in req.p.values():
            cols.update(p.cols.keys())

    return sorted(list(cols)) or None


def _find_columns_1way(X: list[AttrSelectors]) -> list[str] | None:
    cols = set()
    for x in X:
        for sel in x.values():
            cols.update(sel.cols.keys())

    return sorted(list(cols)) or None


def sequential_load(
    chunk: LazyChunk, attrs: Attributes, columns: list[str] | None = None
):
    df = chunk(columns=columns)
    cols, cols_noncommon, domains = expand_table(attrs, df)

    mem_cols, info_cols = map_to_memory(cols)
    del cols
    mem_noncommon, info_noncommon = map_to_memory(cols_noncommon)
    return mem_cols, info_cols, mem_noncommon, info_noncommon, domains


def _parallel_load_worker(
    mem_cols,
    info_cols,
    mem_noncommon,
    info_noncommon,
    chunk: LazyChunk,
    chunk_range: tuple[int, int],
    attrs: Attributes,
    columns: list[str] | None = None,
):
    cols = load_from_memory(mem_cols, info_cols, range=chunk_range)
    cols_noncommon = load_from_memory(mem_noncommon, info_noncommon, range=chunk_range)

    df = chunk(columns=columns)
    _, _, domains = expand_table(attrs, df, out_cols=cols, out_noncommon=cols_noncommon)
    return domains


def parallel_load(data: LazyFrame, attrs: Attributes, columns: list[str] | None = None):
    # Both memory allocations share the range
    mem_cols, info_cols, _ = allocate_memory(data, attrs, common=False)
    mem_noncommon, info_noncommon, ranges = allocate_memory(data, attrs, common=True)

    base_args = {
        "mem_cols": mem_cols,
        "info_cols": info_cols,
        "mem_noncommon": mem_noncommon,
        "info_noncommon": info_noncommon,
        "attrs": attrs,
        "columns": columns,
    }
    per_call_args = [
        {"chunk": data[name], "chunk_range": ranges[name]} for name in data.keys()
    ]
    out = process_in_parallel(
        _parallel_load_worker, per_call_args, base_args, desc="Loading data"
    )
    domains = out[0]

    return mem_cols, info_cols, mem_noncommon, info_noncommon, domains


def _marginal_initializer(base_args, per_call_args):
    copy = base_args["copy"]

    cols = load_from_memory(base_args["mem_cols"], base_args["info_cols"], copy)
    cols_noncommon = load_from_memory(
        base_args["mem_noncommon"], base_args["info_noncommon"], copy
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
    normalize: bool,
    zero_fill: float,
    postprocess: Callable | None,
    **_,
):
    x, p, partial = req

    if partial:
        # Native implementation is unfinished
        res = calc_marginal_np(cols, cols_noncommon, domains, x, p, partial)
    else:
        res = calc_marginal(cols, cols_noncommon, domains, x, p, partial)

    if normalize:
        if postprocess is not None:
            return postprocess(*normalize_2way(res, zero_fill))
        else:
            return normalize_2way(res, zero_fill)

    if postprocess is not None:
        return postprocess(res)
    return res


def _marginal_worker_1way(
    cols,
    cols_noncommon,
    domains,
    x: AttrSelectors,
    normalize: bool,
    zero_fill: float,
    postprocess: Callable | None,
    **_,
) -> np.ndarray:
    res = calc_marginal_1way(cols, cols_noncommon, domains, x)

    if normalize:
        if postprocess is not None:
            return postprocess(normalize_1way(res, zero_fill))
        else:
            return normalize_1way(res, zero_fill)

    if postprocess is not None:
        return postprocess(res)
    return res


def _marginal_batch_worker(
    requests: list[MarginalRequest],
    attrs: Attributes,
    chunk: LazyChunk,
    shared,
    progress_lock,
    progress_send,
) -> list[np.ndarray]:
    from time import time_ns

    if chunk is not None:
        cols, cols_noncommon, domains = expand_table(
            attrs, chunk(columns=_find_columns(requests))
        )
    else:
        (
            mem_cols,
            info_cols,
            mem_noncommon,
            info_noncommon,
            domains,
            chunk_range,
        ) = shared
        cols = load_from_memory(mem_cols, info_cols, range=chunk_range, copy=True)
        cols_noncommon = load_from_memory(
            mem_noncommon, info_noncommon, range=chunk_range, copy=True
        )

    out = []
    u = 0
    last_updated = time_ns()
    for x, p, partial in requests:
        if partial:
            # Native implementation is unfinished
            out.append(calc_marginal_np(cols, cols_noncommon, domains, x, p, partial))
        else:
            out.append(calc_marginal(cols, cols_noncommon, domains, x, p, partial))

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


def _marginal_batch_worker_1way(
    X: list[AttrSelectors],
    attrs: Attributes,
    chunk: LazyChunk | None,
    shared,
    progress_lock,
    progress_send,
) -> list[np.ndarray]:
    from time import time_ns

    if chunk is not None:
        cols, cols_noncommon, domains = expand_table(
            attrs, chunk(columns=_find_columns_1way(X))
        )
    else:
        (
            mem_cols,
            info_cols,
            mem_noncommon,
            info_noncommon,
            domains,
            chunk_range,
        ) = shared
        cols = load_from_memory(mem_cols, info_cols, range=chunk_range, copy=True)
        cols_noncommon = load_from_memory(
            mem_noncommon, info_noncommon, range=chunk_range, copy=True
        )

    out = []
    u = 0
    last_updated = time_ns()
    for x in X:
        out.append(calc_marginal_1way(cols, cols_noncommon, domains, x))

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
        attrs: Attributes,
        data: LazyFrame,
        mode: "MarginalOracle.MODES" = "out_of_core",
        *,
        min_chunk_size: int = 1,
        max_worker_mult: int = 1,
        repartitions: int | None = None,
        log: bool = True,
    ) -> None:
        self.attrs = attrs
        self.data = data

        if mode == "out_of_core" and not data.partitioned:
            logger.info("Data is not partitioned, switching to mode `inmemory_copy`.")
            self.mode = "inmemory_copy"
        elif mode == "inmemory":
            # inmemory is an alias for inmemory_copy
            self.mode = "inmemory_copy"
        else:
            self.mode = mode

        self.repartitions = repartitions or len(data)
        if self.repartitions == 1 and mode == "inmemory_batched":
            logger.info(
                "Data is not partitioned and `repartitions` is not provided. Can't infer partition number, switching to mode `inmemory_copy`."
            )
            self.mode = "inmemory_copy"

        self.min_chunk_size = min_chunk_size
        self.max_worker_mult = max_worker_mult
        self.counts = None
        self.log = log
        self._loaded = False

        self.marginal_count = 0

    def get_domains(self):
        return get_domains(self.attrs)

    def get_shape(self):
        return self.data.shape

    def load_data(self, columns: list[str] | None = None):
        if self._loaded:
            return

        if self.data.partitioned:
            # Load data in parallel
            (
                self.mem_cols,
                self.info_cols,
                self.mem_noncommon,
                self.info_noncommon,
                self.domains,
            ) = parallel_load(self.data, self.attrs, columns)
        else:
            # Load data sequentially
            (
                self.mem_cols,
                self.info_cols,
                self.mem_noncommon,
                self.info_noncommon,
                self.domains,
            ) = sequential_load(self.data, self.attrs, columns)

        self._loaded = True

    def unload_data(self):
        if not self._loaded:
            return

        self.mem_cols.close()
        self.mem_cols.unlink()

        self.mem_noncommon.close()
        self.mem_noncommon.unlink()
        self._loaded = False

    def _process_inmemory(
        self,
        requests: list[MarginalRequest] | list[AttrSelectors],
        desc: str,
        normalize: bool,
        zero_fill: float | None,
        postprocess: Callable | None,
    ):
        assert self.mode in ("inmemory_shared", "inmemory_copy")

        if len(requests) == 0:
            return []
        is_1way = not isinstance(requests[0], MarginalRequest)

        self.load_data()
        base_args = {
            "mem_cols": self.mem_cols,
            "info_cols": self.info_cols,
            "mem_noncommon": self.mem_noncommon,
            "info_noncommon": self.info_noncommon,
            "copy": self.mode == "inmemory_copy",
            "domains": self.domains,
            "normalize": normalize,
            "zero_fill": zero_fill,
            "postprocess": postprocess,
        }

        if is_1way:
            per_call_args = [{"x": x} for x in requests]
        else:
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
        postprocess: Callable | None,
    ):
        assert self.mode in ("inmemory_batched", "out_of_core")

        from multiprocessing import Pipe
        from threading import Thread, Lock

        from pasteur.utils.progress import get_manager, MULTIPROCESS_ENABLE

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
            "attrs": self.attrs,
            "progress_send": progress_send,
            "progress_lock": progress_lock,
        }
        if is_1way:
            base_args["X"] = requests
        else:
            base_args["requests"] = requests

        if self.mode == "out_of_core":
            base_args["shared"] = None
            per_call_args = [{"chunk": chunk} for chunk in self.data.values()]
            l = len(requests) * len(self.data)
        else:
            self.load_data()
            base_args["chunk"] = None
            shared_base = (
                self.mem_cols,
                self.info_cols,
                self.mem_noncommon,
                self.info_noncommon,
                self.domains,
            )
            n = self.data.shape[0]
            chunk_n_suggestion = min(n, self.repartitions)
            chunk_len = (n - 1) // chunk_n_suggestion + 1
            chunk_n = (n - 1) // chunk_len + 1
            chunk_ranges = [
                (chunk_len * j, min(chunk_len * (j + 1), n)) for j in range(chunk_n)
            ]
            per_call_args = [
                {"shared": (*shared_base, chunk_range)}
                for chunk_range in chunk_ranges
            ]
            l = len(requests) * chunk_n

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
                _marginal_batch_worker_1way if is_1way else _marginal_batch_worker,
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
        postprocess: Callable = ...,
    ) -> list[Any]:
        ...

    def process(
        self,
        requests: list[MarginalRequest] | list[AttrSelectors],
        desc: str = "Processing partition",
        normalize: bool = True,
        zero_fill: float | None = ZERO_FILL,
        postprocess: Callable | None = None,
    ) -> list[np.ndarray] | list[tuple[np.ndarray, np.ndarray, np.ndarray]] | list[Any]:
        self.marginal_count += len(requests)

        if self.mode in ("inmemory_batched", "out_of_core"):
            logger.debug(
                f"Processing {len(requests)} marginals by loading partitions in parallel."
            )
            return self._process_batched(
                requests, desc, normalize, zero_fill, postprocess
            )
        else:
            logger.debug(
                f"Processing {len(requests)} marginals by loading dataset in memory."
            )
            return self._process_inmemory(requests, desc, normalize, zero_fill, postprocess)  # type: ignore

    def get_counts(self, desc: str = "Calculating counts"):
        if self.counts:
            return self.counts

        cols = []
        reqs = []
        for name, attr in self.attrs.items():
            for val in attr.vals:
                cols.append(val)
                reqs.append({name: AttrSelector(name, attr.common, {val: 0})})

        count_arr = self.process(reqs, desc=desc)
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

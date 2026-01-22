from collections import defaultdict
import logging
from typing import (
    Callable,
    Generic,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import pandas as pd

from ..utils.data import LazyDataset, LazyPartition

from ..attribute import (
    Attribute,
    Attributes,
    CatValue,
    DatasetAttributes,
    SeqAttributes,
)
from ..utils import LazyChunk, LazyFrame
from ..utils.progress import PROGRESS_STEP_NS, piter, process_in_parallel
from .memory import load_from_memory, map_to_memory, merge_memory
from .numpy import (
    AttrName,
    AttrSelectors,
    CalculationInfo,
    ChildSelector,
    CommonSelector,
    TableSelector,
    expand_table,
)

logger = logging.getLogger(__name__)

try:
    from .native_py import has_simd

    if has_simd():
        from .native_py import calc_marginal
    else:
        from .numpy import calc_marginal
except Exception as e:
    from .numpy import calc_marginal

A = TypeVar("A", covariant=True)

MarginalRequest = Sequence[
    tuple[TableSelector, AttrName, ChildSelector | CommonSelector]
    | tuple[AttrName, ChildSelector | CommonSelector]
]


def convert_reqs(
    reqs: list[MarginalRequest],
) -> list[AttrSelectors]:
    return [[y if len(y) == 3 else (None, *y) for y in x] for x in reqs]


class PreprocessFun(Protocol):
    def __call__(
        self,
        data: Mapping[str, LazyPartition],
    ) -> dict[TableSelector, pd.DataFrame]: ...


class PostprocessFun(Protocol, Generic[A]):
    def __call__(
        self, req: AttrSelectors, mar: np.ndarray, info: CalculationInfo
    ) -> A: ...


def counts_preprocess(
    data: Mapping[str, LazyPartition]
) -> dict[TableSelector, pd.DataFrame]:
    return {k: v() for k, v in data.items() if not k.startswith("ids_")}


def _tabular_load(
    data: Mapping[str, LazyPartition],
) -> dict[TableSelector, pd.DataFrame]:
    return {None: next(iter(v for d, v in data.items() if "ids_" not in d))()}


def sequential_load(
    attrs: Mapping[str | None, Attributes | SeqAttributes],
    data: Mapping[str, LazyPartition],
    preprocess: PreprocessFun,
):
    out = preprocess(data)

    cols, info = expand_table(attrs, out)
    mem_arr, mem_info = map_to_memory(cols)
    return mem_arr, mem_info, info


def parallel_load(
    attrs: Mapping[str | None, Attributes | SeqAttributes],
    data: Mapping[str, LazyPartition],
    preprocess: PreprocessFun,
):

    base_args = {
        "attrs": attrs,
        "preprocess": preprocess,
    }
    per_call_args = [{"data": chunks} for chunks in LazyFrame.zip_values(data)]
    out = process_in_parallel(
        sequential_load, per_call_args, base_args, desc="Loading data"
    )
    info = out[0][-1]
    mem_arr, mem_info = merge_memory(out)

    return mem_arr, mem_info, info


def get_info(
    attrs: Mapping[str | None, Attributes | SeqAttributes],
    data: Mapping[str, LazyPartition],
    preprocess: PreprocessFun,
):
    out = preprocess(
        {k: v.sample if isinstance(v, LazyDataset) else v() for k, v in data.items()}
    )

    _, info = expand_table(attrs, out)
    return info


def _marginal_initializer(base_args, per_call_args):
    copy = base_args["copy"]
    data = load_from_memory(base_args["mem_arr"], base_args["mem_info"], copy)

    new_base_args = base_args.copy()
    new_base_args["data"] = data
    return new_base_args, per_call_args


def _marginal_worker(
    data,
    info,
    req: AttrSelectors,
    postprocess: PostprocessFun | None,
    **_,
) -> np.ndarray:
    res = calc_marginal(data, info, req)

    if postprocess is not None:
        return postprocess(req, res, info)
    return res


def _marginal_batch_worker_inmem(
    mem_arr,
    mem_info,
    info,
    arange,
    requests: list,
    progress_lock,
    progress_send,
) -> list[np.ndarray]:
    from time import time_ns

    data = load_from_memory(mem_arr, mem_info, range=arange, copy=True)
    out = []
    u = 0
    last_updated = time_ns()
    for x in requests:
        out.append(calc_marginal(data, info, x))

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
    attrs: dict[str | None, Attributes | SeqAttributes],
    data: dict[str, LazyPartition],
    preprocess: PreprocessFun,
    requests: list,
    progress_lock,
    progress_send,
) -> list[np.ndarray]:
    from time import time_ns

    out = preprocess(data)
    cols, info = expand_table(attrs, out)

    out = []
    u = 0
    last_updated = time_ns()
    for x in requests:
        out.append(calc_marginal(cols, info, x))

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


def _is_attributes(a) -> TypeGuard[Attributes]:
    if not len(a):
        return False
    return isinstance(next(iter(a.values())), Attribute)


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
        data: Mapping[str, LazyPartition],
        attrs: DatasetAttributes | Attributes,
        preprocess: PreprocessFun = _tabular_load,
        mode: "MarginalOracle.MODES" = "out_of_core",
        *,
        min_chunk_size: int = 1,
        max_worker_mult: int = 1,
        repartitions: int | None = None,
        log: bool = True,
    ) -> None:
        if _is_attributes(attrs):
            self.attrs: DatasetAttributes = {None: attrs}
        else:
            self.attrs = cast(DatasetAttributes, attrs)
        self.data = data
        self.preprocess = preprocess

        if mode == "out_of_core" and not LazyFrame.are_partitioned(data):
            logger.info("Data is not partitioned, switching to mode `inmemory_copy`.")
            self.mode = "inmemory_copy"
        elif mode == "inmemory":
            # inmemory is an alias for inmemory_copy
            self.mode = "inmemory_copy"
        else:
            self.mode = mode

        data_partitions = 1
        if LazyFrame.are_partitioned(data):
            data_partitions = len(LazyFrame.zip_values(data))
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
        self.info = None

        self.marginal_count = 0

    def load_data(self, preprocess: PreprocessFun):
        if self._load_id:
            if self._load_id == id(preprocess):
                return
            else:
                self.unload_data()

        if LazyFrame.are_partitioned(self.data):
            # Load data in parallel
            (self.mem_arr, self.mem_info, self.info) = parallel_load(
                self.attrs, self.data, preprocess
            )
        else:
            # Load data sequentially
            (self.mem_arr, self.mem_info, self.info) = sequential_load(
                self.attrs, self.data, preprocess
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
        requests: list[AttrSelectors],
        desc: str,
        preprocess: PreprocessFun,
        postprocess: PostprocessFun | None,
    ):
        assert self.mode in ("inmemory_shared", "inmemory_copy")

        if len(requests) == 0:
            return []

        self.load_data(preprocess)
        base_args = {
            "mem_arr": self.mem_arr,
            "mem_info": self.mem_info,
            "info": self.info,
            "copy": self.mode == "inmemory_copy",
            "postprocess": postprocess,
        }

        per_call_args = [{"req": req} for req in requests]

        res = process_in_parallel(
            _marginal_worker,
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
        requests: list[AttrSelectors],
        desc: str,
        preprocess: PreprocessFun,
        postprocess: PostprocessFun | None,
    ):
        assert self.mode in ("inmemory_batched", "out_of_core")

        from multiprocessing import Pipe
        from threading import Lock, Thread

        from ..utils.progress import MULTIPROCESS_ENABLE, get_manager

        if len(requests) == 0:
            return []

        progress_recv, progress_send = Pipe(duplex=False)
        if MULTIPROCESS_ENABLE:
            progress_lock = get_manager().Lock()
        else:
            # Use a thread lock to prevent launching a pool with multiprocess
            # disabled
            progress_lock = Lock()

        base_args = {
            "progress_send": progress_send,
            "progress_lock": progress_lock,
            "requests": requests,
        }

        if self.mode == "out_of_core":
            base_args.update({"attrs": self.attrs})
            per_call_args = [
                {"data": chunks} for chunks in LazyFrame.zip_values(self.data)
            ]
            l = len(requests) * len(LazyFrame.zip_values(self.data))
            base_args.update({"preprocess": preprocess})
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
                if not self.info:
                    self.info = get_info(self.attrs, self.data, preprocess)
                out.append(postprocess(requests[i], mar, self.info))
            else:
                out.append(mar)

        return out

    @overload
    def process(
        self,
        requests: list[MarginalRequest],
        desc: str = ...,
        preprocess: PreprocessFun | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def process(
        self,
        requests: list[MarginalRequest],
        desc: str = ...,
        preprocess: PreprocessFun | None = ...,
        postprocess: PostprocessFun[A] = ...,  # type: ignore
    ) -> list[A]: ...

    def process(
        self,
        requests: list[MarginalRequest],
        desc: str = "Processing partition",
        preprocess: PreprocessFun | None = None,
        postprocess: PostprocessFun[A] | None = None,
    ) -> list[np.ndarray] | list[A]:
        self.marginal_count += len(requests)

        if not preprocess:
            preprocess = self.preprocess

        if self.mode in ("inmemory_batched", "out_of_core"):
            logger.debug(
                f"Processing {len(requests)} marginals by loading partitions in parallel."
            )
            return self._process_batched(
                convert_reqs(requests), desc, preprocess, postprocess
            )
        else:
            logger.debug(
                f"Processing {len(requests)} marginals by loading dataset in memory."
            )
            return self._process_inmemory(
                convert_reqs(requests), desc, preprocess, postprocess
            )

    def get_counts(
        self, desc: str = "Calculating counts"
    ) -> dict[str | None, dict[str, np.ndarray]]:
        if self.counts:
            return self.counts

        cols = []
        reqs: list[AttrSelectors] = []
        for table, table_attrs in self.attrs.items():
            attrs_dict: dict[TableSelector, Attributes]
            if isinstance(table_attrs, SeqAttributes):
                assert table is not None
                attrs_dict = {(table, k): v for k, v in table_attrs.hist.items()}
                if table_attrs.attrs is not None:
                    attrs_dict[table] = table_attrs.attrs
            else:
                attrs_dict = {table: table_attrs}

            for table_name, attrs in attrs_dict.items():
                for attr in attrs.values():
                    if attr.common:
                        reqs.append([(table_name, attr.name, 0)])
                        cols.append((table_name, attr.common.name))
                    for val_name, val in attr.vals.items():
                        if isinstance(val, CatValue):
                            reqs.append([(table_name, attr.name, {val_name: 0})])
                            cols.append((table_name, val_name))

        count_arr = self.process(reqs, desc=desc)  # type: ignore

        self.counts = defaultdict(dict)
        for (table, name), count in zip(cols, count_arr):
            self.counts[table][name] = count

        return dict(self.counts)

    def close(self):
        if self.log:
            logger.info(f"Processed {self.marginal_count} marginals.")
        self.unload_data()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

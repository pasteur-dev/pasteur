import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from pasteur.hierarchy import rebalance_attributes
from pasteur.marginal import AttrSelector, MarginalOracle, MarginalRequest
from pasteur.utils import LazyDataset, LazyPartition
from pasteur.utils.progress import init_pool

Ns = [
    1,
    10,
    100,
    1_000,
    2_000,
    5_000,
    10_000,
    # 20_000
]

mar = {
    "gender": AttrSelector("gender", 0, {"gender": 0}),
    "warning": AttrSelector("warning", 0, {"warning": 0}),
    "intime": AttrSelector("intime", 0, {"intime_day": 1}),
    "outtime": AttrSelector("outtime", 0, {"outtime_day": 1}),
    "charttime": AttrSelector(
        "charttime", 0, {"charttime_day": 0, "charttime_time": 2}
    ),
    "first_careunit": AttrSelector("first_careunit", 0, {"first_careunit": 2}),
    "valueuom": AttrSelector("valueuom", 0, {"valueuom": 5}),
}

req = MarginalRequest(None, mar, False)


def bench_oracle(oracle: MarginalOracle, workers: int, desc: str):
    with init_pool(workers):
        print(f"\n########\n{desc}, workers={workers}")
        for N in Ns:
            reqs = [req for _ in range(N)]

            start = time.perf_counter()
            oracle.process(reqs)
            end = time.perf_counter()

            duration = end - start
            print(
                f"########\n{desc}, workers={workers:2d} N={N:10d}: {int(duration // 60)}:{duration % 60:06.3f}"
            )


def load_inmemory(new_attrs, wrk, min_chunk_size):
    inmemory = MarginalOracle(
        new_attrs, wrk, batched=False, min_chunk_size=min_chunk_size
    )

    start = time.perf_counter()
    inmemory._load_batch(wrk)
    inmemory._loaded = True
    end = time.perf_counter()

    print(f"\n#########\nInmemory load: {end - start:.2f}")
    duration = end - start
    print(f"{int(duration // 60)}:{duration % 60:.3f}")
    nbytes = inmemory.mem_cols.size + inmemory.mem_noncommon.size
    print(f"Total memory use: {nbytes//1_000_000:,} MB")

    return inmemory


def benchmark(
    catalog,
    workers: int,
    worker_mult: int,
    min_chunk_size: int,
    sequential_chunks: int,
    inmemory: bool,
    parallel: bool,
    sequential: bool,
):
    old_attrs = catalog.load("mimic_billion.trn.table")["idx"].get_attributes()
    wrk = catalog.load("mimic_billion.wrk.idx_table")

    m = MarginalOracle(old_attrs, wrk)
    counts = m.get_counts()
    m.close()

    attrs = rebalance_attributes(counts, old_attrs, fixed=[2, 4, 8, 16, 32, 48], u=4)

    print("#########\nAttributes")
    for name, attr in attrs.items():
        for val_name, val in attr.vals.items():
            print(
                f"{val_name:>25s} | {', '.join(f'{h}:{val.get_domain(h):3d}' for h in range(val.height))}"
            )

    def get_dom(cols):
        dom = 1
        for sel in cols.values():
            for n, h in sel.cols.items():
                dom *= attrs[sel.name][n].get_domain(h)
        return dom

    print(f"{get_dom(mar):,}")

    if inmemory:
        oracle = load_inmemory(attrs, wrk, min_chunk_size)
        desc = "Inmemory"
    elif parallel:
        oracle = MarginalOracle(
            attrs,
            wrk,
            batched=True,
            sequential_min=100_000,
            max_worker_mult=worker_mult,
            min_chunk_size=min_chunk_size,
            sequential_chunks=sequential_chunks,
        )
        desc = "Parallel"
    elif sequential:
        oracle = MarginalOracle(
            attrs,
            wrk,
            batched=True,
            sequential_min=0,
            max_worker_mult=worker_mult,
            min_chunk_size=min_chunk_size,
            sequential_chunks=sequential_chunks,
        )
        desc = "Sequential"
    else:
        assert False

    bench_oracle(oracle, workers, desc)
    oracle.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Marginal Benchmark")
    parser.add_argument("-w", "--workers", type=int, default=16)
    parser.add_argument("-m", "--worker-mult", type=int, default=2)
    parser.add_argument("-c", "--min-chunk-size", type=int, default=1)
    parser.add_argument("-s", "--sequential-chunks", type=int, default=1)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inmemory", action="store_true", default=False)
    group.add_argument("--parallel", action="store_true", default=False)
    group.add_argument("--sequential", action="store_true", default=False)

    args = parser.parse_args()

    bootstrap_project(Path(".").expanduser().resolve())
    with KedroSession.create() as session:
        ctx = session.load_context()
        benchmark(
            ctx.catalog,
            args.workers,
            args.worker_mult,
            args.min_chunk_size,
            args.sequential_chunks,
            args.inmemory,
            args.parallel,
            args.sequential,
        )

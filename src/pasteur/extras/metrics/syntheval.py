from __future__ import annotations

import logging
import time

import pandas as pd

from pasteur.metric import Summaries
from pasteur.utils import LazyDataset

from ...attribute import Attributes, CatValue, SeqValue, get_dtype
from ...metric import Metric, Summaries
from ...utils import LazyChunk, LazyFrame, data_to_tables
from ...utils.progress import process_in_parallel

ROW_LIMIT = 10000
FONT_SIZE = "13px"

logger = logging.getLogger(__name__)


def _process_se(
    table: str,
    metric: str,
    args: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    syn: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    import pandas as pd
    from syntheval import SynthEval
    import warnings

    #  RuntimeWarning: invalid value encountered in cast
    warnings.filterwarnings("ignore")

    try:
        start = time.time()
        logger.info(
            f"Processing SynthEval metric '{metric:10s}' for table '{table:10s}'"
        )
        SE = SynthEval(
            train,
            holdout_dataframe=test,
            unique_threshold=400,
            verbose=False,
            cat_cols=list(train.columns),
        )
        data = SE.evaluate_metric(syn, metric, args)

        logger.info(
            f"Processed  SynthEval metric '{metric:10s}' for table '{table:10s}' in {time.time() - start:5.2f} s"
        )

        assert data, "SynthEval failed to return any data"

        return data
    except Exception as e:
        logger.exception(f"Error in SynthEval metric '{metric}' in table '{table}':")
        return {}, []


def _process_outer(
    wrk: dict[str, LazyDataset],
    ref: dict[str, LazyDataset],
    syn: dict[str, LazyDataset],
):
    metrics = {
        # Privacy
        "corr_diff": {"mixed_corr": True},
        "att_discl": {},
        "dcr": {},
        "eps_risk": {},
        "hit_rate": {},
        # "mia": {}, # too slow
        # "nnaa": {}, # too slow
        "nndr": {},
    }

    per_call = []

    for table in wrk:
        train = wrk[table]()[:ROW_LIMIT]
        test = ref[table]()[:ROW_LIMIT]
        synth = syn[table]()[:ROW_LIMIT]

        for metric, args in metrics.items():
            if table.endswith("_ids"):
                continue
            per_call.append(
                {
                    "table": table,
                    "train": train,
                    "test": test,
                    "syn": synth,
                    "metric": metric,
                    "args": args,
                }
            )

    # Process marginals
    raw = process_in_parallel(
        _process_se,
        per_call,
        # base_args=base_args,
        desc="Processing SynthEval metrics",
    )

    out = []
    for (_, norm), info in zip(raw, per_call):
        if not norm:
            continue

        table = info["table"]
        base_metric = info["metric"]

        for data in norm:
            out.append(
                {
                    "table": table,
                    "base_metric": base_metric,
                    **data,
                }
            )

    return pd.DataFrame(
        out,
        columns=[
            "table",
            "base_metric",
            "metric",
            "dim",
            "val",
            "err",
            "n_val",
            "n_err",
        ],
    )


class SynthEvalMetric(Metric[None, list[pd.DataFrame]]):
    name = "syntheval"
    encodings = "idx"

    def fit(
        self,
        meta: dict[str, Attributes],
        data: dict[str, LazyFrame],
    ):
        pass

    def preprocess(
        self,
        wrk: dict[str, LazyDataset],
        ref: dict[str, LazyDataset],
    ) -> Summaries[pd.DataFrame]:
        return Summaries(
            wrk=_process_outer(wrk, ref, wrk), ref=_process_outer(wrk, ref, ref)
        )

    def process(
        self,
        wrk: dict[str, LazyDataset],
        ref: dict[str, LazyDataset],
        syn: dict[str, LazyDataset],
        pre: Summaries[pd.DataFrame],
    ) -> Summaries[pd.DataFrame]:
        return pre.replace(syn=_process_outer(wrk, ref, syn))

    def visualise(
        self,
        data: dict[
            str,
            Summaries[pd.DataFrame],
        ],
    ):

        from pasteur.utils.styles import use_style
        from pasteur.utils.mlflow import color_dataframe, gen_html_table
        import matplotlib.pyplot as plt
        import mlflow

        use_style("mlflow")

        first = next(iter(data.values()))
        wrk = first.wrk
        ref = first.ref
        df = {
            "wrk": wrk,
            "ref": ref,
            **{k: v.syn for k, v in data.items() if k != "ref"},
        }

        vals = [
            "val",
            # "err",
            "n_val",
            # "n_err",
        ]
        formatters = {v: {"precision": 2} for v in vals}

        dfs = {}
        for table in ref["table"].unique():
            dfs[table] = color_dataframe(
                {k: v[v["table"] == table] for k, v in df.items()},
                idx=["base_metric", "metric"],
                cols=[],
                vals=vals,
                formatters=formatters,
                split_ref="ref",
            )

        fn = f"syntheval.html"
        mlflow.log_text(gen_html_table(dfs, FONT_SIZE), fn)

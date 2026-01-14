from pasteur.amalgam.synth import MARGINAL_PARAMS_DEFAULT
from pasteur.amalgam.synth import AmalgamMarginalParams
from pasteur.marginal.oracle import MarginalOracle
from typing import Any

import pandas as pd

from pasteur.amalgam.llm import (
    AmalgamHFParams,
    AmalgamORParams,
    load_llm_model_eval,
    hold_gpu_lock,
    evaluate,
)
from pasteur.amalgam.synth import MODEL_PARAMS_QWEN3
from pasteur.metric import Summaries
from pasteur.utils import LazyDataset

from ...attribute import Attributes, CatValue, SeqValue, get_dtype
from ...metric import Metric, Summaries
from ...utils import LazyChunk, LazyFrame, data_to_tables
from ...utils.progress import process_in_parallel

DEFAULT_PROMPT = """
You are an expert data scientist.

You are given the following <samples_n> real samples as a reference:
<samples>

Then, you are asked to comment on how real the following sample is and give it a rating from 1 to 5 (5 being very real):
<eval>
"""

class LlmEvaluatorMetric(Metric[None, None | pd.DataFrame]):
    name = "llmeval"
    encodings = ["json", "flat"]

    def __init__(
        self,
        samples: int = 10,
        samples_ref: int | None = None,
        model: AmalgamHFParams | AmalgamORParams = MODEL_PARAMS_QWEN3,
        prompt: str = DEFAULT_PROMPT,
        marginal: AmalgamMarginalParams = MARGINAL_PARAMS_DEFAULT,
        reason: bool = False,
        topk: int = 3,
        **_,
    ):
        self.samples = samples
        self.samples_ref = samples_ref if samples_ref is not None else samples
        self.prompt = prompt
        self.marginal = marginal
        self.reason = reason
        self.topk = topk
        self.model = {
            **MODEL_PARAMS_QWEN3,
            **model,
        }

    def fit(
        self,
        meta: dict[str, Any],
        data: dict[str, LazyFrame],
    ):
        self.meta = meta
        with MarginalOracle(
            data["flat"],  # type: ignore
            self.meta["flat"]["meta"],  # type: ignore
            mode=self.marginal["mode"],
            min_chunk_size=self.marginal["min_chunk"],
            max_worker_mult=self.marginal["worker_mult"],
        ) as o:
            self.counts = o.get_counts(desc="Calculating counts for column rebalancing")

    def evaluate_dataset(
        self,
        split: str,
        samples_n: int,
        wrk: dict[str, dict[str, LazyDataset]],
        ref: dict[str, dict[str, LazyDataset]],
        _llm=None,
    ) -> dict[str, pd.DataFrame]:
        if not _llm:
            llm = load_llm_model_eval(
                self.model,
                reason=self.reason,
            )
            if _llm is not None:
                _llm.update(llm)
        else:
            llm = _llm

        return evaluate(
            llm,
            self.prompt,
            self.counts[None],
            wrk["flat"]["table"](),
            wrk["json"],
            ref["flat"]["table"](),
            ref["json"],
            samples_n,
            self.topk,
            split,
        )

    def preprocess(
        self,
        wrk: dict[str, dict[str, LazyDataset]],
        ref: dict[str, dict[str, LazyDataset]],
        _llm=None,
    ) -> Summaries[None | pd.DataFrame]:
        with hold_gpu_lock():
            return Summaries(wrk=None, ref=self.evaluate_dataset("ref", self.samples_ref, wrk, ref, _llm=_llm))

    def process(
        self,
        wrk: dict[str, dict[str, LazyDataset]],
        ref: dict[str, dict[str, LazyDataset]],
        syn: dict[str, dict[str, LazyDataset]],
        pre: Summaries[pd.DataFrame],
        _llm=None,
    ) -> Summaries[pd.DataFrame]:
        with hold_gpu_lock():
            return pre.replace(syn=self.evaluate_dataset("syn", self.samples, wrk, syn, _llm=_llm))

    def visualise(
        self,
        data: dict[
            str,
            Summaries[pd.DataFrame],
        ],
    ):
        # from pasteur.utils.styles import use_style
        # from pasteur.utils.mlflow import color_dataframe, gen_html_table
        # import matplotlib.pyplot as plt
        # import mlflow

        # use_style("mlflow")

        # fn = f"syntheval.html"
        # mlflow.log_text(gen_html_table(dfs, FONT_SIZE), fn)

        # import json

        # fn = f"_raw/metrics/distr/syntheval.json"
        # mlflow.log_text(json.dumps({k: v.to_dict() for k, v in df.items()}), fn)
        pass

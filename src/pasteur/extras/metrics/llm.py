import logging
from typing import Any

import pandas as pd

from pasteur.amalgam.llm import (
    AmalgamHFParams,
    AmalgamORParams,
    evaluate,
    hold_gpu_lock,
    load_llm_model_eval,
)
from pasteur.amalgam.synth import (
    MARGINAL_PARAMS_DEFAULT,
    MODEL_PARAMS_QWEN3,
    AmalgamMarginalParams,
)
from pasteur.marginal.oracle import MarginalOracle
from pasteur.metric import Summaries
from pasteur.utils import LazyDataset

from ...metric import Metric, Summaries
from ...utils import LazyFrame

logger = logging.getLogger(__name__)


def _get_model_name(model: dict) -> str:
    """Derive a short display name from model params."""
    if model.get("type") == "or":
        return model["model"].split("/")[-1]
    return model.get("filename", model.get("repo_id", "unknown")).removesuffix(".gguf")


DEFAULT_PROMPT = """
You are an expert data scientist.

You are given the following <samples_n> real samples as a reference:
<samples>

Then, you are asked to comment on how real the following sample is and give it a rating from 1 to 5 (5 being very real):
<eval>
"""


class LlmEvaluatorMetric(Metric[None, None | list[int]]):
    name = "llmeval"
    encodings = ["json", "flat"]
    tabular = False

    def __init__(
        self,
        samples: int | None,
        samples_ref: int | None = None,
        model: AmalgamHFParams | AmalgamORParams | list[AmalgamHFParams | AmalgamORParams] = MODEL_PARAMS_QWEN3,
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
        if isinstance(model, list):
            self.models = [{**MODEL_PARAMS_QWEN3, **m} for m in model]
        else:
            self.models = [{**MODEL_PARAMS_QWEN3, **model}]
        self.model = self.models[0]

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
        import numpy as np

        if not _llm:
            llm = load_llm_model_eval(
                self.model,
                reason=self.reason,
            )
            if _llm is not None:
                _llm.update(llm)
        else:
            llm = _llm

        data = evaluate(
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

        return list(np.bincount(np.array([x["score"] for x in data]), minlength=6)[1:6])

    def preprocess(
        self,
        wrk: dict[str, dict[str, LazyDataset]],
        ref: dict[str, dict[str, LazyDataset]],
        _llm=None,
    ) -> Summaries[None | pd.DataFrame]:
        if len(self.models) > 1:
            results = {}
            for model in self.models:
                self.model = model
                name = _get_model_name(model)
                logger.info(f"Evaluating ref data with model: {name}")
                with hold_gpu_lock(f"eval.{name}.ref"):
                    results[name] = self.evaluate_dataset(f"{name}.ref", self.samples_ref, wrk, ref, _llm=_llm)
            return Summaries(wrk=None, ref=results)

        with hold_gpu_lock("eval.ref"):
            return Summaries(
                wrk=None,
                ref=self.evaluate_dataset("ref", self.samples_ref, wrk, ref, _llm=_llm),
            )

    def process(
        self,
        wrk: dict[str, dict[str, LazyDataset]],
        ref: dict[str, dict[str, LazyDataset]],
        syn: dict[str, dict[str, LazyDataset]],
        pre: Summaries[pd.DataFrame],
        _llm=None,
    ) -> Summaries[pd.DataFrame]:
        if len(self.models) > 1:
            results = {}
            for model in self.models:
                self.model = model
                name = _get_model_name(model)
                logger.info(f"Evaluating syn data with model: {name}")
                with hold_gpu_lock(f"eval.{name}.syn"):
                    results[name] = self.evaluate_dataset(f"{name}.syn", self.samples, wrk, syn, _llm=_llm)
            return pre.replace(syn=results)

        with hold_gpu_lock("eval.syn"):
            return pre.replace(
                syn=self.evaluate_dataset("syn", self.samples, wrk, syn, _llm=_llm)
            )

    def _visualise(
        self,
        data: dict[
            str,
            Summaries[pd.DataFrame],
        ],
        model_name: str | None = None,
    ):
        import matplotlib.pyplot as plt
        import mlflow
        import numpy as np
        import pandas as pd

        from pasteur.utils.mlflow import (
            color_dataframe,
            gen_html_table,
            mlflow_log_figures,
        )
        from pasteur.utils.styles import use_style

        from .visual import _gen_bar, _percent_formatter

        use_style("mlflow")

        splits = {}
        splits["ref"] = next(iter(data.values())).ref
        for k, v in data.items():
            splits[k] = v.syn

            fig, ax = plt.subplots()

        cols = ["1", "2", "3", "4", "5"]
        title = f"LLM Evaluation Scores Distribution"
        if model_name:
            title += f" ({model_name})"
        x = np.array(range(len(cols)))
        w = 0.9 / len(splits)

        df_data = {}
        raw_data = {}
        avgs = {}
        for i, (name, c) in enumerate(splits.items()):
            h = c / sum(c) if sum(c) > 0 else c
            avg = sum((i + 1) * v for i, v in enumerate(h))
            avgs[name] = avg

            ax.bar(
                x - 0.45 + w * i,
                h,
                width=w,
                align="edge",
                label=f"{name} (avg: {avg:.2f})",
                # log=y_log,
            )

            df_data[name] = pd.Series(
                h,
                index=pd.Index(cols, name="Score"),
                name=name,
            )

            raw_data[name] = {str(i + 1): int(c[i]) for i in range(5)}

        plt.xticks(x, cols)
        rot = min(3 * len(cols), 90)
        if rot > 10:
            plt.setp(ax.get_xticklabels(), rotation=rot, horizontalalignment="right")

        ax.legend()
        ax.set_title(title)
        ax.yaxis.set_major_formatter(_percent_formatter)

        plt.tight_layout()

        fig_path = f"llm_scores/{model_name}" if model_name else "llm_scores"
        raw_path = f"_raw/llmeval/{model_name}.json" if model_name else "_raw/llmeval.json"
        mlflow_log_figures(fig_path, fig)
        mlflow.log_dict(raw_data, raw_path)

        label = f" for '{model_name}'" if model_name else ""
        logger.info(
            f"LLM Evaluation Scores{label} (%)\n"
            + (pd.DataFrame(df_data) * 100).to_markdown()
            + "\nAverages:\n"
            + "\n".join(f"- {k:>10s}: {v:.2f}" for k, v in avgs.items())
        )

    def visualise(
        self,
        data: dict[str, Summaries[pd.DataFrame]],
    ):
        if len(self.models) == 1:
            return self._visualise(data)

        ref_all = next(iter(data.values())).ref
        for model_name in ref_all:
            model_data = {
                k: Summaries(wrk=None, ref=ref_all[model_name], syn=v.syn[model_name])
                for k, v in data.items()
            }
            self._visualise(model_data, model_name=model_name)

import logging
from typing import Literal, NamedTuple, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ....attribute import Attributes
from ....metadata import Metadata
from ....metric import TableData, TableMetric, TableMetricFactory
from ....utils import LazyChunk, LazyFrame
from ....utils.progress import process_in_parallel

logger = logging.getLogger(__name__)


def _split_data(
    *data: pd.DataFrame,
    ratio: float,
    random_state: int | None,
    i: int,
    split: Literal["train", "test"],
):
    index = data[0].index
    train_idx, test_idx = train_test_split(
        index,
        test_size=ratio,
        random_state=random_state + i if random_state else None,
    )

    # TODO: expand tables
    l = [
        (
            d.loc[train_idx if split == "train" else test_idx]
            .sort_index()
            .reset_index(drop=True)
        )
        for d in data
    ]
    return l[0] if len(l) == 1 else l


class DataIterator:
    def __init__(
        self,
        x: LazyFrame,
        y: LazyFrame,
        drop_x_cols: list[str],
        y_col: str,
        random_state: int | None,
        ratio: float = 0.2,
        split: Literal["train", "test"] = "train",
    ) -> None:
        self.data = enumerate(zip(x.values(), y.values()))
        self.ratio = ratio
        self.random_state = random_state
        self.split: Literal["train", "test"] = split

        self.drop_x_cols = drop_x_cols
        self.y_col = y_col

    def __next__(self):
        i, (x, y) = next(self.data)
        return _split_data(
            x().drop(columns=self.drop_x_cols),
            y()[[self.y_col]],
            ratio=self.ratio,
            random_state=self.random_state,
            i=i,
            split=self.split,
        )


class DataIterable:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return DataIterator(*self.args, **self.kwargs)


class BaseModel:
    name: str
    x_trn_type: str
    y_trn_type: str
    size_limit: int | None = None
    y_col_types: list[str] | None = None

    def __init__(self, random_state: int):
        self.random_state = random_state

    def fit(self, data: DataIterable):
        raise NotImplementedError()

    def score(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        raise NotImplementedError()


def _get_model_types(cls: tuple[type[BaseModel]]):
    types = set()
    for c in cls:
        types.add(c.x_trn_type)
        types.add(c.y_trn_type)

    return sorted(list(types))


def _fit_model_worker(
    data: dict[str, LazyFrame],
    random_state: int,
    ratio: float,
    cls: type[BaseModel],
    drop_x_cols: list[str],
    y_col: str,
):
    model = cls(random_state=random_state)
    model.fit(
        DataIterable(
            data[cls.x_trn_type],
            data[cls.y_trn_type],
            drop_x_cols,
            y_col,
            random_state,
            ratio,
            "train",
        )
    )
    return model


class ModelData(NamedTuple):
    model: BaseModel
    drop_x_cols: list[str]
    y_col: str


def _score_models_worker(
    models: list[ModelData],
    random_state: int,
    ratio: float,
    chunks: dict[str, LazyChunk],
    i: int | None,
    split=None,
) -> tuple[int, list[float]]:
    data = {}
    l = -1
    for enc in chunks.keys():
        if i is not None and split:
            data[enc] = _split_data(
                chunks[enc](), ratio=ratio, random_state=random_state, i=i, split=split
            )
        else:
            data[enc] = chunks[enc]()
        l = len(data[enc])

    return l, [
        m.score(
            data[m.x_trn_type].drop(columns=drop_x_cols), data[m.y_trn_type][[y_col]]
        )
        for (m, drop_x_cols, y_col) in models
    ]


def _process_models(
    name: str,
    test_cols: dict[str, str],
    wrk: TableData,
    ref: TableData,
    syn: TableData | None,
    classes: tuple[type[BaseModel]],
    random_state: int | None,
    ratio: float,
    orig_to_enc_cols: dict[str, dict[str, list[str]]],
):
    #
    # Fit models
    #
    data = syn or wrk
    data = data["tables"]
    data = {enc: data[enc][name] for enc in data}

    base_args = {
        "data": data,
        "ratio": ratio,
        "random_state": random_state,
    }
    model_info = []
    per_call_args = []
    for cls in classes:
        for col in test_cols:
            drop_x_cols = orig_to_enc_cols[cls.x_trn_type][col]
            for y_col in orig_to_enc_cols[cls.y_trn_type][col]:
                model_info.append({"drop_x_cols": drop_x_cols, "y_col": y_col})
                per_call_args.append(
                    {"cls": cls, "drop_x_cols": drop_x_cols, "y_col": y_col}
                )

    res = process_in_parallel(
        _fit_model_worker, per_call_args, base_args, desc="Fitting models..."
    )

    model_data: list[ModelData] = []
    for info, out in zip(model_info, res):
        model_data.append(ModelData(out, info["drop_x_cols"], info["y_col"]))

    #
    # Score models
    #
    score_info = []
    base_args = {"models": model_data, "random_state": random_state, "ratio": ratio}
    per_call_args = []

    # Training data
    for i, chunks in enumerate(LazyFrame.zip_values(**data)):
        for split in ("train", "test"):
            score_info.append({"data": split})
            per_call_args.append({"i": i, "chunks": chunks, "split": split})

    # Ref data
    for i, chunks in enumerate(
        LazyFrame.zip_values(**{enc: d[name] for enc, d in ref["tables"].items()})
    ):
        score_info.append({"data": "ref"})
        per_call_args.append({"chunks": chunks, "i": i})

    # Wrk data if trained on syn
    if syn:
        for i, chunks in enumerate(
            LazyFrame.zip_values(**{enc: d[name] for enc, d in wrk["tables"].items()})
        ):
            score_info.append({"data": "wrk"})
            per_call_args.append({"chunks": chunks, "i": i})

    res = process_in_parallel(
        _score_models_worker, per_call_args, base_args, desc="Scoring models..."
    )

    results = []
    for info, out in zip(score_info, res):
        l, scores = out
        for data, score in zip(model_data, scores):
            results.append(
                {
                    "model": data.model.name,
                    "y_col": data.y_col,
                    "train": "syn" if syn else "wrk",
                    "data": info["data"],
                    "length": l,
                    "score": score,
                }
            )

    return pd.DataFrame(results)


def mlflow_log_model_results(name: str, res: pd.DataFrame, comparison: bool):
    import mlflow
    import pandas as pd

    from ....utils.mlflow import gen_html_table

    if not mlflow.active_run():
        return

    if len(res) == 0:
        return

    style = res.style

    if comparison:
        # If comparison, style per column to showcase differences between algorithms
        style = style.background_gradient(axis=0)
    else:
        # If only 1 alg, then gradient all accuracies together and all low percentages
        # (leak, penalty) together.
        style = style.background_gradient(
            axis=None, subset=["train", "test", "ref", "orig"]
        ).background_gradient(axis=None, subset=["leak", "penalty"])

    style = style.format(lambda x: f"{100*cast(float, x):.1f}%").applymap(
        lambda x: "color: transparent; background-color: transparent"
        if pd.isnull(x) or x == 0
        else ""
    )

    mlflow.log_text(
        gen_html_table(style),
        f"models/{name}.html" if name != "table" else "models.html",
    )
    # mlflow.log_text(res.to_csv(), f"logs/_raw/models/{name}.csv")


class ModelMetricFactory(TableMetricFactory):
    def __init__(
        self,
        cls: type["TableMetric"],
        *models: type[BaseModel],
        name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(cls, *models, name=name, **kwargs)
        self.encodings = _get_model_types(models)


class ModelMetric(TableMetric[pd.DataFrame, pd.DataFrame]):
    _factory = ModelMetricFactory
    name = "model"

    def __init__(self, *models: type[BaseModel], **kwargs) -> None:
        super().__init__(**kwargs)
        self.cls = models

    @property
    def encodings(self):
        return _get_model_types(self.cls)

    def fit(
        self,
        table: str,
        meta: Metadata,
        attrs: dict[str, dict[str, Attributes]],
        data: TableData,
    ):
        m = meta[table].metrics.model
        targets = list(dict.fromkeys(m.sensitive + m.targets))
        self.targets = {col: meta[table, col].type for col in targets}
        self.random_state = meta.random_state
        self.table = table

        orig_to_enc_cols = {}
        for type, attr in attrs.items():
            cols = {n: list(a.vals.keys()) for n, a in attr[table].items()}
            orig_to_enc_cols[type] = cols
        self.orig_to_enc_cols = orig_to_enc_cols

    def preprocess(self, wrk: TableData, ref: TableData) -> pd.DataFrame:
        if not self.targets:
            return pd.DataFrame()

        return _process_models(
            self.table,
            self.targets,
            wrk,
            ref,
            None,
            self.cls,
            self.random_state,
            0.2,
            self.orig_to_enc_cols,
        )

    def process(
        self, wrk: TableData, ref: TableData, syn: TableData, pre: pd.DataFrame
    ) -> pd.DataFrame:
        if not self.targets:
            return pd.DataFrame()

        return pd.concat(
            [
                _process_models(
                    self.table,
                    self.targets,
                    wrk,
                    ref,
                    syn,
                    self.cls,
                    self.random_state,
                    0.2,
                    self.orig_to_enc_cols,
                ),
                pre,
            ],
            axis=0,
        )

    def visualise(self, data: dict[str, pd.DataFrame]):
        if not self.targets:
            return

        # Concatenate dataframes
        splits: list[pd.DataFrame] = []
        for name, datum in data.items():
            split = datum.replace({"train": "syn"}, name).replace(
                {"train": "wrk"}, "orig"
            )
            splits.append(split)
        res = pd.concat(splits, axis=0)

        # Create score series
        w_score = res["length"] * res["score"]
        new_data = res.assign(w_score=w_score).drop(columns="score")
        summed = new_data.groupby(["model", "y_col", "train", "data"]).sum()

        score = summed["w_score"] / summed["length"]
        score = score.rename("score").reset_index()

        # Convert to pivot table and rename columns
        pt = score.pivot(index=["y_col", "model", "train"], columns=["data"])
        # Fix columns
        pt = (
            pt.droplevel(None, axis="columns")
            .reindex(columns=["train", "test", "ref", "wrk"])
            .rename(columns={"wrk": "orig"})
            .rename_axis(columns=None)
        )
        # Fix index
        pt = pt.rename_axis(
            index={"y_col": "target", "model": "classifier", "train": "alg"}
        )

        # Add additional columns for privacy leak and synthetic penalty
        leak = pt["orig"] - pt["ref"]
        penalty = pt.xs("orig", level="alg")["ref"] - pt["ref"]
        pt = pt.assign(leak=leak, penalty=penalty)

        # Log to mlflow
        mlflow_log_model_results(self.table, pt, len(data) > 1)

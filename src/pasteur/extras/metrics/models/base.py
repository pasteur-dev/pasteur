from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ....attribute import Attributes
from ....metadata import Metadata
from ....metric import TableMetric, TableMetricFactory, TableData
from ....utils.progress import process_in_parallel
from ....utils import LazyDataset, LazyFrame, LazyChunk

from typing import Literal, NamedTuple


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
    return [
        (
            d.loc[train_idx if split == "train" else test_idx]
            .sort_index()
            .reset_index(drop=True)
        )
        for d in data
    ]


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
        model_data.append(ModelData(out, info["drop_x_cols"], info["y_cols"]))

    return model_data

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
    for chunks in enumerate(
        LazyFrame.zip_values(**{enc: d[name] for enc, d in ref["tables"].items()})
    ):
        score_info.append({"data": "ref"})
        per_call_args.append({"chunks": chunks})

    # Wrk data if trained on syn
    if syn:
        for chunks in enumerate(
            LazyFrame.zip_values(**{enc: d[name] for enc, d in wrk["tables"].items()})
        ):
            score_info.append({"data": "wrk"})
            per_call_args.append({"chunks": chunks})

    res = process_in_parallel(
        _score_models_worker, per_call_args, base_args, desc="Scoring models..."
    )

    results = []
    for info, out in zip(score_info, res):
        l, scores = out
        for data, score in zip(model_data, scores):
            model = data.model

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


def tst():
    # Create train/test sets for splits
    wrk = splits[wrk_set]
    ref = splits[ref_set]
    train = {}
    test = {}
    for name, split in splits.items():
        train[name] = split.train
        test[name] = split.test

    # Create work set which combines work train and test
    assert wrk.test is not None
    wrk_data = {}
    for split in wrk.train:
        wrk_data[split] = pd.concat([wrk.train[split], wrk.test[split]])
    ref_data = ref.train

    base_args = {"wrk": wrk_data, "ref": ref_data, "train": train, "test": test}
    jobs = []
    job_info = []

    # Add work
    for split_name, split in splits.items():
        if split_name == ref_set:
            continue
        assert split.models

        for (name, y_col, model) in split.models:
            if model.size_limit is not None and model.size_limit < len(
                next(iter(wrk_data.values()))
            ):
                continue

            for orig_col, col_type in test_cols.items():
                # Filter, say, categorical columns from regression models.
                if model.y_col_types is not None and col_type not in model.y_col_types:
                    continue

                drop_x_cols = orig_to_enc_cols[model.x_trn_type][orig_col]
                for y_col in orig_to_enc_cols[model.y_trn_type][orig_col]:
                    jobs.append(
                        {
                            "model": model,
                            "drop_x_cols": drop_x_cols,
                            "y_col": y_col,
                            "split": split_name,
                            "test_wrk": split_name != wrk_set,
                        }
                    )
                    job_info.append(
                        {
                            "model": name,
                            "train_data": split_name,
                            "target": y_col,
                        }
                    )
    scores = process_in_parallel(_score_model, jobs, base_args, 1, "Scoring models")

    all_scores = pd.concat([pd.DataFrame(job_info), pd.DataFrame(scores)], axis=1)
    wrk_scores = (
        all_scores[all_scores["train_data"] == wrk_set]
        .drop(columns=["train_data", "wrk"])
        .rename(
            columns={
                "train": "orig_train",
                "test": "orig_test",
                "ref": "orig_test_real",
            }
        )
    )
    alg_scores = all_scores[all_scores["train_data"] != wrk_set].rename(
        columns={
            "train_data": "alg",
            "train": "synth_train",
            "test": "synth_test",
            "wrk": "synth_test_orig",
            "ref": "synth_test_real",
        }
    )

    final_scores = alg_scores.merge(wrk_scores, on=["model", "target"])
    # Reorder columns
    final_scores = final_scores[
        [
            "alg",
            "model",
            "target",
            "orig_train",
            "synth_train",
            "orig_test",
            "synth_test",
            "orig_test_real",
            "synth_test_real",
            "synth_test_orig",
        ]
    ]

    if comparison:
        return final_scores.set_index(["target", "model", "alg"])
    return final_scores.drop(columns=["alg"]).set_index(["target", "model"])


def mlflow_log_model_results(name: str, res: pd.DataFrame):
    import mlflow
    import pandas as pd

    from ....utils.mlflow import gen_html_table

    if not mlflow.active_run():
        return

    if len(res) == 0:
        return

    res = res.copy()

    res["privacy_leak"] = res["synth_test_orig"] - res["synth_test_real"]
    res["synth_penalty"] = res["orig_test_real"] - res["synth_test_real"]
    style = (
        res.style.format(lambda x: f"{100*cast(float, x):.1f}%")
        .background_gradient(axis=0)
        .applymap(
            lambda x: "color: transparent; background-color: transparent"
            if pd.isnull(x)
            else ""
        )
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

    def process(self, wrk: TableData, ref: TableData, syn: TableData, pre: pd.DataFrame) -> pd.DataFrame:
        if not self.targets:
            return pd.DataFrame()

        return pd.concat([_process_models(
            self.table,
            self.targets,
            wrk,
            ref,
            syn,
            self.cls,
            self.random_state,
            0.2,
            self.orig_to_enc_cols,
        ), pre], axis=0) 

    def visualise(
        self,
        data: dict[str, pd.DataFrame]
    ):
        if not self.targets:
            return

        res = score_models(
            data, self.targets, self.orig_to_enc_cols, wrk_set, ref_set, comparison
        )
        mlflow_log_model_results(self.table, res)

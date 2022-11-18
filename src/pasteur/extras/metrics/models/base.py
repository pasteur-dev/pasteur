from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ....attribute import Attributes
from ....metadata import Metadata
from ....metric import TableMetric, TableMetricFactory
from ....utils.progress import process_in_parallel

from typing import NamedTuple


class BaseModel:
    name: str
    x_trn_type: str
    y_trn_type: str
    size_limit: int | None = None
    y_col_types: list[str] | None = None

    def __init__(self, random_state: int):
        self.random_state = random_state

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        raise NotImplementedError()

    def score(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        raise NotImplementedError()


def _get_model_types(cls: tuple[type[BaseModel]]):
    types = set()
    for c in cls:
        types.add(c.x_trn_type)
        types.add(c.y_trn_type)

    return sorted(list(types))


def _fit_model(
    train: dict[str, pd.DataFrame],
    cls: type[BaseModel],
    drop_x_cols: list[str],
    y_col: str,
    random_state: int,
):
    model = cls(random_state=random_state)
    model.fit(
        train[cls.x_trn_type].drop(columns=drop_x_cols), train[cls.y_trn_type][[y_col]]
    )
    return model


class TrainedModel(NamedTuple):
    name: str
    y_col: str
    model: BaseModel


class ModelData(NamedTuple):
    models: list[TrainedModel] | None
    train: dict[str, pd.DataFrame]
    test: dict[str, pd.DataFrame] | None


def _score_model(
    train: dict[str, dict[str, pd.DataFrame]],
    test: dict[str, dict[str, pd.DataFrame]],
    wrk: dict[str, pd.DataFrame],
    ref: dict[str, pd.DataFrame],
    model: BaseModel,
    drop_x_cols: list[str],
    y_col: str,
    split: str,
    test_wrk: bool,
):
    sets = {
        "train": train[split],
        "test": test[split],
        "ref": ref,
    }
    if test_wrk:
        sets["wrk"] = wrk
    x_sets = {n: s[model.x_trn_type].drop(columns=drop_x_cols) for n, s in sets.items()}
    y_sets = {n: s[model.y_trn_type][[y_col]] for n, s in sets.items()}
    return {n: model.score(x_sets[n], y_sets[n]) for n in sets}


def train_models(
    name: str,
    test_cols: dict[str, str],
    data: dict[str, dict[str, pd.DataFrame]],
    ids: pd.DataFrame | None,
    classes: tuple[type[BaseModel]],
    random_state: int | None,
    ratio: float,
    orig_to_enc_cols: dict[str, dict[str, list[str]]],
):
    train = {}
    test = {}

    index = next(iter(data.values()))[name].index
    train_idx, test_idx = train_test_split(
        index,
        test_size=ratio,
        random_state=random_state if random_state else None,
    )

    # Create train, test sets, and remove index from dataframes.
    for type, tables in data.items():
        # TODO: expand tables
        train[type] = tables[name].loc[train_idx].sort_index().reset_index(drop=True)
        test[type] = tables[name].loc[test_idx].sort_index().reset_index(drop=True)

    base_args = {"train": train}
    jobs = []
    job_info = []
    i = 0
    for cls in classes:
        # TODO: Handle work and synth set with diff lengths hitting size limit differently
        if cls.size_limit is not None and cls.size_limit < len(index):
            continue

        for orig_col, col_type in test_cols.items():
            # Filter, say, categorical columns from regression models.
            if cls.y_col_types is not None and col_type not in cls.y_col_types:
                continue

            drop_x_cols = orig_to_enc_cols[cls.x_trn_type][orig_col]
            for y_col in orig_to_enc_cols[cls.y_trn_type][orig_col]:
                jobs.append(
                    {
                        "cls": cls,
                        "drop_x_cols": drop_x_cols,
                        "y_col": y_col,
                        "random_state": (random_state + i) if random_state else None,
                    }
                )
                job_info.append((cls.name, y_col))
                i += 1

    models = process_in_parallel(
        _fit_model, jobs, base_args, 1, "Fitting models to data"
    )

    trained_models = []
    for (name, y_col), model in zip(job_info, models):
        trained_models.append(TrainedModel(name, y_col, model))

    return ModelData(trained_models, train, test)


def score_models(
    splits: dict[str, ModelData],
    test_cols: dict[str, str],
    orig_to_enc_cols: dict[str, dict[str, list[str]]],
    wrk_set: str = "wrk",
    ref_set: str = "ref",
    comparison: bool = False,
) -> pd.DataFrame:

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


class ModelMetric(TableMetric[ModelData]):
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
        tables: dict[str, dict[str, pd.DataFrame]],
        ids: pd.DataFrame | None = None,
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

    def process(
        self,
        split: int,
        tables: dict[str, dict[str, pd.DataFrame]],
        ids: pd.DataFrame | None = None,
    ):
        if not self.targets:
            return ModelData(None, {}, None)

        if split == self.REF_SPLIT:
            return ModelData(
                None, {name: split[self.table] for name, split in tables.items()}, None
            )
        return train_models(
            self.table,
            self.targets,
            tables,
            ids,
            self.cls,
            self.random_state,
            0.2,
            self.orig_to_enc_cols,
        )

    def visualise(
        self,
        data: dict[str, ModelData],
        comparison: bool = False,
        wrk_set: str = "wrk",
        ref_set: str = "ref",
    ):
        if not self.targets:
            return

        res = score_models(
            data, self.targets, self.orig_to_enc_cols, wrk_set, ref_set, comparison
        )
        mlflow_log_model_results(self.table, res)

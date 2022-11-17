from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ....attribute import Attributes
from ....metadata import Metadata
from ....metric import TableMetric, TableMetricFactory
from ....utils.progress import process_in_parallel


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
    sets: dict[str, pd.DataFrame],
    cls: type[BaseModel],
    drop_x_cols: list[str],
    y_col: str,
    random_state: int,
):
    model = cls(random_state=random_state)
    model.fit(
        sets[cls.x_trn_type].drop(columns=drop_x_cols), sets[cls.y_trn_type][[y_col]]
    )
    return model


def _score_model(
    sets: dict[str, dict[str, pd.DataFrame]],
    model: BaseModel,
    drop_x_cols: list[str],
    y_col: str,
):
    eval_sets = list(sets[model.x_trn_type].keys())
    x_sets = {n: sets[model.x_trn_type][n].drop(columns=drop_x_cols) for n in eval_sets}
    y_sets = {n: sets[model.y_trn_type][n][[y_col]] for n in eval_sets}
    return {n: model.score(x_sets[n], y_sets[n]) for n in eval_sets}


def _calculate_score(
    sets: dict[str, dict[str, pd.DataFrame]],
    cls: type[BaseModel],
    train_set: str,
    eval_sets: dict[str, str],
    drop_x_cols: list[str],
    y_col: str,
    random_state: int,
):
    x_sets = {n: sets[cls.x_trn_type][n].drop(columns=drop_x_cols) for n in eval_sets}
    y_sets = {n: sets[cls.y_trn_type][n][[y_col]] for n in eval_sets}

    model = cls(random_state=random_state)
    model.fit(x_sets[train_set], y_sets[train_set])

    return {n: model.score(x_sets[k], y_sets[k]) for k, n in eval_sets.items()}


def calculate_model_scores(
    models: tuple[type[BaseModel]],
    random_state: int | None,
    ratio: float,
    test_cols: dict[str, str],
    sets: dict[str, dict[str, pd.DataFrame]],
    orig_to_enc_cols: dict[str, dict[str, list[str]]],
    wrk_set: str = "wrk",
    ref_set: str = "ref",
    comparison: bool = False,
) -> pd.DataFrame:
    types = list(sets.keys())
    splits = list(next(iter(sets.values())))
    new_sets = {t: {} for t in types}

    # Create train, test sets, and remove index from dataframes.
    for i, train_data in enumerate(splits):
        if train_data == ref_set:
            continue
        index = sets[types[0]][train_data].index

        train_idx, test_idx = train_test_split(
            index,
            test_size=ratio,
            random_state=(random_state + i) if random_state else None,
        )

        for t in types:
            new_sets[t][f"{train_data}_train"] = (
                sets[t][train_data].loc[train_idx].sort_index().reset_index(drop=True)
            )
            new_sets[t][f"{train_data}_test"] = (
                sets[t][train_data].loc[test_idx].sort_index().reset_index(drop=True)
            )

    for t in types:
        for s in (wrk_set, ref_set):
            # Sort and nuke the index to avoid issues with sklearn drawing from
            # dataframes that are sorted differently
            new_sets[t][s] = sets[t][s].sort_index().reset_index(drop=True)

    # # TODO: Recheck this speeds up the calculations and works
    # for t in new_sets:
    #     for s in new_sets[t]:
    #         if t == "num":
    #             new_sets[t][s] = new_sets[t][s].astype("float16")
    #         elif t == "idx":
    #             new_sets[t][s] = new_sets[t][s].astype("uint16")

    base_args = {"sets": new_sets}
    jobs = []
    job_info = []
    i = 0
    for cls in models:
        if cls.size_limit is not None and cls.size_limit < len(sets[types[0]]["wrk"]):
            continue

        for orig_col, col_type in test_cols.items():
            # Filter, say, categorical columns from regression models.
            if cls.y_col_types is not None and col_type not in cls.y_col_types:
                continue

            drop_x_cols = orig_to_enc_cols[cls.x_trn_type][orig_col]
            for y_col in orig_to_enc_cols[cls.y_trn_type][orig_col]:
                for train_data in splits:
                    if train_data == ref_set:
                        continue
                    train_set = f"{train_data}_train"

                    eval_sets = {
                        f"{train_data}_train": "train",
                        f"{train_data}_test": "test",
                        ref_set: ref_set,
                    }
                    if train_data not in (ref_set, wrk_set):
                        eval_sets[wrk_set] = wrk_set

                    jobs.append(
                        {
                            "cls": cls,
                            "train_set": train_set,
                            "eval_sets": eval_sets,
                            "drop_x_cols": drop_x_cols,
                            "y_col": y_col,
                            "random_state": (random_state + i)
                            if random_state
                            else None,
                        }
                    )
                    job_info.append(
                        {
                            "model": cls.name,
                            "train_data": train_data,
                            "target": y_col,
                        }
                    )

                    i += 1

    scores = process_in_parallel(
        _calculate_score, jobs, base_args, 1, "Fitting models to data"
    )

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


ModelData = tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame | None] | tuple


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
        if self.targets:
            return (tables, ids)
        return ()

    def visualise(
        self,
        data: dict[str, ModelData],
        comparison: bool = False,
        wrk_set: str = "wrk",
        ref_set: str = "ref",
    ):
        if not self.targets:
            return
        sets = {}
        for split_name, split in data.items():
            for type, tables in split[0].items():
                if type not in sets:
                    sets[type] = {}
                # TODO: Add table expansion
                sets[type][split_name] = tables[self.table]

        res = calculate_model_scores(
            self.cls,
            self.random_state,
            0.2,
            self.targets,
            sets,
            self.orig_to_enc_cols,
            wrk_set,
            ref_set,
            comparison,
        )
        mlflow_log_model_results(self.table, res)

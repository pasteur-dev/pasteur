import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ...progress import process_in_parallel
from ...transform import TableTransformer
from .models import BaseModel, get_models, get_required_types


def _calculate_score(
    sets: dict[str, dict[str, pd.DataFrame]],
    cls: type[BaseModel],
    train_set: str,
    eval_sets: list[str],
    drop_x_cols: list[str],
    y_col: str,
    random_state: int,
):
    x_sets = {n: sets[cls.x_trn_type][n].drop(columns=drop_x_cols) for n in eval_sets}
    y_sets = {n: sets[cls.y_trn_type][n][[y_col]] for n in eval_sets}

    model = cls(random_state=random_state)
    model.fit(x_sets[train_set], y_sets[train_set])

    return {n: model.score(x_sets[n], y_sets[n]) for n in eval_sets}


def _enc_to_orig_cols(
    mapping: dict[str, dict[str, list[str]]]
) -> dict[str, dict[str, str]]:
    """Receives a set of dictionaries of 1 to many mappings of original columns
    to encoded columns and returns a set of 1-1 mappings of encoded to original
    columns."""
    out = {}
    for type, type_map in mapping.items():
        out[type] = {}
        for orig, enc_cols in type_map.items():
            for col in enc_cols:
                out[type][col] = orig

    return out


def node_calculate_model_scores(transformer: TableTransformer, **tables: pd.DataFrame):
    types = get_required_types()
    meta = transformer.meta
    table_name = transformer.name

    orig_to_enc_cols = {}
    sets = {}

    for type in types:
        sets[type] = {}

        attr = transformer[type].get_attributes()
        cols = {n: list(a.cols.keys()) for n, a in attr.items()}
        orig_to_enc_cols[type] = cols

    for arg_name, table in tables.items():
        type, split, name = arg_name.split(".")
        if name == table_name:
            sets[type][split] = table

    model_meta = meta[table_name].metrics.model
    test_cols = list(dict.fromkeys(model_meta.sensitive + model_meta.targets))
    test_cols = {col: meta[table_name, col].type for col in test_cols}

    if len(test_cols) == 0:
        return pd.DataFrame()

    # TODO: Add varioable ratio to metadata
    return calculate_model_scores(meta.seed, 0.2, test_cols, sets, orig_to_enc_cols)


def calculate_model_scores(
    random_state: int,
    ratio: float,
    test_cols: dict[str, str],
    sets: dict[str, dict[str, pd.DataFrame]],
    orig_to_enc_cols: dict[str, dict[str, list[str]]],
):
    types = list(sets.keys())
    new_sets = {t: {} for t in types}

    # Create train, test sets, and remove index from dataframes.
    for i, train_data in enumerate(("wrk", "syn")):
        index = sets[types[0]][train_data].index

        train_idx, test_idx = train_test_split(
            index, test_size=ratio, random_state=random_state + i
        )

        for t in types:
            new_sets[t][f"{train_data}_train"] = (
                sets[t][train_data].loc[train_idx].sort_index().reset_index(drop=True)
            )
            new_sets[t][f"{train_data}_test"] = (
                sets[t][train_data].loc[test_idx].sort_index().reset_index(drop=True)
            )

    for t in types:
        for s in ("wrk", "ref"):
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
    for model, cls in get_models().items():
        if cls.size_limit is not None and cls.size_limit < len(sets[types[0]]["wrk"]):
            continue

        for orig_col, col_type in test_cols.items():
            # Filter, say, categorical columns from regression models.
            if cls.y_col_types is not None and col_type not in cls.y_col_types:
                continue

            drop_x_cols = orig_to_enc_cols[cls.x_trn_type][orig_col]
            for y_col in orig_to_enc_cols[cls.y_trn_type][orig_col]:
                for train_data in ("wrk", "syn"):
                    train_set = f"{train_data}_train"

                    eval_sets = [f"{train_data}_train", f"{train_data}_test", "ref"]
                    if train_data == "syn":
                        eval_sets.extend(["wrk"])

                    jobs.append(
                        {
                            "cls": cls,
                            "train_set": train_set,
                            "eval_sets": eval_sets,
                            "drop_x_cols": drop_x_cols,
                            "y_col": y_col,
                            "random_state": random_state + i,
                        }
                    )
                    job_info.append(
                        {
                            "model": model,
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
        all_scores[all_scores["train_data"] == "wrk"]
        .drop(columns=["syn_train", "syn_test", "train_data", "wrk"])
        .rename(
            columns={
                "wrk_train": "orig_train",
                "wrk_test": "orig_test",
                "ref": "orig_test_real",
            }
        )
    )
    alg_scores = (
        all_scores[all_scores["train_data"] == "syn"]
        .drop(columns=["wrk_train", "wrk_test", "train_data"])
        .rename(
            columns={
                "syn_train": "synth_train",
                "syn_test": "synth_test",
                "wrk": "synth_test_orig",
                "ref": "synth_test_real",
            }
        )
    )

    final_scores = alg_scores.merge(wrk_scores, on=["model", "target"])
    # Reorder columns
    final_scores = final_scores[
        [
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
    return final_scores.set_index(["target", "model"])

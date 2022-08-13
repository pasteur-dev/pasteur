import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .models import BaseModel, get_models

from ...progress import process_in_parallel
from ...transform import TableTransformer


def _calculate_score(
    sets: dict[str, dict[str, pd.DataFrame]],
    cls: type[BaseModel],
    train_set: str,
    eval_sets: list[str],
    drop_x_cols: list[str],
    y_col: str,
):
    x_sets = {n: s.drop(columns=drop_x_cols) for n, s in sets[cls.x_trn_type].items()}
    y_sets = {n: s[[y_col]] for n, s in sets[cls.y_trn_type].items()}

    model = cls()
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
    types = transformer.types
    meta = transformer.meta
    table_name = transformer.name

    orig_to_enc_cols = {}
    sets = {}

    for type in types:
        sets[type] = {}
        orig_to_enc_cols[type] = transformer[type].get_col_mapping()

    for arg_name, table in tables.items():
        type, split, name = arg_name.split(".")
        if name == table_name:
            sets[type][split] = table

    test_cols = list(meta.get_table(name).cols.keys())

    # TODO: Add varioable ratio to metadata
    return calculate_model_scores(meta.seed, 0.2, test_cols, sets, orig_to_enc_cols)


def calculate_model_scores(
    random_state: int,
    ratio: float,
    test_cols: list[str],
    sets: dict[str, dict[str, pd.DataFrame]],
    orig_to_enc_cols: dict[str, dict[str, list[str]]],
):

    types = list(sets.keys())
    new_sets = {t: {} for t in types}

    for i, train_data in enumerate(("wrk", "alg")):
        orig_sets = [sets[t][train_data] for t in types]
        split_sets = train_test_split(
            orig_sets, test_size=ratio, random_state=random_state + i
        )
        for i, t in enumerate(types):
            new_sets[t][f"{train_data}_train"] = split_sets[i]
            new_sets[t][f"{train_data}_test"] = split_sets[len(types) + i]

    for t in types:
        for s in ("wrk", "dev", "val"):
            new_sets[t][s] = sets[t][s]

    base_args = {"sets": new_sets}
    jobs = []
    job_info = []
    for model, cls in get_models().items():
        for orig_col in test_cols:
            drop_x_cols = orig_to_enc_cols[cls.x_trn_type][orig_col]
            for y_col in orig_to_enc_cols[cls.y_trn_type][orig_col]:
                for train_data in ("wrk", "alg"):
                    train_set = f"{train_data}_train"

                    eval_sets = [
                        f"{train_data}_train",
                        f"{train_data}_test",
                        "dev",
                        "val",
                    ]
                    if train_data == "alg":
                        eval_sets.append("wrk")

                    jobs.append(
                        {
                            "cls": cls,
                            "train_set": train_set,
                            "eval_sets": eval_sets,
                            "drop_x_cols": drop_x_cols,
                            "y_col": y_col,
                        }
                    )
                    job_info.append(
                        {
                            "model": model,
                            "train_data": train_data,
                            "orig_col": orig_col,
                            "enc_col": y_col,
                        }
                    )

    scores = process_in_parallel(
        _calculate_score, jobs, base_args, 5, "Fitting models to data"
    )
    pass

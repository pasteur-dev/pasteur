from typing import Collection

import pandas as pd
from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...metadata import Metadata
from ...transform import TableTransformer
from ...views import View
from .utils import gen_closure


def _fit_table(
    name: str,
    types: Collection[str],
    meta: Metadata,
    **tables: dict[str, pd.DataFrame],
):
    t = TableTransformer(meta, name, types)
    t.fit(tables)
    return t


def _find_ids(transformer: TableTransformer, **tables: dict[str, pd.DataFrame]):
    return transformer.find_ids(tables)


def _transform_table(
    type: str,
    transformer: TableTransformer,
    ids: pd.DataFrame,
    **tables: dict[str, pd.DataFrame],
):
    return transformer[type].transform(tables, ids)


def _reverse_table(
    type: str,
    transformer: TableTransformer,
    ids: pd.DataFrame,
    table: pd.DataFrame,
    **parents: dict[str, pd.DataFrame],
):
    return transformer[type].reverse(table, ids, parents)


def create_transform_pipeline(
    view: View,
    split: str,
    types: list[str] | str = [],
    trn_split: str | None = None,
    gen_ids: bool = True,
):
    if isinstance(types, str):
        types = [types]
    table_nodes = []

    for t in view.tables:
        # Allow using an existing transformer with trn_split
        if trn_split is None:
            table_nodes += [
                node(
                    func=gen_closure(
                        _fit_table, t, types, _fn=f"fit_transformer_to_{t}"
                    ),
                    inputs={
                        "meta": "metadata",
                        **{t: t for t in view.tables},
                    },
                    outputs=f"trn_{t}",
                ),
            ]
        if gen_ids:
            table_nodes += [
                node(
                    func=gen_closure(_find_ids, _fn=f"generate_{t}_ids"),
                    inputs={"transformer": f"trn_{t}", **{t: t for t in view.tables}},
                    outputs=f"ids_{t}",
                ),
            ]

        table_nodes += [
            node(
                func=gen_closure(_transform_table, type, _fn=f"transform_{type}_{t}"),
                inputs={
                    "transformer": f"trn_{t}",
                    "ids": f"ids_{t}",
                    **{t: t for t in view.tables},
                },
                outputs=f"{type}_{t}",
            )
            for type in types
        ]

    if trn_split is None:
        inputs = {"metadata": f"{view}.view.metadata"}
    else:
        inputs = {f"trn_{t}": f"{view}.{trn_split}.trn_{t}" for t in view.tables}

    return modular_pipeline(
        pipe=pipeline(table_nodes),
        namespace=f"{view}.{split}",
        inputs=inputs,
    )


def create_reverse_pipeline(view: View, alg: str, type: str, trn_split: str):
    decode_pipe = pipeline(
        [
            node(
                func=gen_closure(_reverse_table, type, _fn=f"reverse_transform_{t}"),
                inputs={
                    "transformer": f"trn_{t}",
                    "ids": f"ids_{t}",
                    "table": f"enc_{t}",
                    **{req: req for req in view.trn_deps.get(t, [])},
                },
                outputs=t,
            )
            for t in view.tables
        ]
    )

    return modular_pipeline(
        pipe=decode_pipe,
        namespace=f"{view}.{alg}",
        inputs={
            **{f"trn_{t}": f"{view}.{trn_split}.trn_{t}" for t in view.tables},
        },
    )

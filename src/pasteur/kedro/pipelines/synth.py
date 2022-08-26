from typing import Collection, Dict

import pandas as pd
from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...metadata import Metadata
from ...synth import synth_fit, synth_sample
from ...transform import TableTransformer
from .utils import gen_closure


def fit_table(
    name: str,
    types: Collection[str],
    meta: Metadata,
    **tables: dict[str, pd.DataFrame],
):
    t = TableTransformer(meta, name, types)
    t.fit(tables)
    return t


def find_ids(transformer: TableTransformer, **tables: dict[str, pd.DataFrame]):
    return transformer.find_ids(tables)


def transform_table(
    type: str,
    transformer: TableTransformer,
    ids: pd.DataFrame,
    **tables: dict[str, pd.DataFrame],
):
    return transformer[type].transform(tables, ids)


def reverse_table(
    type: str,
    transformer: TableTransformer,
    ids: pd.DataFrame,
    table: pd.DataFrame,
    **parents: dict[str, pd.DataFrame],
):
    return transformer[type].reverse(table, ids, parents)


def create_transform_pipeline(
    view: str,
    split: str,
    tables: Collection[str],
    types: Collection[str] | str = [],
    trn_split: str | None = None,
    gen_ids: bool = True,
):
    if isinstance(types, str):
        types = [types]
    tables = [t.split(".")[-1] for t in tables]
    table_nodes = []

    for t in tables:
        # Allow using an existing transformer with trn_split
        if trn_split is None:
            table_nodes += [
                node(
                    func=gen_closure(
                        fit_table, t, types, _fn=f"fit_transformer_to_{t}"
                    ),
                    inputs={
                        "meta": "metadata",
                        **{t: t for t in tables},
                    },
                    outputs=f"trn_{t}",
                ),
            ]
        if gen_ids:
            table_nodes += [
                node(
                    func=find_ids,
                    inputs={"transformer": f"trn_{t}", **{t: t for t in tables}},
                    outputs=f"ids_{t}",
                ),
            ]

        table_nodes += [
            node(
                func=gen_closure(transform_table, type, _fn=f"transform_{type}_{t}"),
                inputs={
                    "transformer": f"trn_{t}",
                    "ids": f"ids_{t}",
                    **{t: t for t in tables},
                },
                outputs=f"{type}_{t}",
            )
            for type in types
        ]

    if trn_split is None:
        inputs = {"metadata": f"{view}.metadata"}
    else:
        inputs = {f"trn_{t}": f"{view}.{trn_split}.trn_{t}" for t in tables}

    return modular_pipeline(
        pipe=pipeline(table_nodes),
        namespace=f"{view}.{split}",
        inputs=inputs,
    )


def create_synth_pipeline(
    view: str,
    split: str,
    cls: str,
    tables: Collection[str],
    requirements: Dict[str, Collection[str]] | None = None,
):
    alg = cls.name
    type = cls.type
    if requirements is None:
        requirements = {}

    synth_pipe = pipeline(
        [
            node(
                func=gen_closure(synth_fit, cls),
                inputs={
                    **{f"trn_{t}": f"trn_{t}" for t in tables},
                    **{f"ids_{t}": f"in_ids_{t}" for t in tables},
                    **{f"enc_{t}": f"in_enc_{t}" for t in tables},
                },
                outputs="model",
            ),
            node(
                func=synth_sample,
                inputs="model",
                outputs={
                    **{f"ids_{t}": f"ids_{t}" for t in tables},
                    **{f"enc_{t}": f"enc_{t}" for t in tables},
                },
            ),
        ]
    )

    decode_pipe = pipeline(
        [
            node(
                func=gen_closure(reverse_table, type, _fn=f"reverse_transform_{t}"),
                inputs={
                    "transformer": f"trn_{t}",
                    "ids": f"ids_{t}",
                    "table": f"enc_{t}",
                    **{req: req for req in requirements.get(t, [])},
                },
                outputs=t,
            )
            for t in tables
        ]
    )

    return modular_pipeline(
        pipe=synth_pipe + decode_pipe,
        namespace=f"{view}.{alg}",
        inputs={
            **{f"in_enc_{t}": f"{view}.{split}.{type}_{t}" for t in tables},
            **{f"in_ids_{t}": f"{view}.{split}.ids_{t}" for t in tables},
            **{f"trn_{t}": f"{view}.{split}.trn_{t}" for t in tables},
        },
    )

from typing import Collection, Dict

import pandas as pd
from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...metadata import Metadata
from ...synth import synth_fit_closure, synth_sample
from ...transform import TableTransformer
from ...utils import get_params_for_pipe


def fit_table_closure(view: str, name: str, types: Collection[str]):
    def fit_table(params: dict, **tables: dict[str, pd.DataFrame]):
        meta_dict = get_params_for_pipe(view, params)
        meta = Metadata(meta_dict, tables)
        t = TableTransformer(meta, name, types)
        t.fit(tables)
        return t

    return fit_table


def find_ids(transformer: TableTransformer, **tables: dict[str, pd.DataFrame]):
    return transformer.find_ids(tables)


def transform_table_closure(type: str):
    def transform_table(
        transformer: TableTransformer,
        ids: pd.DataFrame,
        **tables: dict[str, pd.DataFrame],
    ):
        return transformer[type].transform(tables, ids)

    return transform_table


def reverse_table_closure(type: str):
    def reverse_table(
        transformer: TableTransformer,
        ids: pd.DataFrame,
        table: pd.DataFrame,
        **parents: dict[str, pd.DataFrame],
    ):
        return transformer[type].reverse(table, ids, parents)

    return reverse_table


def transform_table_tab(
    transformer: TableTransformer, **tables: dict[str, pd.DataFrame]
):
    table = transformer.transform(tables)
    return table


def reverse_table_tab(
    transformer: TableTransformer,
    table: pd.DataFrame,
    **parents: dict[str, pd.DataFrame],
):
    return transformer.reverse(table, None, parents)


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
                    func=fit_table_closure(view, t, types),
                    inputs={
                        "params": "parameters",
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
                func=transform_table_closure(type),
                inputs={
                    "transformer": f"trn_{t}",
                    "ids": f"ids_{t}",
                    **{t: t for t in tables},
                },
                outputs=f"{type}_{t}",
            )
            for type in types
        ]

    inputs = (
        {}
        if trn_split is None
        else {f"trn_{t}": f"{view}.{trn_split}.trn_{t}" for t in tables}
    )

    return modular_pipeline(
        pipe=pipeline(table_nodes),
        namespace=f"{view}.{split}",
        inputs=inputs,
        parameters={
            "parameters": f"parameters",
        }
        if trn_split is None
        else {},
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
                func=synth_fit_closure(cls),
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
                func=reverse_table_closure(type),
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

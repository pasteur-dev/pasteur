import pandas as pd
from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...metadata import Metadata
from ...transform import TableTransformer
from ...views import View
from .utils import gen_closure


def _fit_table(
    name: str,
    types: list[str],
    meta: Metadata,
    **tables: dict[str, pd.DataFrame],
):
    t = TableTransformer(meta, name, types)
    tables, ids = t.fit_transform(tables)
    return t, ids


def _base_transform_table(
    transformer: TableTransformer,
    **tables: dict[str, pd.DataFrame],
):
    ids = transformer.find_ids()
    return transformer.transform(tables, ids), ids


def _base_reverse_table(
    transformer: TableTransformer,
    ids: pd.DataFrame,
    table: pd.DataFrame,
    **parents: dict[str, pd.DataFrame],
):
    return transformer.reverse(table, ids, parents)


def _enc_transform_table(type: str, transformer: TableTransformer, table: pd.DataFrame):
    return transformer[type].transform(table)


def _enc_reverse_table(type: str, transformer: TableTransformer, table: pd.DataFrame):
    return transformer[type].reverse(table)


def create_transformers_pipeline(view: View, types: list[str]):
    return pipeline(
        [
            node(
                func=gen_closure(_fit_table, types, _fn=f"fit_transformer_to_{table}"),
                inputs={
                    "meta": f"{view}.metadata",
                    **{t: f"{view}.view.{t}" for t in view.tables},
                },
                outputs=[f"{view}.trn.{table}", f"{view}.trn.ids_{table}"],
                namespace=f"{view}.trn",
            )
            for table in view.tables
        ]
    )


def create_transform_pipeline(
    view: View, split: str, types: list[str], only_encode: bool = False
):
    table_nodes = []

    for t in view.tables:
        if not only_encode:
            table_nodes += [
                node(
                    func=gen_closure(_base_transform_table, _fn=f"transform_{t}"),
                    inputs={
                        "transformer": f"trn_{t}",
                        **{t: t for t in view.tables},
                    },
                    outputs=[f"bst_{t}", f"ids_{t}"],
                ),
            ]

        for type in types:
            table_nodes += [
                node(
                    func=gen_closure(_enc_transform_table, type, _fn=f"encode_{t}"),
                    inputs={
                        "transformer": f"trn_{t}",
                        "table": f"bst_{t}",
                    },
                    outputs=f"{type}_{t}",
                )
            ]

    inputs = {f"trn_{t}": f"{view}.trn.{t}" for t in view.tables}

    return modular_pipeline(
        pipe=pipeline(table_nodes),
        namespace=f"{view}.{split}",
        inputs=inputs,
    )


def create_reverse_pipeline(view: View, alg: str, type: str):
    decode_nodes = []
    for t in view.tables:
        decode_nodes += [
            node(
                func=gen_closure(_enc_reverse_table, type, _fn=f"decode_{t}"),
                inputs={
                    "transformer": f"trn_{t}",
                    "table": f"enc_{t}",
                },
                outputs=f"bst_{t}",
            ),
            node(
                func=gen_closure(_base_reverse_table, _fn=f"reverse_{t}"),
                inputs={
                    "transformer": f"trn_{t}",
                    "ids": f"ids_{t}",
                    "table": f"bst_{t}",
                    **{req: req for req in view.trn_deps.get(t, [])},
                },
                outputs=t,
            ),
        ]

    return modular_pipeline(
        pipe=pipeline(decode_nodes),
        namespace=f"{view}.{alg}",
        inputs={
            **{f"trn_{t}": f"{view}.trn.{t}" for t in view.tables},
        },
    )

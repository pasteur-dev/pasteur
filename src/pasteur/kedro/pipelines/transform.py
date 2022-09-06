from typing import Collection

import pandas as pd
from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...metadata import Metadata
from ...transform import TableTransformer, Attributes, EncodingTransformer
from ...views import View
from .utils import gen_closure


def _fit_table(
    name: str,
    meta: Metadata,
    **tables: dict[str, pd.DataFrame],
):
    t = TableTransformer(meta, name)
    t.fit(tables)
    return t, t.get_attributes()


def _find_ids(transformer: TableTransformer, **tables: dict[str, pd.DataFrame]):
    return transformer.find_ids(tables)


def _transform_table(
    transformer: TableTransformer,
    ids: pd.DataFrame,
    **tables: dict[str, pd.DataFrame],
):
    return transformer.transform(tables, ids)


def _reverse_table(
    transformer: TableTransformer,
    ids: pd.DataFrame,
    table: pd.DataFrame,
    **parents: dict[str, pd.DataFrame],
):
    return transformer.reverse(table, ids, parents)


def _fit_enc_table(
    type: str,
    attrs: Attributes,
    table: pd.DataFrame,
):
    t = EncodingTransformer(type)
    out_attrs = t.fit(attrs, table)
    return t, out_attrs


def _transform_enc_table(transformer: EncodingTransformer, table: pd.DataFrame):
    return transformer.transform(table)


def _reverse_enc_table(transformer: EncodingTransformer, table: pd.DataFrame):
    return transformer.reverse(table)


def _fit_transform_enc_table(
    type: str,
    attrs: Attributes,
    table: pd.DataFrame,
):
    t = EncodingTransformer(type)
    t.fit(attrs, table)
    return t.transform(table)


def create_base_transformers_pipeline(
    view: View,
):
    return pipeline(
        [
            node(
                func=gen_closure(_fit_table, table, _fn=f"fit_transformer_to_{table}"),
                inputs={
                    "meta": f"{view}.metadata",
                    **{t: f"{view}.view.{t}" for t in view.tables},
                },
                outputs=[f"{view}.trn.bst_{table}", f"{view}.trn.atr_{table}"],
                namespace=f"{view}.trn",
            )
            for table in view.tables
        ]
    )


def create_transform_pipeline(
    view: View,
    split: str,
    gen_ids: bool = True,
):
    table_nodes = []

    for t in view.tables:
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
                func=gen_closure(_transform_table, _fn=f"transform_{t}"),
                inputs={
                    "transformer": f"trn_{t}",
                    "ids": f"ids_{t}",
                    **{t: t for t in view.tables},
                },
                outputs=f"enc_{t}",
            )
        ]

        inputs = {f"trn_{t}": f"{view}.trn.bst_{t}" for t in view.tables}

    return modular_pipeline(
        pipe=pipeline(table_nodes),
        namespace=f"{view}.{split}",
        inputs=inputs,
    )


def create_reverse_pipeline(view: View, alg: str):
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
            **{f"trn_{t}": f"{view}.trn.bst_{t}" for t in view.tables},
        },
    )


def _create_measure_transform_pipeline(view: View, split: str, type: str):
    table_nodes = []

    for t in view.tables:
        table_nodes += [
            node(
                func=gen_closure(
                    _fit_transform_enc_table, type, _fn=f"encode_{t}_to_{type}"
                ),
                inputs=[f"{view}.trn.atr_{t}", f"{view}.{split}.enc_{t}"],
                outputs=f"{view}.{split}.msr_{type}_{t}",
            )
        ]
    return pipeline(table_nodes)


def create_measure_transform_pipelines(view: View, splits: list[str], types: list[str]):
    pipe = pipeline([])
    for type in types:
        for split in splits:
            pipe += _create_measure_transform_pipeline(view, split, type)

    return pipe

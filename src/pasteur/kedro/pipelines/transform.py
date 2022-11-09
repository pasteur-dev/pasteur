from __future__ import annotations

from typing import TYPE_CHECKING

from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from .module import DatasetMeta as D
from .module import PipelineMeta

if TYPE_CHECKING:
    import pandas as pd
    from ...metadata import Metadata
    from ...table import TableHandler
    from ...transform import Transformer
    from ...encode import Encoder
    from ...view import View

from .utils import gen_closure


def _fit_table(
    name: str,
    transformers: dict[str, type[Transformer]],
    encoders: dict[str, type[Transformer]],
    meta: Metadata,
    **tables: dict[str, pd.DataFrame],
):
    from ...table import TableHandler

    t = TableHandler(meta, name, encoders, transformers)
    tables, ids = t.fit_transform(tables)
    return t, ids


def _transform_table(
    transformer: TableHandler,
    **tables: dict[str, pd.DataFrame],
):
    ids = transformer.find_ids(tables)
    return transformer.transform(tables, ids), ids


def _base_reverse_table(
    transformer: TableHandler,
    ids: pd.DataFrame,
    table: pd.DataFrame,
    **parents: dict[str, pd.DataFrame],
):
    return transformer.reverse(table, ids, parents)


def _encode_table(type: str, transformer: TableHandler, table: pd.DataFrame):
    return transformer[type].encode(table)


def _decode_table(type: str, transformer: TableHandler, table: pd.DataFrame):
    return transformer[type].decode(table)


def create_transformers_pipeline(
    view: View, transformers: dict[str, Transformer], encoders: dict[str, Encoder]
):
    pipe = pipeline(
        [
            node(
                func=gen_closure(
                    _fit_table,
                    table,
                    transformers,
                    encoders,
                    _fn=f"fit_transformer_to_{table}",
                ),
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

    outputs = [
        D(
            "transformers",
            f"{view}.trn.{table}",
            ["views", "transformer", view, table],
            type="pkl",
        )
        for table in view.tables
    ] + [
        D(
            "transformers",
            f"{view}.trn.ids_{table}",
            ["views", "ids", view, table],
        )
        for table in view.tables
    ]

    return PipelineMeta(pipe, outputs)


def create_transform_pipeline(
    view: View, split: str, types: list[str], only_encode: bool = False
):
    table_nodes = []
    outputs = []

    for t in view.tables:
        if not only_encode:
            table_nodes += [
                node(
                    func=gen_closure(_transform_table, _fn=f"transform_{t}"),
                    inputs={
                        "transformer": f"trn_{t}",
                        **{t: t for t in view.tables},
                    },
                    outputs=[f"bst_{t}", f"ids_{t}"],
                ),
            ]
            outputs.append(
                D(
                    "split_transformed",
                    f"{view}.{split}.bst_{t}",
                    ["views", "bst", f"{view}.{split}", t],
                )
            )
            outputs.append(
                D(
                    "split_transformed",
                    f"{view}.{split}.ids_{t}",
                    ["views", "ids", f"{view}.{split}", t],
                )
            )

        for type in types:
            table_nodes += [
                node(
                    func=gen_closure(_encode_table, type, _fn=f"encode_{t}"),
                    inputs={
                        "transformer": f"trn_{t}",
                        "table": f"bst_{t}",
                    },
                    outputs=f"{type}_{t}",
                )
            ]
            outputs.append(
                D(
                    "split_encoded",
                    f"{view}.{split}.{type}_{t}",
                    ["views", type, f"{view}.{split}", t],
                )
            )

    inputs = {f"trn_{t}": f"{view}.trn.{t}" for t in view.tables}

    pipe = modular_pipeline(
        pipe=pipeline(table_nodes),
        namespace=f"{view}.{split}",
        inputs=inputs,
    )

    return PipelineMeta(pipe, outputs)


def create_reverse_pipeline(view: View, alg: str, type: str):
    decode_nodes = []
    outputs = []
    for t in view.tables:
        decode_nodes += [
            node(
                func=gen_closure(_decode_table, type, _fn=f"decode_{t}"),
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

        outputs.append(
            D(
                "synth_decoded",
                f"{view}.{alg}.bst_{t}",
                ["synth", "bst", f"{view}.{alg}", t],
                versioned=True,
            )
        )
        outputs.append(
            D(
                "synth_reversed",
                f"{view}.{alg}.{t}",
                ["synth", "dec", f"{view}.{alg}", t],
                versioned=True,
            )
        )

    pipe = modular_pipeline(
        pipe=pipeline(decode_nodes),
        namespace=f"{view}.{alg}",
        inputs={f"trn_{t}": f"{view}.trn.{t}" for t in view.tables},
    )

    return PipelineMeta(pipe, outputs)

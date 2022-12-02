from __future__ import annotations

from typing import TYPE_CHECKING

from kedro.pipeline import node, Pipeline as pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from .meta import DatasetMeta as D
from .meta import PipelineMeta, TAGS_TRANSFORM, TAGS_REVERSE, TAGS_RETRANSFORM
from ...utils import to_chunked, LazyFrame, LazyChunk, LazyDataset
from ...utils.progress import process

if TYPE_CHECKING:
    import pandas as pd
    from ...metadata import Metadata
    from ...table import TransformHolder
    from ...transform import TransformerFactory
    from ...encode import EncoderFactory
    from ...view import View

from .utils import gen_closure


def _fit_table_internal(
    name: str,
    transformers: dict[str, TransformerFactory],
    encoders: dict[str, EncoderFactory],
    meta: Metadata,
    **tables: LazyFrame,
):
    from ...table import TransformHolder

    t = TransformHolder(meta, name, transformers, encoders)
    for partitions in LazyDataset.zip(tables).values():
        t.fit_transform({name: partition() for name, partition in partitions.items()})
    return t

def _fit_table(
    name: str,
    transformers: dict[str, TransformerFactory],
    encoders: dict[str, EncoderFactory],
    meta: Metadata,
    **tables: LazyFrame):
    return process(_fit_table_internal, name, transformers, encoders, meta, **tables)

@to_chunked
def _transform_table(
    transformer: TransformHolder,
    **tables: LazyChunk,
):
    loaded = {name: table() for name, table in tables.items()}
    ids = transformer.find_ids(loaded)
    return transformer.transform(loaded, ids), ids


@to_chunked
def _base_reverse_table(
    transformer: TransformHolder,
    ids: LazyChunk,
    table: LazyChunk,
    **parents: LazyChunk,
):
    return transformer.reverse(
        table(), ids(), {name: parent() for name, parent in parents.items()}
    )


@to_chunked
def _encode_table(type: str, transformer: TransformHolder, table: LazyChunk):
    return transformer[type].encode(table())


@to_chunked
def _decode_table(type: str, transformer: TransformHolder, table: LazyChunk):
    return transformer[type].decode(table())


def create_transformer_pipeline(
    view: View,
    transformers: dict[str, TransformerFactory],
    encoders: dict[str, EncoderFactory],
    split: str,
):
    nodes = [
        node(
            func=gen_closure(
                _fit_table,
                t,
                transformers,
                encoders,
                _fn=f"fit_transformer_to_{t}",
            ),
            inputs={
                "meta": f"{view}.metadata",
                **{t: f"{view}.{split}.{t}" for t in view.tables},
            },
            outputs=f"{view}.trn.{t}",
            namespace=f"{view}.trn",
        )
        for t in view.tables
    ]

    return PipelineMeta(
        pipeline(nodes, tags=TAGS_TRANSFORM),
        [
            D("transformers", f"{view}.trn.{t}", ["views", "trn", view, t], type="pkl")
            for t in view.tables
        ],
    )


def create_transform_pipeline(
    view: View,
    split: str,
    types: list[str],
    retransform: bool = False,
):
    table_nodes = []
    outputs = []

    for t in view.tables:
        if not retransform:
            table_nodes += [
                node(
                    func=gen_closure(_transform_table, _fn=f"transform_{t}"),
                    inputs={
                        "transformer": f"{view}.trn.{t}",
                        **{t: f"{view}.{split}.{t}" for t in view.tables},
                    },
                    outputs=[f"{view}.{split}.bst_{t}", f"{view}.{split}.ids_{t}"],
                    namespace=f"{view}.{split}",
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
            if type in ("bst", "raw"):
                continue

            table_nodes += [
                node(
                    func=gen_closure(_encode_table, type, _fn=f"encode_{t}"),
                    inputs={
                        "transformer": f"{view}.trn.{t}",
                        "table": f"{view}.{split}.bst_{t}",
                    },
                    outputs=f"{view}.{split}.{type}_{t}",
                    namespace=f"{view}.{split}",
                )
            ]
            outputs.append(
                D(
                    # FIXME: Pass proper layer properly, don't infer
                    "synth_reencoded" if retransform else "split_encoded",
                    f"{view}.{split}.{type}_{t}",
                    ["views", type, f"{view}.{split}", t],
                )
            )

    if not table_nodes:
        return PipelineMeta(pipeline([]), outputs)

    return PipelineMeta(
        pipeline(table_nodes, tags=TAGS_RETRANSFORM if retransform else TAGS_TRANSFORM),
        outputs,
    )


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
        tags=TAGS_REVERSE,
    )

    return PipelineMeta(pipe, outputs)

from __future__ import annotations

from typing import TYPE_CHECKING

from kedro.pipeline import Pipeline as pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...utils import LazyChunk, LazyDataset, LazyFrame, to_chunked
from ...utils.progress import piter, process
from .meta import TAGS_RETRANSFORM, TAGS_REVERSE, TAGS_TRANSFORM
from .meta import DatasetMeta as D
from .meta import PipelineMeta, node

if TYPE_CHECKING:
    import pandas as pd
    from ...metadata import Metadata
    from ...table import TransformHolder
    from ...transform import TransformerFactory
    from ...encode import EncoderFactory
    from ...view import View


def _fit_table_internal(
    name: str,
    transformers: dict[str, TransformerFactory],
    encoders: dict[str, EncoderFactory],
    meta: Metadata,
    tables: dict[str, LazyFrame]
):
    from ...table import TransformHolder

    t = TransformHolder(meta, name, transformers, encoders)
    # for partitions in piter(
    #     LazyDataset.zip_values(**tables), desc="Fitting transformers (per chunk)"
    # ):
    t.fit_transform({name: table.sample for name, table in tables.items()})
    return t


def _fit_table(
    name: str,
    transformers: dict[str, TransformerFactory],
    encoders: dict[str, EncoderFactory],
    meta: Metadata,
    tables: dict[str, LazyFrame],
):
    return process(_fit_table_internal, name, transformers, encoders, meta, tables)


@to_chunked
def _get_ids(transformer: TransformHolder, **tables: LazyChunk):
    return transformer.find_ids({name: table() for name, table in tables.items()})


@to_chunked
def _transform_table(
    transformer: TransformHolder,
    ids: LazyChunk,
    **tables: LazyChunk,
):
    return transformer.transform(tables, ids)


@to_chunked
def _base_reverse_table(
    transformer: TransformHolder,
    ids: LazyChunk,
    table: LazyChunk,
    **parents: LazyChunk,
):
    return transformer.reverse(table, ids, parents)


@to_chunked
def _encode_table(type: str, transformer: TransformHolder, table: LazyChunk):
    return transformer[type].encode(table)


@to_chunked
def _decode_table(type: str, transformer: TransformHolder, table: LazyChunk):
    return transformer[type].decode(table)


def create_transformer_pipeline(
    view: View,
    transformers: dict[str, TransformerFactory],
    encoders: dict[str, EncoderFactory],
    split: str,
):
    nodes = [
        node(
            name=f"fit_transformer_to_{t}",
            func=_fit_table,
            args=[t, transformers, encoders],
            inputs={
                "meta": f"{view}.metadata",
                "tables": {t: f"{view}.{split}.{t}" for t in view.tables}
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
                    func=_get_ids,
                    name=f"get_ids_{t}",
                    inputs={
                        "transformer": f"{view}.trn.{t}",
                        **{t: f"{view}.{split}.{t}" for t in view.tables},
                    },
                    outputs=f"{view}.{split}.ids_{t}",
                    namespace=f"{view}.{split}",
                ),
                node(
                    func=_transform_table,
                    name=f"transform_{t}",
                    inputs={
                        "transformer": f"{view}.trn.{t}",
                        "ids": f"{view}.{split}.ids_{t}",
                        **{t: f"{view}.{split}.{t}" for t in view.tables},
                    },
                    outputs=f"{view}.{split}.bst_{t}",
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
                    func=_encode_table,
                    name=f"encode_{t}_{type}",
                    args=[type],
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
                    ["synth" if retransform else "views", type, f"{view}.{split}", t],
                    versioned=retransform
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
                func=_decode_table,
                args=[type],
                name=f"decode_{t}",
                inputs={
                    "transformer": f"trn_{t}",
                    "table": f"enc_{t}",
                },
                outputs=f"bst_{t}",
            ),
            node(
                func=_base_reverse_table,
                name=f"reverse_{t}",
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

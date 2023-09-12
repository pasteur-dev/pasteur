from __future__ import annotations


import pandas as pd
from kedro.pipeline import Pipeline as pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...encode import AttributeEncoderFactory, EncoderFactory, Encoder
from ...metadata import Metadata
from ...module import Module, get_module_dict
from ...table import AttributeEncoderHolder, TableTransformer
from ...utils import LazyFrame
from ...view import View
from .meta import TAGS_RETRANSFORM, TAGS_REVERSE, TAGS_TRANSFORM
from .meta import DatasetMeta as D
from .meta import PipelineMeta, node


def _fit_transformers(
    name: str,
    modules: list[Module],
    meta: Metadata,
    tables: dict[str, LazyFrame],
):
    from ...table import TableTransformer

    t = TableTransformer(meta, name, modules)
    t.fit(tables)
    return t


def _transform_table(
    transformer: TableTransformer,
    tables: dict[str, LazyFrame],
):
    return transformer.transform(tables)


def _reverse_table(
    transformer: TableTransformer,
    table: LazyFrame,
    ctx: dict[str, LazyFrame],
    ids: LazyFrame,
    parents: dict[str, LazyFrame],
):
    return transformer.reverse(table, ctx, ids, parents)


def _fit_encoder(
    enc: str,
    modules: list[Module],
    trns: dict[str, TableTransformer],
    tables: dict[str, LazyFrame],
    ctx: dict[str, dict[str, LazyFrame]],
    ids: dict[str, LazyFrame],
):
    # Get encoder factories
    attr_encs = get_module_dict(AttributeEncoderFactory, modules)
    encs = get_module_dict(EncoderFactory, modules)

    assert (
        enc not in attr_encs or enc not in encs
    ), f"Encoding '{enc}' is provided as both an Attribute Encoder and Encoder. Choose one."
    assert enc in attr_encs or enc in encs, f"Encoder for encoding '{enc}' not found."

    # Get attrs
    attrs = {}
    ctx_attrs = {}
    for name, trn in trns.items():
        table_attrs, table_ctx_attrs = trn.get_attributes()
        attrs[name] = table_attrs
        ctx_attrs[name] = table_ctx_attrs

    # Create and fit
    if enc in encs:
        e = encs[enc].build()
    else:
        e = AttributeEncoderHolder(attr_encs[enc])

    e.fit(attrs, tables, ctx_attrs, ctx, ids)
    return e


def _encode_view(
    encoder: Encoder,
    tables: dict[str, LazyFrame],
    ctx: dict[str, dict[str, LazyFrame]],
    ids: dict[str, LazyFrame],
):
    return encoder.encode(tables, ctx, ids)


def _decode_view(encoder: Encoder, data: dict[str, LazyFrame]):
    return encoder.decode(data)


def create_fit_pipeline(
    view: View,
    encs: list[str],
    modules: list[Module],
    split: str,
):
    trn_fit_nodes = [
        node(
            name=f"fit_transformers_to_{t}",
            func=_fit_transformers,
            args=[t, modules],
            inputs={
                "meta": f"{view}.metadata",
                "tables": {t: f"{view}.{split}.{t}" for t in view.tables},
            },
            outputs=f"{view}.trn.{t}",
            namespace=f"{view}.trn",
        )
        for t in view.tables
    ]

    enc_fit_nodes = [
        node(
            name=f"fit_{enc}_encoder",
            func=_fit_encoder,
            args=[enc, modules],
            inputs={
                "trns": {t: f"{view}.trn.{t}" for t in view.tables},
                "tables": {t: f"{view}.{split}.bst_{t}" for t in view.tables},
                "ctx": {t: f"{view}.{split}.ctx_{t}" for t in view.tables},
                "ids": {t: f"{view}.{split}.ids_{t}" for t in view.tables},
            },
            outputs=f"{view}.enc.{enc}",
            namespace=f"{view}.enc",
        )
        for enc in encs
        if enc not in ("raw", "bst")
    ]

    return PipelineMeta(
        pipeline(trn_fit_nodes + enc_fit_nodes, tags=TAGS_TRANSFORM),
        [
            D("transformers", f"{view}.trn.{t}", ["view", view, "trn", t], type="pkl")
            for t in view.tables
        ]
        + [
            D("encoders", f"{view}.enc.{enc}", ["view", view, "enc", enc], type="pkl")
            for enc in encs
            if enc not in ("raw", "bst")
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
                    func=_transform_table,
                    name=f"transform_{t}_for_{split}",
                    inputs={
                        "transformer": f"{view}.trn.{t}",
                        "tables": {t: f"{view}.{split}.{t}" for t in view.tables},
                    },
                    outputs=[
                        f"{view}.{split}.bst_{t}",
                        f"{view}.{split}.ctx_{t}",
                        f"{view}.{split}.ids_{t}",
                    ],
                    namespace=f"{view}.{split}",
                ),
            ]

            layer = "view_transformed" if split == "view" else "split_transformed"
            outputs.append(
                D(
                    layer,
                    f"{view}.{split}.ctx_{t}",
                    ["view", view, split, "ctx", t],
                    type="multi",
                )
            )
            outputs.append(
                D(
                    layer,
                    f"{view}.{split}.bst_{t}",
                    ["view", view, split, "bst", t],
                )
            )
            outputs.append(
                D(
                    layer,
                    f"{view}.{split}.ids_{t}",
                    ["view", view, split, "ids", t],
                )
            )

    for enc in types:
        if enc in ("bst", "raw") or split == "view":
            continue

        table_nodes += [
            node(
                func=_encode_view,
                name=f"encode_{enc}",
                inputs={
                    "encoder": f"{view}.enc.{enc}",
                    "tables": {t: f"{view}.{split}.bst_{t}" for t in view.tables},
                    "ctx": {t: f"{view}.{split}.ctx_{t}" for t in view.tables},
                    "ids": {t: f"{view}.{split}.ids_{t}" for t in view.tables},
                },
                outputs=f"{view}.{split}.{enc}",
                namespace=f"{view}.{split}",
            )
        ]
        outputs.append(
            D(
                # FIXME: Pass proper layer properly, don't infer
                "synth_reencoded" if retransform else "split_encoded",
                f"{view}.{split}.{enc}",
                ["synth" if retransform else "view", view, split, enc],
                versioned=retransform,
                type="multi",
            )
        )

    if not table_nodes:
        return PipelineMeta(pipeline([]), outputs)

    return PipelineMeta(
        pipeline(table_nodes, tags=TAGS_RETRANSFORM if retransform else TAGS_TRANSFORM),
        outputs,
    )


def create_reverse_pipeline(view: View, alg: str, enc: str):
    decode_nodes = [
        node(
            func=_decode_view,
            name=f"decode_synthetic_data",
            inputs={
                "encoder": f"{view}.enc.{enc}",
                "data": f"{view}.{alg}.enc",
            },
            outputs=[
                {t: f"{view}.{alg}.bst_{t}" for t in view.tables},
                {t: f"{view}.{alg}.ctx_{t}" for t in view.tables},
                {t: f"{view}.{alg}.ids_{t}" for t in view.tables},
            ],
            namespace=f"{view}.{alg}",
        ),
    ]
    outputs = []
    for t in view.tables:
        decode_nodes += [
            node(
                func=_reverse_table,
                name=f"reverse_{t}",
                inputs={
                    "transformer": f"{view}.trn.{t}",
                    "table": f"{view}.{alg}.bst_{t}",
                    "ctx": f"{view}.{alg}.ctx_{t}",
                    "ids": f"{view}.{alg}.ids_{t}",
                    "parents": {
                        req: f"{view}.{alg}.{req}" for req in view.trn_deps.get(t, [])
                    },
                },
                outputs=f"{view}.{alg}.{t}",
                namespace=f"{view}.{alg}",
            ),
        ]

        outputs.extend(
            [
                D(
                    "synth_decoded",
                    f"{view}.{alg}.bst_{t}",
                    ["synth", view, alg, "bst", t],
                    versioned=True,
                ),
                D(
                    "synth_decoded",
                    f"{view}.{alg}.ids_{t}",
                    ["synth", view, alg, "ids", t],
                    versioned=True,
                ),
                D(
                    "synth_decoded",
                    f"{view}.{alg}.ctx_{t}",
                    ["synth", view, alg, "ctx", t],
                    versioned=True,
                    type="multi",
                ),
                D(
                    "synth_reversed",
                    f"{view}.{alg}.{t}",
                    ["synth", view, alg, "tables", t],
                    versioned=True,
                ),
            ]
        )

    pipe = modular_pipeline(
        pipe=pipeline(decode_nodes),
        tags=TAGS_REVERSE,
    )

    return PipelineMeta(pipe, outputs)

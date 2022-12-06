from __future__ import annotations

from typing import TYPE_CHECKING

from kedro.pipeline import Pipeline as pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...synth import synth_fit, synth_sample
from .meta import DatasetMeta as D
from .meta import PipelineMeta, node, TAGS_SYNTH, TAG_GPU
from .utils import gen_closure

if TYPE_CHECKING:
    from ...synth import SynthFactory
    from ...view import View


def create_synth_pipeline(
    view: View,
    split: str,
    fr: SynthFactory,
):
    alg = fr.name
    type = fr.type
    tables = view.tables

    tags: list[str] = list(TAGS_SYNTH)
    if fr.gpu:
        tags.append(TAG_GPU)

    pipe = pipeline(
        [
            node(
                func=gen_closure(synth_fit, fr),
                inputs={
                    "metadata": f"{view}.metadata",
                    "trns": {t: f"{view}.trn.{t}" for t in tables},
                    "ids": {t: f"{view}.{split}.ids_{t}" for t in tables},
                    "tables": {t: f"{view}.{split}.{type}_{t}" for t in tables},
                },
                outputs=f"{view}.{alg}.model",
                namespace=f"{view}.{alg}",
                tags=tags,
            ),
            node(
                func=synth_sample,
                inputs=f"{view}.{alg}.model",
                outputs={
                    "ids": {t: f"{view}.{alg}.ids_{t}" for t in tables},
                    "tables": {t: f"{view}.{alg}.enc_{t}" for t in tables},
                },
                namespace=f"{view}.{alg}",
                tags=tags,
            ),
        ]
    )

    outputs = [
        D(
            "synth_models",
            f"{view}.{alg}.model",
            ["synth", "models", f"{view}.{alg}"],
            versioned=True,
            type="pkl",
        ),
        *[
            D(
                "synth_output",
                f"{view}.{alg}.enc_{t}",
                ["synth", "enc", f"{view}.{alg}", t],
                versioned=True,
            )
            for t in view.tables
        ],
        *[
            D(
                "synth_output",
                f"{view}.{alg}.ids_{t}",
                ["synth", "ids", f"{view}.{alg}", t],
                versioned=True,
            )
            for t in view.tables
        ],
    ]

    return PipelineMeta(pipe, outputs)

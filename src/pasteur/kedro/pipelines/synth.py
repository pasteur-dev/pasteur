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
    tags: list[str] = list(TAGS_SYNTH)

    pipe = pipeline(
        [
            node(
                func=synth_fit,
                name=f"fitting_{fr.name}",
                args=[fr],
                inputs={
                    "encoder": f"{view}.enc.{fr.type}",
                    "data": f"{view}.{split}.{fr.type}",
                },
                namespace=f"{view}.{fr.name}",
                outputs=f"{view}.{fr.name}.model",
                tags=tags,
            ),
            node(
                func=synth_sample,
                inputs=f"{view}.{fr.name}.model",
                outputs=f"{view}.{fr.name}.enc",
                namespace=f"{view}.{fr.name}",
                tags=tags,
            ),
        ]
    )

    outputs = [
        D(
            "synth_models",
            f"{view}.{fr.name}.model",
            ["synth", "models", f"{view}.{fr.name}"],
            versioned=True,
            type="pkl",
        ),
        D(
            "synth_output",
            f"{view}.{fr.name}.enc",
            ["synth", "enc", f"{view}.{fr.name}"],
            versioned=True,
            type="multi",
        ),
    ]

    return PipelineMeta(pipe, outputs)

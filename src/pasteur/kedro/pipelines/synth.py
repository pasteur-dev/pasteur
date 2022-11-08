from __future__ import annotations

from typing import TYPE_CHECKING

from kedro.pipeline import node, Pipeline as pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...synth import synth_fit, synth_sample
from .utils import gen_closure

if TYPE_CHECKING:
    from ...view import View
    from ...synth import Synth


def create_synth_pipeline(
    view: View,
    split: str,
    cls: type[Synth],
):
    alg = cls.name
    type = cls.type
    tables = view.tables

    tags = []
    if cls.gpu:
        tags.append("gpu")
    if cls.parallel:
        tags.append("parallel")

    synth_pipe = pipeline(
        [
            node(
                func=gen_closure(synth_fit, cls),
                inputs={
                    "metadata": "metadata",
                    **{f"trn_{t}": f"trn_{t}" for t in tables},
                    **{f"ids_{t}": f"in_ids_{t}" for t in tables},
                    **{f"enc_{t}": f"in_enc_{t}" for t in tables},
                },
                outputs="model",
                tags=tags,
            ),
            node(
                func=synth_sample,
                inputs="model",
                outputs={
                    **{f"ids_{t}": f"ids_{t}" for t in tables},
                    **{f"enc_{t}": f"enc_{t}" for t in tables},
                },
                tags=tags,
            ),
        ]
    )

    return modular_pipeline(
        pipe=synth_pipe,
        namespace=f"{view}.{alg}",
        inputs={
            "metadata": f"{view}.metadata",
            **{f"in_enc_{t}": f"{view}.{split}.{type}_{t}" for t in tables},
            **{f"in_ids_{t}": f"{view}.{split}.ids_{t}" for t in tables},
            **{f"trn_{t}": f"{view}.trn.{t}" for t in tables},
        },
    )

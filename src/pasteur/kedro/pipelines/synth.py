from __future__ import annotations

from typing import TYPE_CHECKING

from kedro.pipeline import Pipeline as pipeline
from kedro.pipeline import node
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...synth import synth_fit, synth_sample
from .module import DatasetMeta as D
from .module import PipelineMeta
from .utils import gen_closure

if TYPE_CHECKING:
    from ...synth import Synth
    from ...view import View


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

    pipe = modular_pipeline(
        pipe=synth_pipe,
        namespace=f"{view}.{alg}",
        inputs={
            "metadata": f"{view}.metadata",
            **{f"in_enc_{t}": f"{view}.{split}.{type}_{t}" for t in tables},
            **{f"in_ids_{t}": f"{view}.{split}.ids_{t}" for t in tables},
            **{f"trn_{t}": f"{view}.trn.{t}" for t in tables},
        },
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
                ["synth", "enc", f"{view}.{alg}.{t}"],
            )
            for t in view.tables
        ],
        *[
            D(
                "synth_output",
                f"{view}.{alg}.ids_{t}",
                ["synth", "enc", f"{view}.{alg}.{t}"],
            )
            for t in view.tables
        ],
    ]

    return PipelineMeta(pipe, outputs)

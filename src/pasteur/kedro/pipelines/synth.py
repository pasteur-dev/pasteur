from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from ...synth import synth_fit, synth_sample
from ...views import View
from .utils import gen_closure


def create_synth_pipeline(
    view: View,
    split: str,
    trn_split: str,
    cls: type,
):
    alg = cls.name
    type = cls.type
    tables = view.tables

    synth_pipe = pipeline(
        [
            node(
                func=gen_closure(synth_fit, cls),
                inputs={
                    **{f"trn_{t}": f"trn_{t}" for t in tables},
                    **{f"ids_{t}": f"in_ids_{t}" for t in tables},
                    **{f"enc_{t}": f"in_enc_{t}" for t in tables},
                },
                outputs="model",
            ),
            node(
                func=synth_sample,
                inputs="model",
                outputs={
                    **{f"ids_{t}": f"ids_{t}" for t in tables},
                    **{f"enc_{t}": f"enc_{t}" for t in tables},
                },
            ),
        ]
    )

    return modular_pipeline(
        pipe=synth_pipe,
        namespace=f"{view}.{alg}",
        inputs={
            **{f"in_enc_{t}": f"{view}.{split}.{type}_{t}" for t in tables},
            **{f"in_ids_{t}": f"{view}.{split}.ids_{t}" for t in tables},
            **{f"trn_{t}": f"{view}.{trn_split}.trn_{t}" for t in tables},
        },
    )

from kedro.pipeline import Pipeline
from typing import NamedTuple, Literal, Any


class DatasetMeta(NamedTuple):
    layer: str
    name: str
    path: list[Any]  # todo: fix any memory leaks that occur with this
    versioned: bool = False
    type: Literal["pkl", "ppq", "pq", "mem"] = "pq"

    @property
    def str_path(self) -> tuple[str]:
        return tuple(map(str, self.path))


class PipelineMeta(NamedTuple):
    """Retain pipeline behavior with addition but enable storing dataset metadata.

    Raw sources should be defined by the dataset. Everything else should be
    defined by the pipeline."""

    pipeline: Pipeline
    outputs: list[DatasetMeta]

    def __add__(self, other):
        if isinstance(other, Pipeline):
            return PipelineMeta(self.pipeline + other, self.outputs)

        assert isinstance(other, PipelineMeta)
        return PipelineMeta(
            self.pipeline + other.pipeline, [*self.outputs, *other.outputs]
        )

    def __radd__(self, other):
        return self.__add__(other)


# Tag each node in the pipeline based on its use
TAG_VIEW = "view"
TAG_DATASET = "dataset"
TAG_TRANSFORM = "transform"
TAG_SYNTH = "synth"
TAG_REVERSE = "reverse"
TAG_METRICS = "metrics"

# Process tags
TAG_GPU = "gpu"
TAG_PARALLEL = "parallel"

# Tag each node in the pipeline based on when it should run.
"""Nodes tagged with `changes_never` always produce the same result (unless their
source code/raw data changes). Example: dataset ingestion."""
TAG_CHANGES_NEVER = "changes_never"

"""Nodes tagged with 'changes_hyperparameter` are influenced by hyperparameters
so they should run every time hyperparameters change (example: view splits)."""
TAG_CHANGES_HYPERPARAMETER = "changes_hyperparameter"

"""Nodes tagged with `changes_per_algorithm` are influenced by or produce synthetic
data (such as metrics, synthesis, and reversal)."""
TAG_CHANGES_PER_ALGORITHM = "changes_per_algorithm"

TAGS_DATASET = (TAG_DATASET, TAG_PARALLEL, TAG_CHANGES_NEVER)
TAGS_VIEW = (TAG_VIEW, TAG_PARALLEL, TAG_CHANGES_NEVER)
TAGS_VIEW_SPLIT = (TAG_VIEW, TAG_CHANGES_HYPERPARAMETER)
TAGS_TRANSFORM = (TAG_TRANSFORM, TAG_CHANGES_HYPERPARAMETER)
TAGS_SYNTH = (TAG_SYNTH, TAG_CHANGES_PER_ALGORITHM)
TAGS_REVERSE = (TAG_REVERSE, TAG_CHANGES_PER_ALGORITHM)
TAGS_RETRANSFORM = (TAG_TRANSFORM, TAG_CHANGES_PER_ALGORITHM)
TAGS_METRICS_INGEST = (TAG_METRICS, TAG_CHANGES_HYPERPARAMETER)
TAGS_METRICS_LOG = (TAG_METRICS, TAG_CHANGES_PER_ALGORITHM)

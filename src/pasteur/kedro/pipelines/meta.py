from kedro.pipeline import Pipeline
from typing import NamedTuple, Literal, Any

class DatasetMeta(NamedTuple):
    layer: str
    name: str
    path: list[Any] # todo: fix any memory leaks that occur with this
    versioned: bool = False
    type: Literal["pkl", "pq"] = "pq"

    @property
    def str_path(self) -> tuple[str]:
        return tuple(map(str, self.path))


class PipelineMeta(NamedTuple):
    """ Retain pipeline behavior with addition but enable storing dataset metadata.
    
    Raw sources should be defined by the dataset. Everything else should be
    defined by the pipeline. """
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
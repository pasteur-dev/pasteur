from kedro.pipeline import Pipeline
from typing import TypeVar, NamedTuple, Literal, Any

A = TypeVar("A")


class DatasetMeta(NamedTuple):
    layer: str
    name: str
    path: list[Any] # todo: fix any memory leaks that occur with this
    versioned: bool = False
    type: Literal["pkl", "pq"] = "pq"


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


def get_module_dict(parent: type[A], modules: list[type]) -> dict[str, A]:
    """Filters the list `modules` for modules which extend"""
    out = {}
    for module in modules:
        if not issubclass(module, parent):
            continue

        assert hasattr(
            module, "name"
        ), f"Module class attr {module.__name__}.name doesn't exist."
        assert (
            isinstance(module.name, str) and module.name
        ), f"{module.__name__} is not str or is empty."
        assert (
            module.name not in out
        ), "There are multiple modules of the same type with the same name."
        out[module.name] = module
    return out

def instantiate_dict(d: dict[str, type]):
    return {k: v() for k, v in d.items()}
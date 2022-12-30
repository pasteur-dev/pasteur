from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Literal, NamedTuple

from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node


class DatasetMeta(NamedTuple):
    layer: str
    name: str
    path: list[Any]  # todo: fix any memory leaks that occur with this
    versioned: bool = False
    type: Literal["pkl", "pq", "mem"] = "pq"

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

        assert isinstance(other, PipelineMeta), f"Other pipeline is type: {type(other)}"
        return PipelineMeta(
            self.pipeline + other.pipeline, [*self.outputs, *other.outputs]
        )

    def __radd__(self, other):
        return self.__add__(other)


NestedInputs = str | list["NestedInputs"] | dict[str, "NestedInputs"]
NestedOutputs = Any | list["NestedOutputs"] | dict[str, "NestedOutputs"]


def _flatten_inputs(inputs: NestedInputs) -> str | list[str]:
    if isinstance(inputs, str):
        return inputs

    out = []
    for nested in inputs.values() if isinstance(inputs, dict) else inputs:
        data = _flatten_inputs(nested)
        if isinstance(data, str):
            out.append(data)
        else:
            out.extend(data)

    return out


def _flatten_outputs(
    nested: NestedInputs, outputs: NestedOutputs, run: bool = False
) -> dict[str, Any]:
    # Allow this function to be used in a partial
    if run and callable(outputs):
        outputs = outputs()

    if isinstance(nested, str):
        return {nested: outputs}

    out = {}
    if isinstance(nested, dict):
        assert isinstance(outputs, dict)
        for idx, vals in nested.items():
            assert idx in outputs
            data = _flatten_outputs(vals, outputs[idx])
            out.update(data)
    else:
        assert isinstance(outputs, list) and isinstance(nested, list)
        assert len(outputs) == len(nested)
        for vals, outs in zip(nested, outputs):
            data = _flatten_outputs(vals, outs)
            out.update(data)

    return out


def _bind_inputs(inputs: NestedInputs, datasets: dict[str, Any]):
    if isinstance(inputs, str):
        return datasets[inputs]

    if isinstance(inputs, dict):
        out = {}
        for name, nested in inputs.items():
            out[name] = _bind_inputs(nested, datasets)
        return out

    assert isinstance(inputs, list)

    out = []
    for nested in inputs:
        out.append(_bind_inputs(nested, datasets))
    return out


def _rename_inputs(inputs: NestedInputs, new_names: list[str], idx: int = 0):

    if isinstance(inputs, str):
        return (new_names[idx] if isinstance(new_names, list) else new_names), idx + 1

    if isinstance(inputs, list):
        out = []
        for inp in inputs:
            new_out, idx = _rename_inputs(inp, new_names, idx)
            out.append(new_out)
        return out, idx

    if isinstance(inputs, dict):
        out = {}
        for name, inp in inputs.items():
            new_out, idx = _rename_inputs(inp, new_names, idx)
            out[name] = new_out
        return out, idx

    raise TypeError()


class ExtendedNode(Node):
    """Extended node is a modification of node that allows for nesting dictionaries
    in inputs and outputs and features a built-in closure.

    Example:
    ```
    inputs: {
        metadata: metadata,
        ids: {
            table1: table1, table2: table2
        },
        tables: {
            table1: table1, table2: table2
        },
    }
    ```
    This way, multiple kedro datasets can be passed in, without having to tangle
    them with ids.

    Also, `ExtendedNode` acts as a partial when `args`, `kwargs` are provided."""

    def __init__(
        self,
        func: Callable,
        inputs: NestedInputs | None,
        outputs: NestedInputs | None,
        *,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        name: str | None = None,
        tags: str | Iterable[str] | None = None,
        confirms: str | List[str] | None = None,
        namespace: str | None = None,
    ):
        self._args = args or []
        self._kwargs = kwargs or {}

        self._nested_inputs = inputs
        flattened_inputs = None
        if inputs is not None:
            flattened_inputs = _flatten_inputs(inputs)
        self._nested_outputs = outputs
        flattened_outputs = None
        if outputs is not None:
            flattened_outputs = _flatten_inputs(outputs)

        super().__init__(
            func,
            flattened_inputs,
            flattened_outputs,
            name=name,  # type: ignore
            tags=tags,  # type: ignore
            confirms=confirms,  # type: ignore
            namespace=namespace,  # type: ignore
        )

        self._validate_exp_inputs()

    def run(
        self, inputs: Dict[str, Any] | None = None
    ) -> Dict[str, Any] | set[Callable[..., Dict[str, Any]]]:
        if self._nested_inputs is None or len(self._nested_inputs) == 0:
            out = self._func(*self._args, **self._kwargs)
        else:
            assert inputs, f"Inputs for node are None, but node expects inputs."

            data = _bind_inputs(self._nested_inputs, inputs)

            if isinstance(self._nested_inputs, str):  # One argument
                out = self._func(*self._args, data, **self._kwargs)
            elif isinstance(data, list):  # List
                out = self._func(*self._args, *data, **self._kwargs)
            else:  # Dictionary
                out = self._func(*self._args, **self._kwargs, **data)

        if not self._nested_outputs:
            return {}

        if isinstance(out, set):
            return {
                partial(_flatten_outputs, self._nested_outputs, o, run=True)
                for o in out
            }

        return _flatten_outputs(self._nested_outputs, out)

    def __str__(self):
        def _set_to_str(xset):
            return f"[{','.join(xset)}]"

        out_str = _set_to_str(self.outputs) if self._outputs else "None"
        in_str = _set_to_str(self.inputs) if self._inputs else "None"

        prefix = self._name + ": " if self._name else ""
        return prefix + f"{self._func_name}({in_str}) -> {out_str}"

    def _validate_inputs(self, func, inputs):
        ...

    def _validate_exp_inputs(self):
        # inspect does not support built-in Python functions written in C.
        # Thus we only validate func if it is not built-in.
        import inspect

        func = self._func

        if inspect.isbuiltin(func):
            return

        if self._nested_inputs:
            data = self._nested_inputs
            if isinstance(data, dict):
                args = self._args
                kwargs = {**self._kwargs, **data}
            elif isinstance(data, list):
                args = [*self._args, *data]
                kwargs = self._kwargs
            else:
                args = [*self._args, data]
                kwargs = self._kwargs
        else:
            args = self._args
            kwargs = self._kwargs

        try:
            inspect.signature(func, follow_wrapped=False).bind(*args, **kwargs)
        except Exception as exc:
            func_args = inspect.signature(func, follow_wrapped=False).parameters.keys()
            func_name = self._func_name

            raise TypeError(
                f"Inputs of '{func_name}' function expected {list(func_args)}, "
                f"but got (*{args},**{kwargs})"
            ) from exc

    def _copy(self, **overwrite_params):
        """
        Helper function to copy the node, replacing some values.
        """
        params = {
            "func": self._func,
            "inputs": self._nested_inputs,
            "outputs": self._nested_outputs,
            "args": self._args,
            "kwargs": self._kwargs,
            "name": self._name,
            "namespace": self._namespace,
            "tags": self._tags,
            "confirms": self._confirms,
        }
        overwrite_params = overwrite_params.copy()
        new_inputs = overwrite_params.pop("inputs", None)
        new_outputs = overwrite_params.pop("outputs", None)

        # FIXME: Botch to make Extended Nodes work with old pipeline
        if self._nested_inputs and new_inputs:
            params["inputs"] = _rename_inputs(self._nested_inputs, new_inputs)[0]
        else:
            assert not new_inputs
        if self._nested_outputs and new_outputs:
            params["outputs"] = _rename_inputs(self._nested_outputs, new_outputs)[0]
        else:
            assert not new_outputs

        params.update(overwrite_params)
        return type(self)(**params)


def node(
    func: Callable,
    inputs: NestedInputs | None,
    outputs: NestedInputs | None,
    *,
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    name: str | None = None,
    tags: str | Iterable[str] | None = None,
    confirms: str | List[str] | None = None,
    namespace: str | None = None,
) -> Node:
    """
    Modified version of kedro node that tweaks the node name to work better with
    kedro viz adds the option of nested inputs, outputs and features closure support.
    """
    return ExtendedNode(
        func,
        inputs,
        outputs,
        args=args,
        kwargs=kwargs,
        name=name,
        tags=tags,
        confirms=confirms,
        namespace=namespace,
    )

# Tag each node in the pipeline based on its use
TAG_VIEW = "view"
TAG_DATASET = "dataset"
TAG_TRANSFORM = "transform"
TAG_SYNTH = "synth"
TAG_REVERSE = "reverse"
TAG_METRICS = "metrics"

# Process tags
TAG_GPU = "gpu"

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

TAGS_DATASET = (TAG_DATASET, TAG_CHANGES_NEVER)
TAGS_VIEW = (TAG_VIEW, TAG_CHANGES_NEVER)
TAGS_VIEW_SPLIT = (TAG_VIEW, TAG_CHANGES_HYPERPARAMETER)
TAGS_VIEW_META = (TAG_VIEW, TAG_CHANGES_PER_ALGORITHM)
TAGS_TRANSFORM = (TAG_TRANSFORM, TAG_CHANGES_HYPERPARAMETER)
TAGS_SYNTH = (TAG_SYNTH, TAG_CHANGES_PER_ALGORITHM)
TAGS_REVERSE = (TAG_REVERSE, TAG_CHANGES_PER_ALGORITHM)
TAGS_RETRANSFORM = (TAG_TRANSFORM, TAG_CHANGES_PER_ALGORITHM)
TAGS_METRICS_INGEST = (TAG_METRICS, TAG_CHANGES_HYPERPARAMETER)
TAGS_METRICS_LOG = (TAG_METRICS, TAG_CHANGES_PER_ALGORITHM)

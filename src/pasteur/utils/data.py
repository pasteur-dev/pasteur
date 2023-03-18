from __future__ import annotations

from functools import partial, update_wrapper
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    TypeVar,
    Generic,
    overload,
    Mapping,
    ParamSpec,
)
import logging

if TYPE_CHECKING:
    import pandas as pd

A = TypeVar("A")
B = TypeVar("B")
P = ParamSpec("P")

logger = logging.getLogger(__name__)


class LazyPartition(Generic[A]):
    def __init__(
        self,
        fun: Callable[..., A],
        shape_fun: Callable[..., tuple[int, ...]] | None,
        /,
        *args,
        **kwargs,
    ):
        self.fun = fun
        self.shape_fun = shape_fun
        self.args = args
        self.kwargs = kwargs

    def __call__(
        self, columns: list[str] | None = None, chunksize: int | None = None
    ) -> A:
        new_kw = {}
        if columns is not None:
            new_kw['columns'] = columns
        if chunksize is not None:
            new_kw['chunksize'] = chunksize

        try:
            return self.fun(
                *self.args, **self.kwargs, **new_kw
            )
        except TypeError:
            return self.fun(*self.args, **self.kwargs)

    @property
    def partitioned(self):
        return False

    @property
    def shape(self):
        if self.shape_fun is not None:
            return self.shape_fun(*self.args, **self.kwargs)
        return self().shape  # type: ignore


LazyChunk = LazyPartition["pd.DataFrame"]


class LazyDataset(Generic[A], LazyPartition[A]):
    def __init__(
        self,
        merged_load: LazyPartition[A] | None,
        partitions: dict[str, LazyPartition[A]] | None = None,
    ) -> None:
        self._partitions = partitions
        self.merged_load = merged_load

    def __call__(
        self, columns: list[str] | None = None, chunksize: int | None = None
    ) -> Any:
        assert self.merged_load, f"Merged loading is not implemented for this dataset."
        if self.partitioned:
            logger.warning(
                f"Loading partitioned dataset as a whole, this may cause memory issues."
            )
        return self.merged_load(columns=columns, chunksize=chunksize)

    def __getitem__(self, pid: str) -> LazyPartition[A]:
        assert self._partitions, "No partitions found."
        return self._partitions[pid]

    def __iter__(self):
        if self._partitions:
            return iter(self._partitions)
        return iter([self.merged_load])

    @property
    def shape(self):
        if self.merged_load is not None:
            return self.merged_load.shape

        partitions = list(self.values())
        if len(partitions) == 0:
            return tuple()
        if len(partitions) == 1:
            return partitions[0].shape

        shape = list(partitions[0].shape)
        for part in partitions[1:]:
            shape[0] += part.shape[0]

        return tuple(shape)

    @property
    def partitioned(self):
        return self._partitions is not None and len(self._partitions) > 1

    @property
    def sample(self):
        if self._partitions:
            return next(iter(self._partitions.values()))
        assert self.merged_load is not None, f"LazyDataset is empty."
        return self.merged_load

    def keys(self):
        assert self._partitions, f"Dataset not partitioned, check `.partitioned` first."
        return self._partitions.keys()

    def values(self):
        if not self._partitions:
            assert self.merged_load
            return [self.merged_load]
        return self._partitions.values()

    def items(self):
        assert self._partitions, f"Dataset not partitioned, check `.partitioned` first."
        return self._partitions.items()

    @staticmethod
    def are_partitioned(*positional: LazyDataset, **keyword: LazyDataset):
        """Returns whether the provided datasets are partitioned. If they are,
        checks they have the same partitions."""

        partitions = [*positional, *list(keyword.values())]
        # Check keys are correct
        keys = None
        for partition in partitions:
            if not partition.partitioned:
                continue

            if keys == None:
                keys = list(partition.keys())
            else:
                assert set(keys) == set(partition.keys())

        return keys is not None

    @staticmethod
    @overload
    def zip(
        datasets: Mapping[str, LazyDataset[B]], /
    ) -> dict[str, dict[str, LazyPartition[B]]]:
        ...

    @staticmethod
    @overload
    def zip(*positional: LazyDataset[B]) -> dict[str, list[LazyPartition[B]]]:
        ...

    @staticmethod
    @overload
    def zip(**keyword: LazyDataset[B]) -> dict[str, dict[str, LazyPartition[B]]]:
        ...

    @staticmethod
    @overload
    def zip(
        *positional: LazyDataset[B], **keyword: LazyDataset[B]
    ) -> dict[str, tuple[list[LazyPartition[B]], dict[str, LazyPartition[B]]]]:
        ...

    @staticmethod
    def zip(
        *positional, **keyword: LazyDataset[B]
    ) -> dict[str, dict[str, LazyPartition[B]]] | dict[
        str, list[LazyPartition[B]]
    ] | dict[str, tuple[list[LazyPartition[B]], dict[str, LazyPartition[B]]]]:
        """Aligns and returns a dictionary of partition ids to partitions.

        Partitions can be a list, if positional arguments were provided, or a dictionary
        if keyword arguments were provided.

        @warning: all partitioned sets should have the same keys."""

        if positional and keyword:
            partitions = [*positional, *list(keyword.values())]
        elif positional and not keyword:
            if isinstance(positional[0], dict):
                keyword = positional[0]
                partitions = keyword.values()
                positional = tuple()
            else:
                partitions = positional
        else:
            partitions = keyword.values()

        # Check keys are correct
        keys = None
        for partition in partitions:
            if not partition.partitioned:
                continue

            if keys == None:
                keys = list(partition.keys())
            else:
                assert set(keys) == set(partition.keys())

        # If no datasets are partitioned, return a single one with all the datasets.
        assert (
            keys
        ), f"None of the datasets are partitioned (or one is empty). Call `are_partitioned()` first."

        if positional and keyword:
            return {
                pid: (
                    [ds[pid] if ds.partitioned else ds for ds in positional],
                    {
                        name: ds[pid] if ds.partitioned else ds
                        for name, ds in keyword.items()
                    },
                )
                for pid in keys
            }
        elif positional:
            return {
                pid: [ds[pid] if ds.partitioned else ds for ds in positional]
                for pid in keys
            }

        return {
            pid: {
                name: ds[pid] if ds.partitioned else ds for name, ds in keyword.items()
            }
            for pid in keys
        }

    @staticmethod
    @overload
    def zip_values(*positional: LazyDataset[B]) -> list[list[LazyPartition[B]]]:
        ...

    @staticmethod
    @overload
    def zip_values(**keyword: LazyDataset[B]) -> list[dict[str, LazyPartition[B]]]:
        ...

    @staticmethod
    def zip_values(
        *positional: LazyDataset[B], **keyword: LazyDataset[B]
    ) -> list[list[LazyPartition[B]]] | list[dict[str, LazyPartition[B]]]:
        """Same as zip, but doesn't return partition names and works even if
        the datasets are not partitioned, by returning a single partition."""
        if not LazyDataset.are_partitioned(*positional, **keyword):
            if positional:
                return [list(positional)]
            else:
                return [keyword]  # type: ignore

        return list(LazyDataset.zip(*positional, **keyword).values())

    def separate(
        **datasets: LazyDataset[A],
    ) -> tuple[dict[str, LazyDataset[A]], dict[str, LazyDataset[A]]]:
        """Splits the datasets into partitioned and not partitioned and returns them.

        `non_partitioned, partitioned = separate_partitioned(datasets)`"""
        return {name: ds for name, ds in datasets.items() if not ds.partitioned}, {
            name: ds for name, ds in datasets.items() if ds.partitioned
        }

    def __len__(self):
        if self._partitions:
            return len(self._partitions)
        if self.merged_load:
            return 1
        return 0


LazyFrame = LazyDataset["pd.DataFrame"]


def get_relative_fn(fn: str):
    """Returns the directory of a file relative to the script calling this function."""
    import inspect
    import os

    script_fn = inspect.currentframe().f_back.f_globals["__file__"]  # type: ignore
    dirname = os.path.dirname(script_fn)
    return os.path.join(dirname, fn)


def list_unique(*args: list[A]) -> list[A]:
    return list(dict.fromkeys(chain(*args)))


class gen_closure(partial):
    """Creates a closure for function `fun`, by passing the positional arguments
    provided in this function to `fun` before the ones given to the function and
    by passing the sum of named arguments given to both functions.

    The closure retains the original function name. If desired, it can
    be renamed using the `_fn` parameter. If fn contains `%s`, it will be
    replaced with the function name.

    The `_eat` parameter can be used to consume keyword arguments passed to the
    child function. Ex. if `_eat=["bob"]`, passing `bob=safd` will have no effect
    on the bound function.

    The `_return` parameter can override what the function returns."""

    def __new__(
        cls,
        func,
        /,
        *args,
        _fn: str | None = None,
        _eat: list[str] | None = None,
        _return: Any | None = None,
        **keywords,
    ):
        self = super().__new__(cls, func, *args, **keywords)

        if hasattr(func, "func"):
            _eat = _eat or func._eat
            _return = _return or func._return
            func = func.func

        if _fn:
            if hasattr(func, "__name__"):
                self.__name__ = _fn.replace("%s", func.__name__)  # type: ignore
            else:
                self.__name__ = _fn  # type: ignore
        else:
            if hasattr(func, "__name__"):
                self.__name__ = func.__name__  # type: ignore
            else:
                self.__name__ = "ukn"  # type: ignore

        # FIXME: probably get lost during multiprocessing
        self._eat = _eat  # type: ignore
        self._return = _return  # type: ignore
        return self

    def __call__(self, /, *args, **keywords):
        kw = keywords.copy()
        if self._eat:  # type: ignore
            for e in self._eat:  # type: ignore
                kw.pop(e, None)
        keywords = {**self.keywords, **kw}
        val = self.func(*self.args, *args, **keywords)
        if self._return is not None:  # type: ignore
            return self._return  # type: ignore
        return val


def _chunk_fun(fun, *args: Any, **kwargs: Any) -> set[Callable] | Callable:
    new_args = []
    datasets = {}
    # Fetch datasets from arguments
    for i, arg in enumerate(args):
        if isinstance(arg, LazyDataset):
            datasets[i] = arg
            new_args.append(None)
        else:
            new_args.append(arg)

    # Fetch datasets from keyword arguments
    new_kwargs = {}
    for name, arg in kwargs.items():
        if isinstance(arg, LazyDataset):
            datasets[name] = arg
        else:
            new_kwargs[name] = arg

    # If datasets are not partitioned, return a closure
    if not LazyDataset.are_partitioned(*datasets.values()):
        return {gen_closure(fun, *args, _canary=True, **kwargs)}

    # If they are, create proper arguments for each funciton call.
    closures = set()
    for pid, dataset_kw in LazyDataset.zip(datasets).items():
        out_args = new_args.copy()
        out_kwargs = new_kwargs.copy()

        for name, ds in dataset_kw.items():
            if isinstance(name, str):
                out_kwargs[name] = ds
            else:
                out_args[name] = ds

        closures.add(
            gen_closure(fun, *out_args, _canary=True, _partition=pid, **out_kwargs)
        )

    return closures


def to_chunked(
    func: Callable[P, A], /
) -> Callable[P, Callable[P, A] | dict[str, Callable[P, A]]]:
    """Makes wrapped function lazy evaluate. If args contain forms of
    `LazyDataset`, a dictionary of lazy functions are returned, each one loading
    one partition."""

    def wrapper(*args, _canary=False, _partition: str | None = None, **kwargs):
        """Wrapper acts like the real function when `_canary` is True, otherwise
        it runs `_chunk_fun()`.

        This way, the decorated function can both be used to create the lazy dictionary
        when called by end users and perform the evaluation when ran by one of the
        closures."""
        if _canary:
            out = func(*args, **kwargs)  # type: ignore

            # Try to partition output if partition is provided
            # dictionaries will have their values become a nested dictionary
            # with 1 key, the partition
            # Similarly with lists
            # Else, the output itself is wrapped
            if _partition:
                if isinstance(out, dict):
                    new_out = {}
                    for name, output in out.items():
                        new_out[name] = {_partition: output}
                    out = new_out
                elif isinstance(out, list):
                    new_out = []
                    for output in out:
                        new_out.append({_partition: out})
                    out = new_out
                else:
                    out = {_partition: out}

            return out
        else:
            return _chunk_fun(wrapper, *args, **kwargs)

    update_wrapper(wrapper, func)

    # Update annotations
    new_annotations = {}
    for param, orig in getattr(func, "__annotations__", {}).items():
        annotation = str(orig)

        if annotation == "LazyChunk":
            annotation = "LazyFrame"
        elif annotation == "LazyFrame":
            raise AttributeError("Can't wrap a LazyFrame with to_chunked")
        elif annotation.startswith("LazyDataset"):
            raise AttributeError("Can't wrap a LazyDataset with to_chunked")
        elif annotation.startswith("LazyPartition"):
            annotation = "LazyDataset" + annotation[len("LazyPartition") :]
        else:
            annotation = (
                orig  # preserve annotation object if it's not one of the lazies
            )

        new_annotations[param] = annotation

    wrapper.__annotations__ = new_annotations

    return wrapper  # type: ignore


def apply_fun(obj: Any, *args, _fun: str, **kwargs):
    """Runs function with name `_fun` of object `obj` with the provided arguments."""
    return getattr(obj, _fun)(*args, **kwargs)

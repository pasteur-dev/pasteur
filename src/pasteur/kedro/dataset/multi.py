import warnings
from copy import deepcopy
from typing import Any, Callable

from kedro.io.core import (
    VERSION_KEY,
    VERSIONED_FLAG_KEY,
    AbstractDataSet,
    DatasetError,
    parse_dataset_definition,
)
from kedro.io.partitioned_dataset import S3_PROTOCOLS


from urllib.parse import urlparse
class Multiset(AbstractDataSet):
    # noqa: too-many-instance-attributes,protected-access
    """Simplified version of the partitioned dataset. Is not lazy."""

    def __init__(  # noqa: too-many-arguments
        self,
        path: str,
        dataset: str | type[AbstractDataSet] | dict[str, Any],
        filepath_arg: str = "filepath",
        filename_suffix: str = "",
        credentials: dict[str, Any] | None = None,
        load_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        # noqa: import-outside-toplevel
        from fsspec.utils import infer_storage_options  # for performance reasons

        super().__init__()

        self._path = path
        self._filename_suffix = filename_suffix
        self._protocol = infer_storage_options(self._path)["protocol"]
        self.metadata = metadata

        dataset = dataset if isinstance(dataset, dict) else {"type": dataset}
        self._dataset_type, self._dataset_config = parse_dataset_definition(dataset)
        if VERSION_KEY in self._dataset_config:
            raise DatasetError(
                f"'{self.__class__.__name__}' does not support versioning of the "
                f"underlying dataset. Please remove '{VERSIONED_FLAG_KEY}' flag from "
                f"the dataset definition."
            )

        self._credentials = deepcopy(credentials) or {}
        self._filepath_arg = filepath_arg
        if self._filepath_arg in self._dataset_config:
            warnings.warn(
                f"'{self._filepath_arg}' key must not be specified in the dataset "
                f"definition as it will be overwritten by partition path"
            )

        self._load_args = deepcopy(load_args) or {}
        self._sep = self._filesystem.sep
        # since some filesystem implementations may implement a global cache
        self._invalidate_caches()

    @property
    def _filesystem(self):
        # for performance reasons
        import fsspec  # noqa: import-outside-toplevel

        protocol = "s3" if self._protocol in S3_PROTOCOLS else self._protocol
        return fsspec.filesystem(protocol, **self._credentials)

    @property
    def _normalized_path(self) -> str:
        if self._protocol in S3_PROTOCOLS:
            return urlparse(self._path)._replace(scheme="s3").geturl()
        return self._path

    def _list_partitions(self) -> list[str]:
        return [
            path
            for path in self._filesystem.find(self._normalized_path, **self._load_args)
            if path.endswith(self._filename_suffix)
        ]

    def _join_protocol(self, path: str) -> str:
        protocol_prefix = f"{self._protocol}://"
        if self._path.startswith(protocol_prefix) and not path.startswith(
            protocol_prefix
        ):
            return f"{protocol_prefix}{path}"
        return path

    def _partition_to_path(self, path: str):
        dir_path = self._path.rstrip(self._sep)
        path = path.lstrip(self._sep)
        full_path = self._sep.join([dir_path, path]) + self._filename_suffix
        return full_path

    def _path_to_partition(self, path: str) -> str:
        dir_path = self._filesystem._strip_protocol(self._normalized_path)
        path = path.split(dir_path, 1).pop().lstrip(self._sep)
        if self._filename_suffix and path.endswith(self._filename_suffix):
            path = path[: -len(self._filename_suffix)]
        return path

    def _load(self) -> dict[str, Callable[[], Any]]:
        partitions = {}

        for partition in self._list_partitions():
            kwargs = deepcopy(self._dataset_config)
            # join the protocol back since PySpark may rely on it
            kwargs[self._filepath_arg] = self._join_protocol(partition)
            dataset = self._dataset_type(**kwargs)  # type: ignore
            partition_id = self._path_to_partition(partition)
            partitions[partition_id] = dataset.load()

        return partitions

    def _save(self, data: dict[str, Any]) -> None:
        for partition_id, partition_data in sorted(data.items()):
            kwargs = deepcopy(self._dataset_config)
            partition = self._partition_to_path(partition_id)
            # join the protocol back since tools like PySpark may rely on it
            kwargs[self._filepath_arg] = self._join_protocol(partition)
            dataset = self._dataset_type(**kwargs)  # type: ignore
            if callable(partition_data):
                partition_data = partition_data()  # noqa: redefined-loop-name
            dataset.save(partition_data)

        self._invalidate_caches()

    def _describe(self) -> dict[str, Any]:
        clean_dataset_config = (
            {k: v for k, v in self._dataset_config.items()}
            if isinstance(self._dataset_config, dict)
            else self._dataset_config
        )
        return {
            "path": self._path,
            "dataset_type": self._dataset_type.__name__,
            "dataset_config": clean_dataset_config,
        }

    def _invalidate_caches(self):
        self._filesystem.invalidate_cache(self._normalized_path)

    def reset(self):
        """Removes the dataset from disk so that there are no stray partitions in subsequent runs."""
        if self._filesystem.exists(self._normalized_path):
            self._filesystem.rm(self._normalized_path, recursive=True, maxdepth=1)

    def _release(self) -> None:
        super()._release()
        self._invalidate_caches()

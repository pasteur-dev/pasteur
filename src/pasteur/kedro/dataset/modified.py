import importlib
import logging
import re
from copy import deepcopy
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any

import fsspec
import pandas as pd
from kedro.io.core import (
    PROTOCOL_DELIMITER,
    AbstractVersionedDataset,
    DatasetError,
    Version,
    get_filepath_str,
    get_protocol_and_path,
)
from kedro_datasets.partitions import PartitionedDataset

from ...utils import LazyDataset, LazyPartition
from .auto import _wrap_retry

logger = logging.getLogger(__name__)


class PatternDataset(PartitionedDataset):
    """Adds pattern support to Partitioned Dataset"""

    def __init__(
        self,
        path: str,
        dataset,
        filepath_arg: str = "filepath",
        filename_suffix: str = "",
        credentials=None,
        load_args=None,
        fs_args=None,
        overwrite: bool = False,
        pattern: str = "",
        replace_pattern: str = "",
        replace_format: str = "",
    ):

        super().__init__(
            path,
            dataset,
            filepath_arg,
            filename_suffix,
            credentials,  # type: ignore
            load_args,  # type: ignore
            fs_args,  # type: ignore
            overwrite,
        )
        self._pattern = pattern
        self._replace_pattern = replace_pattern
        self._replace_format = replace_format

    def _list_partitions(self):
        if not self._pattern:
            return super()._list_partitions()
        return list(
            filter(lambda p: re.search(self._pattern, p), super()._list_partitions())
        )

    def _path_to_partition(self, path: str) -> str:
        id = super()._path_to_partition(path)
        if not self._replace_pattern or not self._replace_format:
            return id

        m = re.search(self._replace_pattern, path)
        if not m:
            return id

        return self._replace_format.format(*m.groups(), **m.groupdict())

    def _load(self):
        return LazyDataset(
            None,
            {pid: LazyPartition(fun, None) for pid, fun in super()._load().items()},
        )


def _load_csv(
    protocol: str,
    load_path: str,
    storage_options,
    load_args: dict,
    columns: list[str] | None = None,
    chunksize: int | None = None,
):
    if columns is not None or chunksize is not None:
        load_args = load_args.copy()
    if columns is not None:
        load_args["usecols"] = columns

        if "parse_dates" in load_args:
            new_dates = []
            for date in load_args["parse_dates"]:
                if date in columns:
                    new_dates.append(date)
            load_args["parse_dates"] = new_dates
    if chunksize is not None:
        load_args["chunksize"] = chunksize

    if protocol == "file":
        return pd.read_csv(load_path, **load_args)

    load_path = f"{protocol}{PROTOCOL_DELIMITER}{load_path}"
    return pd.read_csv(load_path, storage_options=storage_options, **load_args)


class FragmentedCSVDataset(AbstractVersionedDataset[pd.DataFrame, pd.DataFrame]):

    DEFAULT_LOAD_ARGS: dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {"index": False}

    def __init__(
        self,
        filepath: str,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        version: Version | None = None,
        credentials: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        chunksize: int | None = None,
    ) -> None:
        if chunksize:
            if load_args:
                load_args = load_args.copy()
                load_args["chunksize"] = chunksize
            else:
                load_args = {"chunksize": chunksize}

        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}

        protocol, path = get_protocol_and_path(filepath, version)  # type: ignore
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._storage_options = {**_credentials, **_fs_args}
        self._fs = fsspec.filesystem(self._protocol, **self._storage_options)

        self.metadata = metadata

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

        # Handle default load and save arguments
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

        if "storage_options" in self._save_args or "storage_options" in self._load_args:
            logger.warning(
                "Dropping 'storage_options' for %s, "
                "please specify them under 'fs_args' or 'credentials'.",
                self._filepath,
            )
            self._save_args.pop("storage_options", None)
            self._load_args.pop("storage_options", None)

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._filepath,
            "protocol": self._protocol,
            "load_args": self._load_args,
            "save_args": self._save_args,
            "version": self._version,
        }

    def _load(self) -> LazyDataset:
        load_path = str(self._get_load_path())
        return LazyDataset(
            LazyPartition(
                _load_csv,
                None,
                self._protocol,
                load_path,
                self._storage_options,
                self._load_args,
            )
        )

    def _save(self, data: pd.DataFrame) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)

        buf = BytesIO()
        data.to_csv(path_or_buf=buf, **self._save_args)  # type: ignore

        with self._fs.open(save_path, mode="wb") as fs_file:
            fs_file.write(buf.getvalue())

        self._invalidate_cache()

    def _exists(self) -> bool:
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DatasetError:
            return False

        return self._fs.exists(load_path)

    def _release(self) -> None:
        super()._release()
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate underlying filesystem caches."""
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)

    def _preview(self, nrows: int = 40) -> dict:
        # Create a copy so it doesn't contaminate the original dataset
        dataset_copy = self._copy()
        dataset_copy._load_args["nrows"] = nrows  # type: ignore
        data = dataset_copy.load()

        return data.to_dict(orient="split")


class PickleDataset(
    AbstractVersionedDataset[Any, Any]
):  # pylint:disable=too-many-instance-attributes
    """``PickleDataset`` loads/saves data from/to a Pickle file using an underlying
    filesystem (e.g.: local, S3, GCS). The underlying functionality is supported by
    the specified backend library passed in (defaults to the ``pickle`` library), so it
    supports all allowed options for loading and saving pickle files.

    Example usage for the
    `YAML API <https://kedro.readthedocs.io/en/stable/data/\
    data_catalog.html#use-the-data-catalog-with-the-yaml-api>`_:

    .. code-block:: yaml

        test_model: # simple example without compression
          type: pickle.PickleDataset
          filepath: data/07_model_output/test_model.pkl
          backend: pickle

        final_model: # example with load and save args
          type: pickle.PickleDataset
          filepath: s3://your_bucket/final_model.pkl.lz4
          backend: joblib
          credentials: s3_credentials
          save_args:
            compress: lz4

    Example usage for the
    `Python API <https://kedro.readthedocs.io/en/stable/data/\
    data_catalog.html#use-the-data-catalog-with-the-code-api>`_:
    ::

        >>> from kedro_datasets.pickle import PickleDataset
        >>> import pandas as pd
        >>>
        >>> data = pd.DataFrame({'col1': [1, 2], 'col2': [4, 5],
        >>>                      'col3': [5, 6]})
        >>>
        >>> data_set = PickleDataset(filepath="test.pkl", backend="pickle")
        >>> data_set.save(data)
        >>> reloaded = data_set.load()
        >>> assert data.equals(reloaded)
        >>>
        >>> data_set = PickleDataset(filepath="test.pickle.lz4",
        >>>                          backend="compress_pickle",
        >>>                          load_args={"compression":"lz4"},
        >>>                          save_args={"compression":"lz4"})
        >>> data_set.save(data)
        >>> reloaded = data_set.load()
        >>> assert data.equals(reloaded)
    """

    DEFAULT_LOAD_ARGS: dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {}

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(
        self,
        filepath: str,
        backend: str = "pickle",
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        version: Version | None = None,
        credentials: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Creates a new instance of ``PickleDataset`` pointing to a concrete Pickle
        file on a specific filesystem. ``PickleDataset`` supports custom backends to
        serialise/deserialise objects.

        Example backends that are compatible (non-exhaustive):
            * `pickle`
            * `joblib`
            * `dill`
            * `compress_pickle`

        Example backends that are incompatible:
            * `torch`

        Args:
            filepath: Filepath in POSIX format to a Pickle file prefixed with a protocol like
                `s3://`. If prefix is not provided, `file` protocol (local filesystem) will be used.
                The prefix should be any protocol supported by ``fsspec``.
                Note: `http(s)` doesn't support versioning.
            backend: Backend to use, must be an import path to a module which satisfies the
                ``pickle`` interface. That is, contains a `load` and `dump` function.
                Defaults to 'pickle'.
            load_args: Pickle options for loading pickle files.
                You can pass in arguments that the backend load function specified accepts, e.g:
                pickle.load: https://docs.python.org/3/library/pickle.html#pickle.load
                joblib.load: https://joblib.readthedocs.io/en/latest/generated/joblib.load.html
                dill.load: https://dill.readthedocs.io/en/latest/index.html#dill.load
                compress_pickle.load:
                https://lucianopaz.github.io/compress_pickle/html/api/compress_pickle.html#compress_pickle.compress_pickle.load
                All defaults are preserved.
            save_args: Pickle options for saving pickle files.
                You can pass in arguments that the backend dump function specified accepts, e.g:
                pickle.dump: https://docs.python.org/3/library/pickle.html#pickle.dump
                joblib.dump: https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html
                dill.dump: https://dill.readthedocs.io/en/latest/index.html#dill.dump
                compress_pickle.dump:
                https://lucianopaz.github.io/compress_pickle/html/api/compress_pickle.html#compress_pickle.compress_pickle.dump
                All defaults are preserved.
            version: If specified, should be an instance of
                ``kedro.io.core.Version``. If its ``load`` attribute is
                None, the latest version will be loaded. If its ``save``
                attribute is None, save version will be autogenerated.
            credentials: Credentials required to get access to the underlying filesystem.
                E.g. for ``GCSFileSystem`` it should look like `{"token": None}`.
            fs_args: Extra arguments to pass into underlying filesystem class constructor
                (e.g. `{"project": "my-project"}` for ``GCSFileSystem``), as well as
                to pass to the filesystem's `open` method through nested keys
                `open_args_load` and `open_args_save`.
                Here you can find all available arguments for `open`:
                https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.open
                All defaults are preserved, except `mode`, which is set to `wb` when saving.
            metadata: Any arbitrary metadata.
                This is ignored by Kedro, but may be consumed by users or external plugins.

        Raises:
            ValueError: If ``backend`` does not satisfy the `pickle` interface.
            ImportError: If the ``backend`` module could not be imported.
        """
        # We do not store `imported_backend` as an attribute to be used in `load`/`save`
        # as this would mean the dataset cannot be deepcopied (module objects cannot be
        # pickled). The import here is purely to raise any errors as early as possible.
        # Repeated imports in the `load` and `save` methods should not be a significant
        # performance hit as Python caches imports.
        try:
            imported_backend = importlib.import_module(backend)
        except ImportError as exc:
            raise ImportError(
                f"Selected backend '{backend}' could not be imported. "
                "Make sure it is installed and importable."
            ) from exc

        if not (
            hasattr(imported_backend, "load") and hasattr(imported_backend, "dump")
        ):
            raise ValueError(
                f"Selected backend '{backend}' should satisfy the pickle interface. "
                "Missing one of 'load' and 'dump' on the backend."
            )

        _fs_args = deepcopy(fs_args) or {}
        _fs_open_args_load = _fs_args.pop("open_args_load", {})
        _fs_open_args_save = _fs_args.pop("open_args_save", {})
        _credentials = deepcopy(credentials) or {}

        protocol, path = get_protocol_and_path(filepath, version)  # type: ignore
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol, **_credentials, **_fs_args)

        self.metadata = metadata

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

        self._backend = backend

        # Handle default load and save arguments
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

        _fs_open_args_save.setdefault("mode", "wb")
        self._fs_open_args_load = _fs_open_args_load
        self._fs_open_args_save = _fs_open_args_save

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._filepath,
            "backend": self._backend,
            "protocol": self._protocol,
            "load_args": self._load_args,
            "save_args": self._save_args,
            "version": self._version,
        }

    @_wrap_retry
    def _load(self) -> Any:
        load_path = get_filepath_str(self._get_load_path(), self._protocol)

        with self._fs.open(load_path, **self._fs_open_args_load) as fs_file:
            imported_backend = importlib.import_module(self._backend)
            return imported_backend.load(fs_file, **self._load_args)  # type: ignore

    @_wrap_retry
    def _save(self, data: Any) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)

        with self._fs.open(save_path, **self._fs_open_args_save) as fs_file:
            try:
                imported_backend = importlib.import_module(self._backend)
                imported_backend.dump(data, fs_file, **self._save_args)  # type: ignore
            except Exception as exc:
                raise DatasetError(
                    f"{data.__class__} was not serialised due to: {exc}"
                ) from exc

        self._invalidate_cache()

    def _exists(self) -> bool:
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DatasetError:
            return False

        return self._fs.exists(load_path)

    def _release(self) -> None:
        super()._release()
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate underlying filesystem caches."""
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)

""" This module provides kedro datasets that have been customized to suit Pasteur's needs.

The most notable additions are that the datasets lazy load data through `PartitionedDataset`
and can partition save data through custom `Node` return types.

Since Pasteur  """

import importlib
import logging
import os
import re
from copy import deepcopy
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any, Callable

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from kedro.io.core import (
    PROTOCOL_DELIMITER,
    AbstractVersionedDataSet,
    DataSetError,
    Version,
    get_filepath_str,
    get_protocol_and_path,
)
from kedro.io.partitioned_dataset import PartitionedDataSet

from ..utils import LazyDataset, LazyFrame, LazyPartition
from ..utils.progress import get_node_name, process, process_in_parallel

logger = logging.getLogger(__name__)


class PatternDataSet(PartitionedDataSet):
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


def _save_worker(
    pid: str | None,
    path: str,
    chunk: Callable[..., pd.DataFrame] | pd.DataFrame,
    protocol,
    fs,
    save_args,
):
    if pid:
        logging.debug(f"Saving chunk {pid}...")

    if callable(chunk):
        chunk = chunk()
        if callable(chunk):
            logger.error(
                f"Callable `chunk()` got double wrapped (`to_chunked()` bug).\n{str(chunk)[:50]}"
            )
            chunk = chunk()

    from inspect import isgenerator

    # Handle data partitioning using generators, to avoid storing the whole partition in ram
    # or having to use pd.concat()
    if isgenerator(chunk):
        # Grab first chunk with content
        p0 = None
        try:
            while p0 is None or len(p0) == 0:
                p0 = next(chunk)
        except:
            logger.error(f"Generator {chunk} returned no data.")
            return

        old_schema = pa.Schema.from_pandas(p0, preserve_index=True)

        # FIXME: Schema inference for pyarrow
        # null columns will lead to invalid schema
        # int8 dictionaries in first chunk which become int16 will lead to invalid schema
        # try to fix both
        fields = []
        dtypes = p0.dtypes
        for field in old_schema:
            if (
                isinstance(field.type, pa.dictionaryType)
                and field.type.index_type.bit_width == 8
            ):
                # Expand uint8 dictionaries to uint16
                fields.append(
                    pa.field(
                        field.name,
                        pa.dictionary(pa.int16(), field.type.value_type),
                        field.nullable,
                        field.metadata,
                    )
                )
            elif field.name in dtypes and field.type == pa.null():
                # Fix missing types based on pandas dtype
                # might produce larger than required types, but better than failing.
                # If field is not in dtype, assume it's related to pyarrow and skip.
                match (dtypes[field.name].name):
                    case "int64":
                        pa_type = pa.int64()
                    case other:
                        logger.warning(
                            f"Could not infer type for empty column `{field.name}`"
                            + f" with pandas type `{other}` to generate parquet"
                            + "schema. If there's a chunk who's column contains"
                            + "values, saving will crash. Fill in the code for your type."
                        )
                        pa_type = pa.null()

                fields.append(
                    pa.field(
                        field.name,
                        pa_type,
                        field.nullable,
                        field.metadata,
                    )
                )
            else:
                fields.append(field)
        schema = pa.schema(fields, old_schema.metadata)

        # Use parquet writer to write chunks
        with pq.ParquetWriter(path, schema, filesystem=fs) as w:
            w.write(pa.Table.from_pandas(p0, schema=schema))
            del p0

            for p in chunk:
                try:
                    w.write(pa.Table.from_pandas(p, schema=schema))
                except Exception as e:
                    logger.error(f"Error writing chunk:\n{e}")
    else:
        if protocol == "file":
            if fs.isdir(path):
                fs.rm(path, recursive=True, maxdepth=1)

            with fs.open(path, mode="wb") as fs_file:
                chunk.to_parquet(fs_file, **save_args)
        else:
            bytes_buffer = BytesIO()
            chunk.to_parquet(bytes_buffer, **save_args)

            with fs.open(path, mode="wb") as fs_file:
                fs_file.write(bytes_buffer.getvalue())


def _load_worker(
    path: str,
    protocol: str,
    storage_options,
    load_args: dict,
    columns: list[str] | None = None,
):
    if protocol == "file":
        # file:// protocol seems to misbehave on Windows
        # (<urlopen error file not on local host>),
        # so we don't join that back to the filepath;
        # storage_options also don't work with local paths
        return pd.read_parquet(path, **load_args)

    load_path = f"{protocol}{PROTOCOL_DELIMITER}{path}"

    if columns is not None:
        load_args = load_args.copy()
        load_args["columns"] = columns

    return pd.read_parquet(load_path, storage_options=storage_options, **load_args)


def _load_merged_worker(
    load_path: str, filesystem, load_args, columns: list[str] | None = None
):
    if columns is not None:
        load_args = load_args.copy()
        load_args["columns"] = columns

    data = pq.ParquetDataset(load_path, filesystem=filesystem, use_legacy_dataset=False)
    table = data.read_pandas(**load_args)

    # Grab categorical columns from metadata
    # null columns that are specified as categorical in pandas metadata
    # will become objects after loading, ballooning dataset size
    # the following code will remake the column as categorical
    try:
        import json

        categorical = []
        for field in json.loads(table.schema.metadata[b"pandas"])["columns"]:
            if (field["pandas_type"]) == "categorical":
                categorical.append(field["name"])

        dtypes = {name: "category" for name in categorical}
    except:
        dtypes = None

    # Try to avoid double allocation
    out = table.to_pandas(split_blocks=True, self_destruct=True)
    del table

    # restore categorical dtypes
    if dtypes:
        return out.astype(dtypes)
    return out


def _load_shape_worker(load_path: str, filesystem, *_, **__):
    # TODO: verify this returns correct numbers (esp. columns)
    data = pq.ParquetDataset(load_path, filesystem=filesystem, use_legacy_dataset=False)
    rows = 0
    for frag in data.fragments:
        rows += frag.count_rows()

    pm = data.schema.pandas_metadata  # type: ignore
    cols = len(pm["columns"]) - len(
        [c for c in pm["index_columns"] if isinstance(c, str)]
    )

    return (rows, cols)


class AutoDataset(AbstractVersionedDataSet[pd.DataFrame, pd.DataFrame]):
    """Modified kedro parquet dataset that acts similarly to a partitioned dataset
    and implements lazy loading.

    `save()` data can be a table, a callable, or a dictionary combination of both.

    If its a table or a callable, this class acts exactly as ParquetDataSet.
    If its a dictionary, each callable function is called and saved in parallel
    in a different parquet file, making the provided path a directory.
    Parallelism is achieved by using Pasteur's common process pool.

    `load()` returns a dictionary with parquet file names and callables that will
    load each one. In addition, `load()` will include an entry `_all` that will
    load and concatenate all partitions, with memory optimisations.
    If `save()` was called with a single dataframe/callable, then `load()` will
    return a callable instead.
    All callables can receive as input the columns they want to be loaded from the
    dataframe."""

    DEFAULT_LOAD_ARGS: dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: dict[str, Any] = {}

    def __init__(
        self,
        filepath: str,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        version: Version | None = None,
        credentials: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
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
            filepath=PurePosixPath(path),  # type: ignore
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

    def _exists(self) -> bool:
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DataSetError:
            return False

        return self._fs.exists(load_path)

    def _release(self) -> None:
        super()._release()
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate underlying filesystem caches."""
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)

    def _load(self) -> LazyFrame:
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        if not self._fs.isdir(load_path):
            return LazyDataset(
                LazyPartition(
                    _load_merged_worker,
                    _load_shape_worker,
                    load_path,
                    self._fs,
                    self._load_args,
                )
            )

        partitions = {}
        for fn in self._fs.listdir(load_path):
            partition_id = fn["name"].split("/")[-1].split("\\")[-1].replace(".pq", "")
            partition_data = LazyPartition(
                _load_merged_worker,
                _load_shape_worker,
                fn["name"],
                self._fs,
                self._load_args,
            )
            partitions[partition_id] = partition_data

        merged_partition = LazyPartition(
            _load_merged_worker,
            _load_shape_worker,
            load_path,
            self._fs,
            self._load_args,
        )

        return LazyDataset(merged_partition, partitions)

    def _get_save_path(self):
        if not self._version:
            # When versioning is disabled, return original filepath
            return self._filepath

        save_version = self.resolve_save_version()
        versioned_path = self._get_versioned_path(save_version)  # type: ignore

        # TODO; Redo check that respects partitioning
        # if self._exists_function(str(versioned_path)):
        #     raise DataSetError(
        #         f"Save path '{versioned_path}' for {str(self)} must not exist if "
        #         f"versioning is enabled."
        #     )

        return versioned_path

    def _save(self, data: pd.DataFrame) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)

        if (not isinstance(data, dict) and not isinstance(data, LazyDataset)) or (
            isinstance(data, LazyDataset) and not data.partitioned
        ):
            process(
                _save_worker,
                protocol=self._protocol,
                fs=self._fs,
                save_args=self._save_args,
                pid=None,
                path=save_path,
                chunk=data,
            )
            return

        base_args = {
            "protocol": self._protocol,
            "fs": self._fs,
            "save_args": self._save_args,
        }
        jobs = []
        for pid, partition_data in sorted(data.items()):
            chunk_save_path = os.path.join(
                save_path, pid if pid.endswith(".pq") else pid + ".pq"  # type: ignore
            )
            jobs.append({"pid": pid, "path": chunk_save_path, "chunk": partition_data})

        if not jobs:
            return

        self._fs.mkdirs(save_path, exist_ok=True)

        process_in_parallel(
            _save_worker,
            jobs,
            base_args,
            1,
            f"Processing chunks ({get_node_name():>25s})",
        )

        self._invalidate_cache()

    def _save_no_buff(self, save_path: str, data: pd.DataFrame):
        if self._protocol == "file":
            with self._fs.open(save_path, mode="wb") as fs_file:
                data.to_parquet(fs_file, **self._save_args)
        else:
            bytes_buffer = BytesIO()
            data.to_parquet(bytes_buffer, **self._save_args)

            with self._fs.open(save_path, mode="wb") as fs_file:
                fs_file.write(bytes_buffer.getvalue())

    def reset(self):
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        if self._fs.exists(save_path):
            self._fs.rm(save_path, recursive=True, maxdepth=1)


""" Backwards compatibility. `AutoDataset` replaces `FragmentedParquetDataset`"""
FragmentedParquetDataset = AutoDataset


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


class FragmentedCSVDataset(AbstractVersionedDataSet[pd.DataFrame, pd.DataFrame]):

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
        data.to_csv(path_or_buf=buf, **self._save_args)

        with self._fs.open(save_path, mode="wb") as fs_file:
            fs_file.write(buf.getvalue())

        self._invalidate_cache()

    def _exists(self) -> bool:
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DataSetError:
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


class PickleDataSet(
    AbstractVersionedDataSet[Any, Any]
):  # pylint:disable=too-many-instance-attributes
    """``PickleDataSet`` loads/saves data from/to a Pickle file using an underlying
    filesystem (e.g.: local, S3, GCS). The underlying functionality is supported by
    the specified backend library passed in (defaults to the ``pickle`` library), so it
    supports all allowed options for loading and saving pickle files.

    Example usage for the
    `YAML API <https://kedro.readthedocs.io/en/stable/data/\
    data_catalog.html#use-the-data-catalog-with-the-yaml-api>`_:

    .. code-block:: yaml

        test_model: # simple example without compression
          type: pickle.PickleDataSet
          filepath: data/07_model_output/test_model.pkl
          backend: pickle

        final_model: # example with load and save args
          type: pickle.PickleDataSet
          filepath: s3://your_bucket/final_model.pkl.lz4
          backend: joblib
          credentials: s3_credentials
          save_args:
            compress: lz4

    Example usage for the
    `Python API <https://kedro.readthedocs.io/en/stable/data/\
    data_catalog.html#use-the-data-catalog-with-the-code-api>`_:
    ::

        >>> from kedro_datasets.pickle import PickleDataSet
        >>> import pandas as pd
        >>>
        >>> data = pd.DataFrame({'col1': [1, 2], 'col2': [4, 5],
        >>>                      'col3': [5, 6]})
        >>>
        >>> data_set = PickleDataSet(filepath="test.pkl", backend="pickle")
        >>> data_set.save(data)
        >>> reloaded = data_set.load()
        >>> assert data.equals(reloaded)
        >>>
        >>> data_set = PickleDataSet(filepath="test.pickle.lz4",
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
        metadata: dict[str, Any] | None= None,
    ) -> None:
        """Creates a new instance of ``PickleDataSet`` pointing to a concrete Pickle
        file on a specific filesystem. ``PickleDataSet`` supports custom backends to
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

        protocol, path = get_protocol_and_path(filepath, version) # type: ignore
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

    def _load(self) -> Any:
        load_path = get_filepath_str(self._get_load_path(), self._protocol)

        with self._fs.open(load_path, **self._fs_open_args_load) as fs_file:
            imported_backend = importlib.import_module(self._backend)
            return imported_backend.load(fs_file, **self._load_args)  # type: ignore

    def _save(self, data: Any) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)

        with self._fs.open(save_path, **self._fs_open_args_save) as fs_file:
            try:
                imported_backend = importlib.import_module(self._backend)
                imported_backend.dump(data, fs_file, **self._save_args)  # type: ignore
            except Exception as exc:
                raise DataSetError(
                    f"{data.__class__} was not serialised due to: {exc}"
                ) from exc

        self._invalidate_cache()

    def _exists(self) -> bool:
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DataSetError:
            return False

        return self._fs.exists(load_path)

    def _release(self) -> None:
        super()._release()
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate underlying filesystem caches."""
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)

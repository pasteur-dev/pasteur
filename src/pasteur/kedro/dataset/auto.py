import logging
import os
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

from ...utils import LazyDataset, LazyFrame, LazyPartition
from ...utils.progress import get_node_name, process, process_in_parallel

logger = logging.getLogger(__name__)


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

            for p in chunk: # type: ignore
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
    
    In the future, this dataset will automatically handle pickling, pyarrow 
    Tables, DataFrames, and Tensors automatically based on what is saved.

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

    def reset(self):
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        if self._fs.exists(save_path):
            self._fs.rm(save_path, recursive=True, maxdepth=1)
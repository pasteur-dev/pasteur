import logging
import os
import re
from io import BytesIO
from typing import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from kedro.extras.datasets.pandas.csv_dataset import CSVDataSet
from kedro.extras.datasets.pandas.parquet_dataset import ParquetDataSet
from kedro.io.core import PROTOCOL_DELIMITER, get_filepath_str
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
                isinstance(field.type, pa.DictionaryType)
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
    cols = len(pm["columns"]) - len([c for c in pm["index_columns"] if isinstance(c, str)])

    return (rows, cols)


class FragmentedParquetDataset(ParquetDataSet):
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

    def __init__(
        self,
        filepath: str,
        load_args=None,
        save_args=None,
        version=None,
        credentials=None,
        fs_args=None,
    ) -> None:
        super().__init__(filepath, load_args, save_args, version, credentials, fs_args)  # type: ignore

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


class FragmentedCSVDataset(CSVDataSet):
    def __init__(
        self,
        filepath: str,
        load_args=None,
        save_args=None,
        version=None,
        credentials=None,
        fs_args=None,
        chunksize: int | None = None,
    ) -> None:
        if chunksize:
            if load_args:
                load_args = load_args.copy()
                load_args["chunksize"] = chunksize
            else:
                load_args = {"chunksize": chunksize}
        super().__init__(filepath, load_args, save_args, version, credentials, fs_args)  # type: ignore

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

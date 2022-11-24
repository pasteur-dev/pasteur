import logging
import os
import re
from io import BytesIO
from typing import Callable, Any

import pandas as pd
import pyarrow.parquet as pq
from kedro.extras.datasets.pandas.parquet_dataset import ParquetDataSet
from kedro.io.core import DataSetError, get_filepath_str, PROTOCOL_DELIMITER
from kedro.io.partitioned_dataset import PartitionedDataSet

from ..utils.progress import process_in_parallel
from ..utils import gen_closure, LazyFrame

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


def parallel_save_worker(
    pid: str,
    path: str,
    chunk: Callable[..., pd.DataFrame] | pd.DataFrame,
    protocol,
    fs,
    save_args,
):
    logging.debug(f"Saving chunk {pid}...")

    if callable(chunk):
        chunk = chunk()

    if protocol == "file":
        with fs.open(path, mode="wb") as fs_file:
            chunk.to_parquet(fs_file, **save_args)
    else:
        bytes_buffer = BytesIO()
        chunk.to_parquet(bytes_buffer, **save_args)

        with fs.open(path, mode="wb") as fs_file:
            fs_file.write(bytes_buffer.getvalue())


def load_worker(path: str, protocol: str, storage_options, load_args):
    if protocol == "file":
        # file:// protocol seems to misbehave on Windows
        # (<urlopen error file not on local host>),
        # so we don't join that back to the filepath;
        # storage_options also don't work with local paths
        return pd.read_parquet(path, **load_args)

    load_path = f"{protocol}{PROTOCOL_DELIMITER}{path}"
    return pd.read_parquet(load_path, storage_options=storage_options, **load_args)


class FragmentedParquetDataset(ParquetDataSet):
    """Modified kedro parquet dataset that acts similarly to a partitioned dataset.

    `save()` data can be a table, a callable, or a dictionary combination of both.

    If its a table or a callable, this class acts exactly as ParquetDataSet.
    If its a dictionary, each callable function is called and saved in parallel
    in a different parquet file, making the provided path a directory.

    `load()` merges all parquet files into one table and returns a DataFrame.
    It also uses optimisations to avoid double memory allocation as much as possible.

    if `partition_load` is set to true and `save()` was called with a dictionary,
    `load()` returns an equivalent dictionary of callables.

    The partitioned form of `load()` is used for datasets, where their size might
    be larger than the available RAM, making them ingestable in machines with little
    RAM. Then, when the views are split, the non partitioned form of `load()` is
    used, merging the chunks. Provided that an appropriate ratio is chosen for the
    splits, the combined chunks will be tractable for the provided RAM.

    Out-of-core learning is experimental in most algorithms, so providing it for the
    whole framework provides little utility for now."""

    def __init__(
        self,
        filepath: str,
        load_args=None,
        save_args=None,
        version=None,
        credentials=None,
        fs_args=None,
        partition_load: bool = False,
    ) -> None:
        super().__init__(filepath, load_args, save_args, version, credentials, fs_args)  # type: ignore
        self.partition_load = partition_load

    def _load_merged(self) -> pd.DataFrame:
        load_path = get_filepath_str(self._get_load_path(), self._protocol)

        data = pq.ParquetDataset(
            load_path, filesystem=self._fs, use_legacy_dataset=False
        )
        table = data.read_pandas(**self._load_args)
        # Try to avoid double allocation
        return table.to_pandas(split_blocks=True, self_destruct=True)

    def _load(self) -> LazyFrame | pd.DataFrame:
        if not self.partition_load:
            return self._load_merged()

        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        if not self._fs.isdir(load_path):
            return self._load_merged()

        partitions = {}
        for fn in self._fs.listdir(load_path):
            partition_id = fn["name"].split("/")[-1].split("\\")[-1].replace(".pq", "")
            partition_data = gen_closure(
                load_worker,
                fn["name"],
                self._protocol,
                self._storage_options,
                self._load_args,
            )
            partitions[partition_id] = partition_data

        return partitions

    def _save(self, data: pd.DataFrame) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)

        if isinstance(data, pd.DataFrame):
            return self._save_no_buff(save_path, data)

        if callable(data):
            return self._save_no_buff(save_path, data())

        if not isinstance(data, dict):
            raise DataSetError(
                f"{self.__class__.__name__} arguments should be DataFrames, callables, or dicts of dataframes and callables, not {type[data]}."
            )

        if self._fs.exists(save_path):
            self._fs.rm(save_path, recursive=True)

        base_args = {
            "protocol": self._protocol,
            "fs": self._fs,
            "save_args": self._save_args,
        }
        jobs = []
        for pid, partition_data in sorted(data.items()):
            chunk_save_path = os.path.join(
                save_path, pid if pid.endswith(".pq") else pid + ".pq"
            )
            jobs.append({"pid": pid, "path": chunk_save_path, "chunk": partition_data})

        process_in_parallel(
            parallel_save_worker, jobs, base_args, 1, f"Saving parquet chunks"
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

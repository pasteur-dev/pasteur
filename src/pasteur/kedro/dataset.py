import logging
import os
import re
from io import BytesIO
from typing import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from kedro.extras.datasets.pandas.parquet_dataset import ParquetDataSet
from kedro.io.core import DataSetError, get_filepath_str
from kedro.io.partitioned_dataset import PartitionedDataSet

from ..utils.progress import process_in_parallel

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


class FragmentedParquetDataset(ParquetDataSet):
    """Modified kedro parquet dataset that allows for lazy saving to multiple
    parquet files.

    Used to prevent consuming 2x RAM when ingesting data from CSVs.
    Each csv is saved to its own parquet file and then parquet loads the directory
    automatically on `_load()`.

    Saving chunks runs in parallel."""

    def _load(self) -> pd.DataFrame:
        load_path = get_filepath_str(self._get_load_path(), self._protocol)

        if not self._fs.isdir(load_path):
            return self._load_from_pandas()
        
        # Create complete schema using dataset
        data = pq.ParquetDataset(
            load_path, filesystem=self._fs, use_legacy_dataset=False
        )
        table = data.read_pandas(**self._load_args)
        # Try to avoid double allocation
        return table.to_pandas(split_blocks=True, self_destruct=True)



    def _save(self, data: pd.DataFrame) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)

        if isinstance(data, pd.DataFrame):
            return self._save_no_buff(save_path, data)

        if callable(data):
            return self._save_no_buff(save_path, data())

        if not isinstance(data, dict):
            raise DataSetError(
                f"{self.__class__.__name__} arguments should be DataFrames, callables, or dicts of dataframes, callables"
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

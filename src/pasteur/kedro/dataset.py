import os
import re
from io import BytesIO
from pathlib import Path

import pandas as pd
from kedro.extras.datasets.pandas.parquet_dataset import ParquetDataSet
from kedro.io.core import DataSetError, get_filepath_str
from kedro.io.partitioned_dataset import PartitionedDataSet


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


class FragmentedParquetDataset(ParquetDataSet):
    """Modified kedro parquet dataset that allows for lazy saving to multiple
    parquet files.
    
    Used to prevent consuming 2x RAM when ingesting data from CSVs.
    Each csv is saved to its own parquet file and then parquet loads the directory
    automatically on `_load()`."""

    def _save(self, data: pd.DataFrame) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)

        if isinstance(data, pd.DataFrame):
            return self._save_no_buff(save_path, data)
        
        if callable(data):
            return self._save_no_buff(save_path, data())
        
        if not isinstance(data, dict):
            raise DataSetError(f"{self.__class__.__name__} arguments should be DataFrames, callables, or dicts of dataframes, callables")

        if self._fs.exists(save_path):
            self._fs.rm(save_path, recursive=True)
        
        for pid, partition_data in sorted(data.items()):
            if callable(partition_data):
                partition_data = partition_data()
            
            chunk_save_path = os.path.join(save_path, pid if pid.endswith(".pq") else pid + ".pq")
            self._save_no_buff(chunk_save_path, partition_data)

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
        
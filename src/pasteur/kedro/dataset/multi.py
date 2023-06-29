from kedro.io.partitioned_dataset import PartitionedDataSet


class Multiset(PartitionedDataSet):
    """Modified Partitioned Dataset for pasteur."""

    def reset(self):
        """Removes the dataset from disk so that there are no stray partitions in subsequent runs."""
        if self._filesystem.exists(self._normalized_path):
            self._filesystem.rm(self._normalized_path, recursive=True, maxdepth=1)

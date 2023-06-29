""" This module provides kedro datasets that have been customized to suit Pasteur's needs.

The most notable additions are that the datasets lazy load data through `PartitionedDataset`
and can partition save data through custom `Node` return types."""

from .auto import AutoDataset
from .multi import Multiset
from .modified import FragmentedCSVDataset, PatternDataSet, PickleDataSet

""" Backwards compatibility. `AutoDataset` replaces `FragmentedParquetDataset`"""
FragmentedParquetDataset = AutoDataset

__all__ = [
    "AutoDataset",
    "FragmentedCSVDataset",
    "FragmentedParquetDataset",
    "PatternDataSet",
    "PickleDataSet",
    "Multiset",
]

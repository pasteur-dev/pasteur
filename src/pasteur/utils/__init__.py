""" Base utility module for Pasteur.

Pasteur provides a range of utilities in separate modules, based on their functionality."""

from .data import (
    LazyChunk,
    LazyDataset,
    LazyFrame,
    LazyPartition,
    RawSource,
    apply_fun,
    gen_closure,
    get_relative_fn,
    list_unique,
    to_chunked,
)

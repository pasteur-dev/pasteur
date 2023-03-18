""" Base utility module for Pasteur.

Pasteur provides a range of utilities in separate modules, based on their functionality."""

from .data import (
    LazyPartition,
    LazyChunk,
    LazyDataset,
    LazyFrame,
    gen_closure,
    to_chunked,
    apply_fun,
    list_unique,
    get_relative_fn
)

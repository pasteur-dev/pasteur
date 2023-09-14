""" Provides the base definition for Encoder modules"""

from typing import Any, Generic, TypeVar

import pandas as pd

from .metadata import Metadata
from .attribute import Attribute, Attributes
from .module import ModuleClass, ModuleFactory
from .utils import (
    LazyFrame,
    LazyDataset,
    LazyPartition,
    tables_to_data,
    data_to_tables,
    LazyChunk,
)


class AttributeEncoderFactory(ModuleFactory):
    """Factory base class for encoders. Use isinstance with this class
    to filter the Pasteur module list into only containing Encoders."""

    ...


class EncoderFactory(ModuleFactory):
    """Factory base class for encoders. Use isinstance with this class
    to filter the Pasteur module list into only containing Encoders."""

    ...


META = TypeVar("META")


class AttributeEncoder(ModuleClass, Generic[META]):
    """Encapsulates a special way to encode an Attribute.

    One encoder is instantiated per attribute and its `fit` function is called to
    adjust it to the base layer data.

    For partitioned datasets, the `fit` method is called once per partition with
    a different instance of AttributeEncoder, and then `reduce` is called by the
    different instances to perform a reduciton.

    The `data` value may contain a superset of columns than that of the encoder.
    It is up to the encoder to filter it prior to processing. `data` should
    not be mutated.

    If the input data of the synthesis algorithm are in Dataframe form and referencing
    other tables is not required, it is natural to use an `AttributeEncoder` to
    handle encoding per-attribute.

    @Warning: after fitting, the module may be serialized, unserialized, and its encode
    and decode methods may be called arbitrarily from different processes to encode
    and decode sets of columns.
    """

    name: str = ""
    _factory = AttributeEncoderFactory

    def fit(self, attr: Attribute, data: pd.DataFrame | None):
        raise NotImplementedError()

    def reduce(self, other: "AttributeEncoder"):
        pass

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def get_metadata(self) -> dict[str | tuple[str, ...], META]:
        raise NotImplementedError()


class PostprocessEncoder(AttributeEncoder[META], Generic[META]):
    """Same as `AttributeEncoder` but allows customizing the tables after they
    have been encoded or adding additional ones.

    Unlike `AttributeEncoder`, this one does not parallelize per-table, so it should
    be avoided unless customization is required.

    The context tables and their metadata are merged to the parent tables automatically,
    so they are not provided as an argument to `finalize()`.

    Default implementations are provided which behave as the normal AttributeEncoder.
    """

    def finalize(
        self,
        meta: dict[str, dict[tuple[str, ...] | str, META]],
        tables: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        return tables_to_data(ids, tables)

    def undo(
        self,
        meta: dict[str, dict[tuple[str, ...] | str, META]],
        data: dict[str, LazyPartition],
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """Undoes the process of `finalize()`, returns a tuple of `(ids, tables)`."""
        ids, tables = data_to_tables(data)
        return {k: v() for k, v in ids.items()}, {k: v() for k, v in tables.items()}


class Encoder(ModuleClass, Generic[META]):
    name: str = ""
    _factory = EncoderFactory

    def fit(
        self,
        attrs: dict[str, Attributes],
        tables: dict[str, LazyFrame],
        ctx_attrs: dict[str, dict[str, Attributes]],
        ctx: dict[str, dict[str, LazyFrame]],
        ids: dict[str, LazyFrame],
    ):
        raise NotImplementedError()

    def encode(
        self,
        tables: dict[str, LazyFrame],
        ctx: dict[str, dict[str, LazyFrame]],
        ids: dict[str, LazyFrame],
    ) -> dict[str, Any | LazyDataset[Any]]:
        raise NotImplementedError()

    def decode(
        self,
        data: dict[str, LazyDataset[Any]],
    ) -> tuple[
        dict[str, LazyFrame],
        dict[str, dict[str, LazyFrame]],
        dict[str, LazyFrame],
    ]:
        raise NotImplementedError()

    def get_metadata(self) -> META:
        raise NotImplementedError()


ViewEncoder = Encoder

__all__ = ["EncoderFactory", "Encoder", "ViewEncoder", "AttributeEncoder"]

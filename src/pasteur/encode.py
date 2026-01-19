""" Provides the base definition for Encoder modules"""

from typing import Any, Generic, TypeVar, Mapping

import pandas as pd

from .metadata import Metadata
from .attribute import Attribute, Attributes
from .module import ModuleClass, ModuleFactory
from .utils import (
    LazyFrame,
    LazyDataset,
    LazyPartition,
    tables_to_data,
    data_to_tables_ctx,
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
POST_META = TypeVar("POST_META")

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


class PostprocessEncoder(AttributeEncoder[META], Generic[META, POST_META]):
    """Same as `AttributeEncoder` but allows customizing the tables after they
    have been encoded or adding additional ones.

    Unlike `AttributeEncoder`, this one does not parallelize per-table, so it should
    be avoided unless customization is required.
    """

    def finalize(
        self,
        meta: POST_META,
        ids: Mapping[str, pd.DataFrame],
        tables: Mapping[str, pd.DataFrame],
        ctx: Mapping[str, Mapping[str, pd.DataFrame]],
    ) -> Mapping[str, Any]:
        return tables_to_data(ids, tables, ctx)

    def undo(
        self,
        meta: POST_META,
        data: Mapping[str, LazyPartition],
    ) -> tuple[
        Mapping[str, pd.DataFrame],
        Mapping[str, pd.DataFrame],
        Mapping[str, Mapping[str, pd.DataFrame]],
    ]:
        """Undoes the process of `finalize()`, returns a tuple of `(ids, tables)`."""
        ids, tables, ctx = data_to_tables_ctx(data)

        ctx_out = {}
        for creator, ctx_tables in ctx.items():
            for parent, table in ctx_tables.items():
                if creator not in ctx_out:
                    ctx_out[creator] = {}
                ctx_out[creator][parent] = table()

        return (
            {k: v() for k, v in ids.items()},
            {k: v() for k, v in tables.items()},
            ctx_out,
        )

    def get_post_metadata(
        self,
        relationships: dict[str, list[str]],
        attrs: Mapping[tuple[str, ...] | str, META],
        ctx_attrs: Mapping[str, Mapping[tuple[str, ...] | str, META]],
    ) -> POST_META:
        raise NotImplementedError()


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

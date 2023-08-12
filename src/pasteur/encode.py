""" Provides the base definition for Encoder modules"""

from typing import Any, Generic, TypeVar

import pandas as pd

from .attribute import Attribute, Attributes
from .module import ModuleClass, ModuleFactory
from .utils import LazyFrame, LazyDataset

ENC = TypeVar("ENC", bound=ModuleClass)
META = TypeVar("META")


class AttributeEncoderFactory(ModuleFactory[ENC], Generic[ENC]):
    """Factory base class for encoders. Use isinstance with this class
    to filter the Pasteur module list into only containing Encoders."""

    ...


class EncoderFactory(ModuleFactory[ENC], Generic[ENC]):
    """Factory base class for encoders. Use isinstance with this class
    to filter the Pasteur module list into only containing Encoders."""

    ...


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
    _factory = AttributeEncoderFactory["AttributeEncoder"]

    def fit(self, attr: Attribute, data: pd.DataFrame | None):
        raise NotImplementedError()

    def reduce(self, other: "AttributeEncoder"):
        pass

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def get_metadata(self) -> dict[str | tuple[str], META]:
        raise NotImplementedError()


class Encoder(ModuleClass, Generic[META]):
    name: str = ""
    _factory = EncoderFactory["Encoder"]

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

""" Contains the definition for Transformer and ReferenceTransformer modules. """

import logging
import pandas as pd

from .module import ModuleClass, ModuleFactory
from .attribute import Attribute, Attributes

logger = logging.getLogger(__name__)


class TransformerFactory(ModuleFactory):
    ...


class Transformer(ModuleClass):
    _factory = TransformerFactory

    deterministic = True
    "For a given output, the input is the same."
    lossless = True
    "The decoded output equals the input."
    stateful = False
    "Transformer fits variables."

    def __init__(self, **_) -> None:
        pass

    def fit(self, data: pd.Series | pd.DataFrame):
        """Fits to the provided data"""
        pass

    def get_attributes(self) -> Attributes:
        raise NotImplementedError()

    def reduce(self, other: "Transformer"):
        pass

    def fit_transform(self, data: pd.Series | pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def transform(self, data: pd.Series | pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def reverse(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


class RefTransformer(Transformer):
    """Reference Transformers use a reference column as an input to create their embeddings.

    They can be used to integrate constraints (and domain knowledge) into embeddings,
    in such a way that all embeddings produce valid solutions and learning is
    easier.

    For example, consider an end date embedding that references a start date.
    The embedding will form a stable histogram with much less entropy, based
    on the period length.
    In addition, provided that the embedding is forced to be positive, any value
    it takes will produce a valid solution."""

    def fit(
        self,
        data: pd.Series | pd.DataFrame,
        ref: pd.Series | pd.DataFrame | None = None,
    ) -> Attribute | None:
        pass

    def reduce(self, other: "RefTransformer"):
        pass

    def fit_transform(
        self,
        data: pd.Series | pd.DataFrame,
        ref: pd.Series | pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        self.fit(data, ref)
        return self.transform(data, ref)

    def transform(
        self,
        data: pd.Series | pd.DataFrame,
        ref: pd.Series | pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def reverse(
        self, data: pd.DataFrame, ref: pd.Series | pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """When reversing, the data column contains encoded data, whereas the ref
        column contains decoded/original data. Therefore, the referred columns have
        to be decoded first."""
        raise NotImplementedError()


class SeqTransformer(Transformer):
    """Sequence Transformers are a generalised version of Reference Transformers
    that can be used to process event data.

    Sequence Transformers receive unprocessed parent columns, references and the ID table.
    Then, it is up to them to process the data and return the encoded version.
    They can also push columns upstream to parents, through context tables.
    """

    def fit(
        self,
        data: pd.Series | pd.DataFrame,
        ref: dict[str, pd.DataFrame] | None = None,
        ids: pd.DataFrame | None = None,
    ) -> tuple[Attributes, dict[str, Attributes]] | None:
        pass

    def reduce(self, other: "SeqTransformer"):
        pass

    def get_attributes(self) -> tuple[Attributes, dict[str, Attributes]]:
        raise NotImplementedError()

    def fit_transform(
        self,
        data: pd.Series | pd.DataFrame,
        ref: dict[str, pd.DataFrame] | None = None,
        ids: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        self.fit(data, ref)
        return self.transform(data, ref)

    def transform(
        self,
        data: pd.Series | pd.DataFrame,
        ref: dict[str, pd.DataFrame] | None = None,
        ids: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        raise NotImplementedError()

    def reverse(
        self,
        data: pd.DataFrame,
        ctx: dict[str, pd.DataFrame],
        ref: dict[str, pd.DataFrame] | None = None,
        ids: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """When reversing, the data column contains encoded data, whereas the ref
        column contains decoded/original data. Therefore, the referred columns have
        to be decoded first."""
        raise NotImplementedError()

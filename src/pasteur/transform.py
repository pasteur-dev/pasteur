import logging
import pandas as pd

from .module import ModuleClass, ModuleFactory
from .attribute import Attribute

logger = logging.getLogger(__name__)

"""Package with base transformers. 

Contains transformers that convert raw data into 2 types (with name suffix):
    - numerical (num): floating point values (float32), including NaN
    - discrete (idx): integer values (uintX) with metadata that make them:
        - categorical: integer values from 0-N with columns with no relations
        - ordinal: integer values from 0-N where `k` val is closer to `k + 1` than other vals.
        - hierarchical: contains a hierarchy of ordinal and categorical values.
"""


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

    attr: Attribute

    def __init__(self, **_) -> None:
        pass

    def fit(self, data: pd.Series) -> Attribute | None:
        """Fits to the provided data"""
        pass

    def fit_transform(self, data: pd.Series) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def transform(self, data: pd.Series) -> pd.DataFrame:
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
        data: pd.Series,
        ref: pd.Series | None = None,
    ) -> Attribute | None:
        pass

    def fit_transform(
        self, data: pd.Series, ref: pd.Series | None = None
    ) -> pd.DataFrame:
        self.fit(data, ref)
        return self.transform(data, ref)

    def transform(self, data: pd.Series, ref: pd.Series | None = None) -> pd.DataFrame:
        raise NotImplementedError()

    def reverse(self, data: pd.DataFrame, ref: pd.Series | None = None) -> pd.DataFrame:
        """When reversing, the data column contains encoded data, whereas the ref
        column contains decoded/original data. Therefore, the referred columns have
        to be decoded first."""
        raise NotImplementedError()

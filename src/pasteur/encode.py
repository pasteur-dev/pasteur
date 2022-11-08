from typing import Collection

import pandas as pd

from .attribute import Attribute, Attributes
from .metadata import Metadata

"""This package contains encoders that receive input from transformers 
encode it to fit certain models.

One example is a one-hot, numerical encoder that adjusts the data to be
suitable for a regression model.

Model specific transformers have their own hyper-parameters and may be considered
part of the model."""


class Encoder:
    """Encapsulates a special way to encode an Attribute."""

    name: str
    attr: Attribute

    def fit(self, attr: Attribute, data: pd.DataFrame) -> Attribute:
        assert 0, "Unimplemented"

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        assert 0, "Unimplemented"

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        assert 0, "Unimplemented"

import pandas as pd

from .attribute import Attribute
from .module import ModuleClass, ModuleFactory


class EncoderFactory(ModuleFactory["Encoder"]):
    ...

class Encoder(ModuleClass):
    """Encapsulates a special way to encode an Attribute."""

    name: str
    attr: Attribute
    _factory = EncoderFactory

    def fit(self, attr: Attribute, data: pd.DataFrame | None) -> Attribute:
        raise NotImplementedError()

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
import pandas as pd

from .attribute import Attribute
from .module import ModuleClass, ModuleFactory


class EncoderFactory(ModuleFactory["Encoder"]):
    """ Factory base class for encoders. Use isinstance with this class
    to filter the Pasteur module list into only containing Encoders. """
    ...


class Encoder(ModuleClass):
    """Encapsulates a special way to encode an Attribute.
    
    One encoder is instantiated per module and its `fit` function is called to
    fit it to the base layer data.
    
    After that, the module may be serialized, unserialized, and its encode
    and decode methods may be called arbitrarily from different processes to encode
    and decode sets of columns.
    
    The `data` value may contain a superset of columns than that of the encoder.
    It is up to the encoder to filter it prior to processing. `data` should
    not be mutated."""

    name: str
    attr: Attribute
    _factory = EncoderFactory

    def fit(self, attr: Attribute, data: pd.DataFrame | None) -> Attribute:
        raise NotImplementedError()

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def decode(self, enc: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


__all__ = ["EncoderFactory", "Encoder"]

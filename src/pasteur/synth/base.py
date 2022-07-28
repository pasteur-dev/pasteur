from copy import deepcopy
from typing import Dict

import pandas as pd


class Synth:
    name = None
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False

    @staticmethod
    def fit(metadata: Dict, **kwargs: pd.DataFrame):
        assert False, "Not implemented"

    @staticmethod
    def sample(model):
        assert False, "Not implemented"


class IdentSynth(Synth):
    """Returns the data it was provided."""

    name = "ident_idx"
    type = "idx"
    tabular = True
    multimodal = True
    timeseries = True

    @staticmethod
    def fit(metadata: Dict, **kwargs: pd.DataFrame):
        return kwargs

    @staticmethod
    def sample(model):
        return model


class NumIdentSynth(IdentSynth):
    name = "ident_num"
    type = "num"


class BinIdentSynth(IdentSynth):
    name = "ident_bin"
    type = "bin"

import pandas as pd


class Synth:
    name = None
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False

    def fit(self, metadata: dict, data: dict[str, pd.DataFrame]):
        assert False, "Not implemented"

    def sample(self) -> dict[str, pd.DataFrame]:
        assert False, "Not implemented"


def synth_fit_closure(cls):
    def fit(metadata: dict, **kwargs: pd.DataFrame):
        model = cls()
        model.fit(metadata, kwargs)
        return model

    return fit


def synth_sample(model: Synth):
    return model.sample()


class IdentSynth(Synth):
    """Returns the data it was provided."""

    name = "ident_idx"
    type = "idx"
    tabular = True
    multimodal = True
    timeseries = True

    def fit(self, metadata: dict, data: dict[str, pd.DataFrame]):
        self._data = data

    def sample(self):
        return self._data


class NumIdentSynth(IdentSynth):
    name = "ident_num"
    type = "num"


class BinIdentSynth(IdentSynth):
    name = "ident_bin"
    type = "bin"

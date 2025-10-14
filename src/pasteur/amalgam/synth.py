from typing import Any

from pasteur.synth import Synth
from pasteur.utils import LazyDataset


class AmalgamSynth(Synth):
    """Samples the data it was provided."""

    name = "amalgam"
    in_types = ["json", "idx"]
    type = "json"
    partitions = 1

    def preprocess(self, meta: Any, data: dict[str, dict[str, LazyDataset]]):
        pass

    def bake(self, data: dict[str, dict[str, LazyDataset]]):
        pass

    def fit(self, data: dict[str, dict[str, LazyDataset]]):
        self.data = data["json"]

    def sample(self, n: int | None = None):
        return self.data

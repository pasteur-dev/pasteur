from typing import cast

import pandas as pd

from ..hierarchy import rebalance_attributes
from .transformers import IdxTransformer, OrdinalTransformer


class ColumnResampler:
    def __init__(self, col: pd.Series, height: int, ordinal: bool = False) -> None:
        import numpy as np

        name = cast(str, col.name)
        self.name = name
        self.trn = (OrdinalTransformer if ordinal else IdxTransformer)("ukn", nullable=True)
        attr = self.trn.fit(col)

        counts = {name: np.bincount(self.trn.transform(col)[name])}
        attrs = {name: attr}

        self.val = rebalance_attributes(counts, attrs, reshape_domain=False, warn=False)[name][name]  # type: ignore
        self.height = height

    def resample(self, data: pd.DataFrame):
        trn = self.trn.transform(data[self.name])[self.name]
        ds = self.val.downsample(trn.to_numpy(), self.height)
        up = self.val.upsample(ds, self.height, deterministic=True)
        return self.trn.reverse(pd.DataFrame(pd.Series(up, name=self.name))).cat.remove_unused_categories()

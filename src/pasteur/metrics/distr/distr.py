import pandas as pd
import numpy as np

from pandas.api.types import is_integer_dtype


def gen_freq(a: pd.DataFrame, b: pd.DataFrame, cols: list[str], fillna=1e-6):
    a, b = a.value_counts(), b.value_counts()
    c = pd.concat([a, b], axis=1)
    c = c / c.sum()
    c = c.fillna(value=fillna)
    c[c == 0] = fillna
    c = c / c.sum()
    return c.iloc[:, 0], c.iloc[:, 1]

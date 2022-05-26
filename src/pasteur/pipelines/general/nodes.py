"""
This file contains nodes for general pre-processing of data.
"""
from hashlib import new
import logging
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from math import floor


def identity(data):
    """Returns the argument that was submitted."""
    return data


def split_keys(
    keys: pd.DataFrame, params: Dict[str, Any], random_state: Optional[int] = None
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Splits keys to work (wrk), reference (ref), test (dev), and validation (val) sets.

    Args:
        data: A Dataframe with 1 column or index with keys
    Returns:
        Dataframe series for each of the keys
    """

    # Sort to ensure consistent split every time
    # Dataframe should consist of up to 1 column (which is the key) or an index
    if keys.keys().empty:
        keys = keys.sort_values(by=keys.index.name)
    else:
        assert False, "Keys df should only have an index (0 columns)"
        # keys = keys.sort_values(by=keys.keys()[0])

    r_wrk = params["wrk"]
    r_ref = params["ref"]
    r_dev = params["dev"]
    r_val = params["val"]
    assert r_wrk + r_ref + r_dev + r_val <= 1, "Dataset ratios exceed 100%"

    n_all = len(keys)
    n_wrk = floor(r_wrk * n_all)
    n_ref = floor(r_ref * n_all)
    n_dev = floor(r_dev * n_all)
    n_val = floor(r_val * n_all)
    assert n_wrk + n_ref + n_dev + n_val <= n_all, "Sizes exceed dataset size"

    wrk = keys.sample(n=n_wrk, random_state=random_state)
    keys = keys.drop(wrk.index)
    ref = keys.sample(n=n_ref, random_state=random_state)
    keys = keys.drop(ref.index)
    dev = keys.sample(n=n_dev, random_state=random_state)
    keys = keys.drop(dev.index)
    val = keys.sample(n=n_val, random_state=random_state)

    return wrk, ref, dev, val


def filter_by_keys(*args):
    assert len(args) > 1, "Keys/Data omitted/not provided"

    keys: pd.DataFrame = args[-1]
    tables: List[pd.DataFrame] = args[:-1]

    # Sort to ensure consistent split every time
    # Dataframe should consist of up to 1 column (which is the key) or an index
    if keys.keys().empty:
        col = keys.index.name
    else:
        assert False, "Keys df should only have an index (0 columns)"
        # assert len.keys() == 1, "Keys df should only have one column"
        # col = keys.keys()[0]

    outputs = []
    for table in tables:
        new_table = table.join(keys, on=col)
        outputs.append(new_table)

    return outputs if len(outputs) > 1 else outputs[0]

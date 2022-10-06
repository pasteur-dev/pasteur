import pandas as pd
import numpy as np

from math import floor


def split_keys(
    keys: pd.DataFrame, split: dict[str, any], random_state: int | None = None
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Splits keys according to the split dictionary.

    Example: split = {"dev": 0.3, "wrk": 0.3}
    Returns {"dev": 0 col Dataframe, "wrk" 0 col Dataframe}
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Sort to ensure consistent split every time
    # Dataframe should consist of one column that is the index
    if keys.keys().empty:
        # If DataFrame is empty assume index is key
        assert keys.index.name, "No index column available"
        idx_name = None
        idx = keys.index
    elif keys.index.name:
        # If index has a name, assume it is the key and drop other columns
        idx_name = keys.index.name
        idx = keys.index
    else:
        # Otherwise, pick first column as index and drop the others
        idx_name = keys.columns[0]
        idx = keys[idx_name]

    assert sum(split.values()) <= 1, "Dataset ratios exceed 100%"

    n_all = len(keys)
    ns = {name: floor(ratio * n_all) for name, ratio in split.items()}
    assert sum(ns.values()) <= n_all, "Sizes exceed dataset size"

    # Sort and shuffle array for a consistent split every time
    keys = np.sort(idx)
    np.random.shuffle(keys)

    # Split array into the required chunks
    splits = {}
    i = 0
    for name, n in ns.items():
        split_keys = keys[i : i + n]
        i += n
        splits[name] = pd.DataFrame(index=split_keys)
        if idx_name is not None:
            splits[name].index.name = idx_name

    return splits

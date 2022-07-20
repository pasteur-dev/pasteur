import pandas as pd
import numpy as np


def test_bin_transformer():
    from pasteur.transform import BinTransformer

    test_data = [1, 2, 5, 23, 643, 122, 324, 542, 543]
    test_data = pd.DataFrame(
        np.transpose(np.array([test_data, test_data])), columns=["a", "b"]
    )

    bins = 20

    t = BinTransformer(bins)

    t.fit(test_data)

    enc = t.transform(test_data)
    assert all(enc["a"] == np.array([0, 0, 0, 0, 20, 3, 10, 16, 16]))

    dec = t.reverse(enc)
    assert all(
        dec["a"]
        == np.array([1.0, 1.0, 1.0, 1.0, 643.0, 97.30000000000001, 322.0, 514.6, 514.6])
    )


def test_onehot_transformer():
    from pasteur.transform import OneHotTransformer

    test_data = pd.DataFrame()
    test_data["floats"] = [1, 2, 5, 23.1, 643, 122, 10, 542, 543]
    test_data["ints"] = [1, 2, 5, 23, 643, 122, 10, 542, 543]
    test_data["objects"] = [1, 2, 5, 23, "sfas", 122, None, 542, 543]

    t = OneHotTransformer(-1)

    t.fit(test_data[:-1])

    enc = t.transform(test_data)

    dec = t.reverse(enc)

    assert all(dec["floats"] == [1.0, 2.0, 5.0, 23.1, 643.0, 122.0, 10.0, 542.0, -1.0])
    assert all(dec["ints"] == [1, 2, 5, 23, 643, 122, 10, 542, -1])
    assert all(
        (dec["objects"] == [1, 2, 5, 23, "sfas", 122, None, 542, -1])
        | pd.isna(dec["objects"])
    )

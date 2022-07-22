import pandas as pd
import numpy as np

from pasteur.transform import Transformer


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


def test_idx_transformer():
    from pasteur.transform import IdxTransformer

    test_data = pd.DataFrame()
    test_data["floats"] = [1, 2, 5, 23.1, 643, 122, 10, 542, 543]
    test_data["ints"] = [1, 2, 5, 23, 643, 122, 10, 542, 543]
    test_data["objects"] = [1, 2, 5, 23, "sfas", 122, None, 542, 543]

    t = IdxTransformer(-1)

    t.fit(test_data[:-1])

    enc = t.transform(test_data)

    dec = t.reverse(enc)

    assert all(dec["floats"] == [1.0, 2.0, 5.0, 23.1, 643.0, 122.0, 10.0, 542.0, -1.0])
    assert all(dec["ints"] == [1, 2, 5, 23, 643, 122, 10, 542, -1])
    assert all(
        (dec["objects"] == [1, 2, 5, 23, "sfas", 122, None, 542, -1])
        | pd.isna(dec["objects"])
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


def test_gray_transformer():
    from pasteur.transform import GrayTransformer

    test_data = pd.DataFrame()
    test_data["tst1"] = [1, 2, 5, 2, 3, 4, 9, 10]
    test_data["tst2"] = [1, 2, 5, 2, 3, 4, 9, 10]

    t = GrayTransformer()

    t.fit(test_data)

    enc = t.transform(test_data)

    dec = t.reverse(enc)
    assert np.all(dec == test_data)


def test_basen_transformer():
    from pasteur.transform import BaseNTransformer

    test_data = pd.DataFrame()
    test_data["tst1"] = [1, 2, 5, 2, 3, 4, 9, 10]
    test_data["tst2"] = [1, 2, 5, 2, 3, 4, 9, 10]

    for b in range(2, 6):
        t = BaseNTransformer(b)

        t.fit(test_data)
        enc = t.transform(test_data)
        dec = t.reverse(enc)

        assert t.out_type == ("bin" if b == 2 else f"b{b}")
        assert np.all(dec == test_data)


def test_norm_transformer():
    from pasteur.transform import NormalizeTransformer

    test_data = pd.DataFrame()
    test_data["tst1"] = [1, 2, 5, 2, 3, 4, 9, 10]
    test_data["tst2"] = [1, 2, 5, 2, 3, 4, 9, 10]

    t = NormalizeTransformer()

    t.fit(test_data[:-1])
    enc = t.transform(test_data)
    dec = t.reverse(enc)

    assert np.all(dec[:-1] == test_data[:-1])
    assert all(dec[-1:] == np.max(test_data[:-1]))


def test_norm_dist_transformer():
    from pasteur.transform import NormalDistTransformer

    test_data = pd.DataFrame()
    test_data["tst1"] = [1, 2, 5, 2, 3, 4, 9, 10]
    test_data["tst2"] = [1, 2, 5, 2, 3, 4, 9, 10]

    t = NormalDistTransformer()

    t.fit(test_data[:-1])
    enc = t.transform(test_data)
    dec = t.reverse(enc)

    assert np.all(dec == test_data)


def test_chain_transformer():
    from pasteur.transform import (
        ChainTransformer,
        NormalDistTransformer,
        BinTransformer,
        GrayTransformer,
    )

    test_data = pd.DataFrame()
    test_data["tst1"] = [1, 2, 5, 2, 3, 4, 9, 10]
    test_data["tst2"] = [1, 2, 5, 2, 3, 4, 9, 10]

    transformers = [NormalDistTransformer(), BinTransformer(8), GrayTransformer()]

    t = ChainTransformer(transformers)

    t.fit(test_data[:-1])
    enc = t.transform(test_data)
    dec = t.reverse(enc)

    assert np.all(dec == np.expand_dims(np.array([1, 2, 5, 2, 3, 3, 9, 9]), axis=1))


def test_chain_transformer_na():
    from pasteur.transform import (
        ChainTransformer,
        NormalDistTransformer,
        BinTransformer,
        GrayTransformer,
    )

    test_data = pd.DataFrame()
    test_data["tst1"] = [1, 2, pd.NA, 2, 3, 4, pd.NA, 7, 10]

    transformers = [NormalDistTransformer(), BinTransformer(8), GrayTransformer()]

    t = ChainTransformer(transformers, nullable=True, na_val=0)

    t.fit(test_data)
    enc = t.transform(test_data)
    dec = t.reverse(enc)

    assert np.all(
        (dec["tst1"] == [1, 1, np.NAN, 1, 2, 3, np.NAN, 6, 10]) | pd.isna(dec["tst1"])
    )

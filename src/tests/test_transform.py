import pandas as pd
import numpy as np


def test_discrete_transformer():
    from pasteur.transform import DiscretizationTransformer

    test_data = [1, 2, 5, 23, 643, 122, 324, 542, 543]
    test_data = pd.DataFrame(
        np.transpose(np.array([test_data, test_data])), columns=["a", "b"]
    )

    bins = 20

    t = DiscretizationTransformer(bins)

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


def test_bin_transformer():
    from pasteur.transform import BinTransformer

    test_data = pd.DataFrame()
    test_data["tst1"] = [1, 2, 5, 2, 3, 4, 9, 10]
    test_data["tst2"] = [1, 2, 5, 2, 3, 4, 9, 10]

    t = BinTransformer()

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
        NormalizeTransformer,
        DiscretizationTransformer,
        GrayTransformer,
    )

    test_data = pd.DataFrame()
    test_data["tst1"] = [1, 2, 5, 2, 3, 4, 9, 10]
    test_data["tst2"] = [1, 2, 5, 2, 3, 4, 9, 10]

    transformers = [
        NormalizeTransformer(),
        DiscretizationTransformer(8),
        GrayTransformer(),
    ]

    t = ChainTransformer(transformers)

    t.fit(test_data[:-1])
    enc = t.transform(test_data)
    dec = t.reverse(enc)

    assert np.all(dec == np.expand_dims(np.array([1, 2, 5, 2, 3, 4, 8, 8]), axis=1))


def test_chain_transformer_na():
    from pasteur.transform import (
        ChainTransformer,
        NormalizeTransformer,
        DiscretizationTransformer,
        GrayTransformer,
    )

    test_data = pd.DataFrame()
    test_data["tst1"] = [1, 2, pd.NA, 2, 3, 4, pd.NA, 7, 10]

    transformers = [
        NormalizeTransformer(),
        DiscretizationTransformer(8),
        GrayTransformer(),
    ]

    t = ChainTransformer(transformers, nullable=True, na_val=0)

    t.fit(test_data)
    enc = t.transform(test_data)
    dec = t.reverse(enc)

    assert np.all(
        (dec.sort_index()["tst1"] == [1, 1, np.NAN, 1, 2, 3, np.NAN, 6, 8])
        | pd.isna(dec["tst1"])
    )


def test_date_transform():
    from pasteur.transform import DateTransformer

    test_data = pd.DataFrame()

    def add_date(col, times):
        test_data[col] = pd.to_datetime(times, unit="s")

    # https://catonmat.net/tools/generate-random-unix-timestamps
    add_date("year_start", [1040950472, 1061321383, 982846203, 1158533834, 1153720128])
    add_date("year_end", [1570024572, 1337927206, 1432776607, 1563197834, 1518650917])
    add_date("month_start", [966805568, 971024885, 960237288, 976287141, 970894107])
    add_date("month_end", [987563531, 1008241424, 989683269, 984544020, 978397836])
    add_date("week_start", [951123803, 947832472, 952228661, 949995448, 948836756])
    add_date("week_end", [959113173, 955559677, 956566230, 956010344, 962402695])
    add_date("day_start", [946819605, 947119107, 948216451, 947159886, 948980663])
    add_date("day_end", [951132021, 950053040, 950753161, 950714127, 949812216])

    for span in ["year", "week", "day"]:
        for use_ref in [False, True]:
            ref = test_data[f"{span}_start"] if use_ref else None
            vals = test_data[[f"{span}_end"]]
            t = DateTransformer(span)
            t.fit(vals, ref)
            enc = t.transform(vals, ref)
            dec = t.reverse(enc, ref)

            assert np.all(dec[f"{span}_end"].dt.year == vals[f"{span}_end"].dt.year)
            assert np.all(dec[f"{span}_end"].dt.month == vals[f"{span}_end"].dt.month)
            assert np.all(dec[f"{span}_end"].dt.day == vals[f"{span}_end"].dt.day)


def test_time_transform():
    from pasteur.transform import TimeTransformer

    test_data = pd.DataFrame()

    def add_date(col, times):
        test_data[col] = pd.to_datetime(times, unit="s")

    # https://catonmat.net/tools/generate-random-unix-timestamps
    add_date(
        "times",
        [
            1040950472,
            1061321383,
            982846203,
            1158533834,
            1153720128,
            1570024572,
            1337927206,
            1432776607,
            1563197834,
            1518650917,
            966805568,
            971024885,
            960237288,
            976287141,
            970894107,
            987563531,
            1008241424,
            989683269,
            984544020,
            978397836,
            951123803,
            947832472,
            952228661,
        ],
    )

    for span in ("hour", "halfhour", "minute", "halfminute", "second"):
        t = TimeTransformer(span)
        t.fit(test_data)
        enc = t.transform(test_data)
        dec = t.reverse(enc)

        assert np.all(dec["times"].dt.hour == test_data["times"].dt.hour)
        if span == "halfhour":
            assert np.all(
                dec["times"].dt.minute == 30 * (test_data["times"].dt.minute > 29)
            )
        if span in ("minute", "halfminute", "second"):
            assert np.all(dec["times"].dt.minute == test_data["times"].dt.minute)
        if span == "halfminute":
            assert np.all(
                dec["times"].dt.second == 30 * (test_data["times"].dt.second > 29)
            )
        if span == "second":
            assert np.all(dec["times"].dt.second == test_data["times"].dt.second)


def test_datetime_transform():
    from pasteur.transform import DatetimeTransformer

    test_data = pd.DataFrame()

    def add_date(col, times):
        test_data[col] = pd.to_datetime(times, unit="s")

    # https://catonmat.net/tools/generate-random-unix-timestamps
    add_date(
        "times",
        [
            1040950472,
            1061321383,
            982846203,
            1158533834,
            1153720128,
            1570024572,
            1337927206,
            1432776607,
            1563197834,
            1518650917,
            966805568,
            971024885,
            960237288,
            976287141,
            970894107,
            987563531,
            1008241424,
            989683269,
            984544020,
            978397836,
            951123803,
            947832472,
            952228661,
        ],
    )

    t = DatetimeTransformer()
    t.fit(test_data)
    enc = t.transform(test_data)
    dec = t.reverse(enc)

    assert np.all(dec["times"].dt.year == test_data["times"].dt.year)
    assert np.all(dec["times"].dt.month == test_data["times"].dt.month)
    assert np.all(dec["times"].dt.day == test_data["times"].dt.day)
    assert np.all(dec["times"].dt.hour == test_data["times"].dt.hour)
    assert np.all(dec["times"].dt.minute == 30 * (test_data["times"].dt.minute > 29))


def test_fixed_value_transform():
    from pasteur.transform import FixedValueTransformer

    arr = list(range(20))
    test_data = pd.DataFrame(np.array([arr, arr]).T, columns=["tst1", "tst2"])

    t = FixedValueTransformer("date")
    t.fit(test_data)
    enc = t.transform(test_data)
    dec = t.reverse(enc)

    assert dec.shape == (20, 2)
    assert dec.loc[0, "tst1"].year == 2000

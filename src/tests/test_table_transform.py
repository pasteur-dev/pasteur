import pandas as pd
import numpy as np

data_dict = {
    "users": [
        [3, 3, 4, 5, "asdf", 3.5],
        [3, 3, 4, 5, "23", 3.5],
        [1, 3, 4, 5, "asdf", 3.5],
        [3, 3, 4, 5, "asdf", 3.5],
        [4, 3, 4, 5, "dsfd", 3.5],
        [3, 3, 4, 5, "hjk", 3.5],
    ],
    "sessions": [
        [2, 0, 4, 5, "asdf", 3.5],
        [3, 3, 4, 5, "3", 2.5],
        [3, 1, 4, 5, "23", 3.3],
        [1, 4, 4, 5, "543", 1.4],
        [3, 2, 4, 5, "dsbv", 2.5],
        [3, 2, 4, 5, "jlk", 4.5],
    ],
    "visits": [
        [3, 3, 0, 2, "asdf", 3.5],
        [2, 3, 0, 3, "23", 3.5],
        [3, 3, 1, 4, "asdf", 3.5],
        [3, 3, 0, 7, "asdf", 3.5],
        [4, 3, 1, 0, "dsfd", 3.5],
        [3, 3, 1, 5, "hjk", 3.5],
    ],
}

columns_dict = {
    "users": ["c1", "c2", "c3", "n1", "c4", "n2"],
    "sessions": ["c1", "user_id", "c2", "n1", "c4", "n2"],
    "visits": ["c1", "tst_id", "session_id", "n1", "c3", "n2"],
}

meta_dict = {
    "tables": {
        "users": {
            "primary_key": "user_id",
            "fields": {
                "user_id": "id",
                "c1": "categorical",
                "c2": "categorical",
                "c3": {"type": "categorical", "ref": "c2"},
                "n1": "numerical",
                "c4": "categorical",
                "n2": "numerical",
            },
        },
        "sessions": {
            "primary_key": "session_id",
            "fields": {
                "session_id": "id",
                "c1": "categorical",
                "user_id": {"type": "id", "ref": "users"},
                "c2": "categorical",
                "n1": "numerical",
                "c4": "categorical",
                "n2": "numerical",
            },
        },
        "visits": {
            "primary_key": "visit_id",
            "fields": {
                "visit_id": "id",
                "c1": "categorical",
                "tst_id": "id",
                "session_id": {"type": "id", "ref": "sessions"},
                "n1": "numerical",
                "c3": {"type": "categorical", "ref": "users.c3"},
                "n2": "numerical",
            },
        },
    }
}


def test_table_transform():
    from pasteur.metadata import Metadata
    from pasteur.transform import TableTransformer

    data = {n: pd.DataFrame(t, columns=columns_dict[n]) for n, t in data_dict.items()}
    data["users"].index.name = "user_id"
    data["sessions"].index.name = "session_id"
    data["visits"].index.name = "visit_id"

    meta = Metadata(meta_dict, data)

    t_t = []
    t_enc = []
    t_ids = []
    for name in data:
        t = TableTransformer(meta, name, ("idx", "bin"))
        ids = t.find_ids(data)
        t.fit(data, ids)
        enc = t["idx"].transform(data)

        t_t.append(t)
        t_enc.append(enc)
        t_ids.append(ids)

    t_dec = {}
    for name, enc, ids, t in zip(data.keys(), t_enc, t_ids, t_t):
        dec = t["idx"].reverse(enc, ids, t_dec)
        t_dec[name] = dec

    assert np.all(dec["tst_id"] == 0)
    assert np.all(t_dec["sessions"]["c4"] == data["sessions"]["c4"])

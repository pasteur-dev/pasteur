meta_dict = {
    "tables": {
        "table": {
            "primary_key": "id",
            "targets": ["education"],
            "sensitive": ["race", "relationship"],
            "fields": {
                "age": "numerical",
                "workclass": {"type": "categorical", "metrics": {"y_log": True}},
                "fnlwgt": "numerical",
                "education": "categorical",
                "education-num": "numerical",
                "marital-status": "categorical",
                "occupation": "categorical",
                "relationship": "categorical",
                "race": "categorical",
                "sex": "categorical",
                "capital-gain": {"type": "numerical", "metrics": {"y_log": True}},
                "capital-loss": {"type": "numerical", "metrics": {"y_log": True}},
                "hours-per-week": "numerical",
                "native-country": {"type": "categorical", "metrics": {"y_log": True}},
            },
        }
    }
}

def test_metadata():
    from pasteur.metadata import Metadata

    meta = Metadata(meta_dict)
    pass
import pandas as pd


def set_index_name_closure(name):
    def set_index_name(df: pd.DataFrame):
        df = df.copy()
        df.index.name = name
        return df

    return set_index_name

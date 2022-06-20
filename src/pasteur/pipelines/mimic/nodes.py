import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime


def mk_dates_relative(
    source_table: pd.DataFrame,
    source_idx_col: str,
    date_table: pd.DataFrame,
    date_col: str,
    date_format: str = None,
):
    rel_table = source_table.copy(deep=False)
    dates = date_table[date_col]

    # Some date indexes can be null, in this case drop the rows that contain them
    # Using a relative date would make no sense for them
    rel_table = rel_table.dropna(subset=[source_idx_col])

    # Some target columns are not dates by default
    # ex. patient table anchor_year is not a date
    if date_format:
        dates = pd.to_datetime(dates, format=date_format)
    else:
        assert is_datetime(
            dates
        ), f"Date col {date_col} is not a datetime and no date_format was provided"

    # Create column that matches source_table
    synced_dates = dates[rel_table[source_idx_col]]
    synced_dates.index = rel_table.index

    for col in source_table.keys():
        if not is_datetime(source_table[col]):
            continue

        rel_table[col] = source_table[col] - synced_dates

    return rel_table


def core_mk_dates_relative(source_table: pd.DataFrame, patients_table: pd.DataFrame):
    return mk_dates_relative(
        source_table, "subject_id", patients_table, "anchor_year", "%Y"
    )


def hosp_mk_dates_relative(source_table: pd.DataFrame, admissions_table: pd.DataFrame):
    return mk_dates_relative(
        source_table, "hadm_id", admissions_table, "admittime"
    )

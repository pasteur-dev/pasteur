import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime


def mk_dates_relative(
    source_table: pd.DataFrame,
    source_idx_col: str,
    dates: pd.Series,
    date_format: str = None,
):
    rel_table = source_table.copy(deep=False)

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
        ), f"Date col is not a datetime and no date_format was provided"

    # Create column that matches source_table
    synced_dates = dates[rel_table[source_idx_col]]
    synced_dates.index = rel_table.index

    for col in source_table.keys():
        if not is_datetime(source_table[col]):
            continue

        rel_table[col] = source_table[col] - synced_dates

    return rel_table


def tab_join_tables(patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    return admissions.join(patients, on="subject_id")


def mm_core_transform_tables(
    patients: pd.DataFrame, admissions: pd.DataFrame, transfers: pd.DataFrame
):
    # # Calculate rel patient date
    birth_year = patients["anchor_year"] - patients["anchor_age"]
    birth_year_date = pd.to_datetime(birth_year, format="%Y")

    patients_new = patients.drop(labels=["anchor_year", "anchor_age"], axis=1)
    patients_new.rename(columns={"anchor_year_group": "year_group"}, inplace=True)
    patients_new["birth_year"] = birth_year_date

    # patients_new["death_age"] = patients["dod"].dt.year.astype("Int64") - birth_year
    # patients_new["death_date"] = patients["dod"] - birth_year_date

    # admissions_new = mk_dates_relative(admissions, "subject_id", birth_year_date)
    # transfers_new = mk_dates_relative(transfers, "hadm_id", admissions["admittime"])

    # return patients_new, admissions_new, transfers_new

    return patients_new, admissions, transfers

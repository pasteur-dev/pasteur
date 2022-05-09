import pandas as pd


def tab_join_tables(patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    return admissions.join(patients, on="subject_id")


def mm_core_transform_tables(
    patients: pd.DataFrame, admissions: pd.DataFrame, transfers: pd.DataFrame
):
    return patients, admissions, transfers

import pandas as pd

from ...view import TabularView, View
from ..datasets import mimic as _

def tab_join_tables(patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    # # Calculate rel patient date
    birth_year = patients["anchor_year"] - patients["anchor_age"]
    birth_year_date = pd.to_datetime(birth_year, format="%Y")

    patients_new = patients.drop(labels=["anchor_year", "anchor_age"], axis=1)
    patients_new.rename(columns={"anchor_year_group": "year_group"}, inplace=True)
    patients_new["birth_year"] = birth_year_date

    return admissions.join(patients_new, on="subject_id")


def mm_core_transform_patients(patients: pd.DataFrame):
    # # Calculate rel patient date
    birth_year = patients["anchor_year"] - patients["anchor_age"]
    birth_year_date = pd.to_datetime(birth_year, format="%Y")

    patients_new = patients.drop(labels=["anchor_year", "anchor_age"], axis=1)
    patients_new.rename(columns={"anchor_year_group": "year_group"}, inplace=True)
    patients_new["birth_year"] = birth_year_date

    return patients_new


class MimicMmCoreView(View):
    """The mimic core tables, slightly post processed."""

    name = "mimic_mm_core"
    dataset = "mimic"
    deps: dict[str, list[str]] = {
        "patients": ["core_patients"],
        "admissions": ["core_admissions"],
        "transfers": ["core_transfers"],
    }
    trn_deps = {
        "admissions": ["patients"],
        "transfers": ["admissions"],
    }

    def ingest(self, name, **tables: pd.DataFrame):
        match name:
            case "patients":
                return mm_core_transform_patients(tables["core_patients"])
            case "admissions":
                return tables["core_admissions"]
            case "transfers":
                return tables["core_transfers"]
            case other:
                assert False, f"Table {other} not part of view {self.name}"


class MimicTabAdmissions(TabularView):
    """The mimic core admissions table, merged with the patients table."""

    name = "mimic_tab_admissions"
    dataset = "mimic"
    deps = {
        "table": ["core_patients", "core_admissions"],
    }

    def ingest(self, name, **tables: pd.DataFrame):
        assert name == "table"
        return tab_join_tables(tables["core_patients"], tables["core_admissions"])

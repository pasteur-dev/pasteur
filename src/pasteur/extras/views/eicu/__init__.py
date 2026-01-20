from __future__ import annotations

import pandas as pd

from ....utils import LazyChunk, get_relative_fn, to_chunked
from ....view import View


def _limit_per_patient(
    table: pd.DataFrame,
    key: str,
    max_rows: int,
    order_col: str | None = None,
) -> pd.DataFrame:
    if key not in table.columns:
        return table

    if order_col and order_col in table.columns:
        table = table.sort_values([key, order_col])

    return table.groupby(key, sort=False).head(max_rows)


class EicuRelational(View):
    """Relational eICU view with patient as parent table."""

    name = "eicu_r1"
    dataset = "eicu"
    deps = {
        "patient": ["patient"],
        "admissiondx": ["admissiondx"],
        "lab": ["lab"],
        "vitalaperiodic": ["vitalaperiodic"],
        "medication": ["medication"],
    }
    trn_deps = {
        "admissiondx": ["patient"],
        "lab": ["patient"],
        "vitalaperiodic": ["patient"],
        "medication": ["patient"],
    }
    parameters = get_relative_fn("parameters_relational.yml")

    @to_chunked
    def ingest(self, name, **tables: LazyChunk):
        match name:
            case "patient":
                return tables["patient"](
                    [
                        "gender",
                        "age",
                        "ethnicity",
                        "hospitalid",
                        "hospitaladmitsource",
                        "hospitaldischargestatus",
                        "unittype",
                        "unitstaytype",
                        "unitvisitnumber",
                        "unitdischargestatus",
                        "admissionheight",
                        "admissionweight",
                        "dischargeweight",
                    ]
                )
            case "admissiondx":
                return tables["admissiondx"](
                    [
                        "patientunitstayid",
                        "admitdxpath",
                        "admitdxname",
                    ]
                )
            case "diagnosis":
                return tables["diagnosis"](
                    [
                        "patientunitstayid",
                        "diagnosisstring",
                        "diagnosispriority",
                    ]
                )
            case "lab":
                return tables["lab"](
                    [
                        "patientunitstayid",
                        "labname",
                        "labresult",
                    ]
                )
            case "vitalaperiodic":
                return tables["vitalaperiodic"](
                    [
                        "patientunitstayid",
                        "noninvasivesystolic",
                        "noninvasivediastolic",
                        "noninvasivemean",
                    ]
                )
            case "medication":
                return tables["medication"](
                    [
                        "patientunitstayid",
                        "drugname",
                        "dosage",
                        "routeadmin",
                        "frequency",
                    ]
                )
            case "treatment":
                return tables["treatment"](
                    [
                        "patientunitstayid",
                        "treatmentstring",
                        "activeupondischarge",
                    ]
                )
            case other:
                assert False, f"Table {other} not part of view {self.name}"

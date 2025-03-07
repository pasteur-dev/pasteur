from functools import partial

import pandas as pd

from ....utils import LazyChunk, LazyFrame, get_relative_fn, to_chunked
from ....view import TabularView, View, filter_by_keys, filter_by_keys_merged


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


class MimicCore(View):
    """The mimic core tables, slightly post processed."""

    name = "mimic_core"
    dataset = "mimic"
    deps: dict[str, list[str]] = {
        "patients": ["patients"],
        "admissions": ["admissions"],
        "transfers": ["transfers"],
    }
    trn_deps = {
        "admissions": ["patients"],
        "transfers": ["admissions"],
    }
    parameters = get_relative_fn("parameters_core.yml")

    @to_chunked
    def ingest(self, name, **tables: LazyChunk):
        match name:
            case "patients":
                return mm_core_transform_patients(tables["patients"]())
            case "admissions":
                return tables["admissions"]()
            case "transfers":
                return tables["transfers"]().dropna(subset=["hadm_id"])
            case other:
                assert False, f"Table {other} not part of view {self.name}"


class MimicTabAdmissions(TabularView):
    """The mimic core admissions table, merged with the patients table."""

    name = "mimic_tab_admissions"
    dataset = "mimic"
    deps = {
        "table": ["patients", "admissions"],
    }
    parameters = get_relative_fn("parameters_tab.yml")

    @to_chunked
    def ingest(self, name, **tables: LazyChunk):
        assert name == "table"
        return tab_join_tables(tables["patients"](), tables["admissions"]())


def _ingest_chunk(
    part_id,
    duplicates: int,
    partitions: int,
    core_patients: LazyChunk,
    icu_icustays: LazyChunk,
    icu_chartevents: LazyChunk,
):
    patients = core_patients()
    stays = icu_icustays()
    events = icu_chartevents()

    birth_year = patients["anchor_year"] - patients["anchor_age"]
    birth_year_date = pd.to_datetime(birth_year, format="%Y")

    # Join tables
    table = (
        events.drop(columns=["itemid", "storetime", "value"])
        .join(
            stays.drop(columns=["subject_id", "hadm_id", "los"]),
            how="inner",
            on="stay_id",
        )
        .join(
            patients.drop(
                columns=["anchor_age", "anchor_year", "anchor_year_group", "dod"]
            ).assign(birth_year=birth_year_date),
            how="inner",
            on="subject_id",
        )
        .drop(columns=["hadm_id", "stay_id"])
    )

    # Change to rangeindex (saves space) and shuffle table (helps marginal calc.)
    table = (
        table.reset_index(drop=True).rename_axis("id").sample(frac=1, random_state=53)
    )

    n = (table.shape[0] - 1) // partitions + 1

    out = {}
    for i in range(partitions):
        for j in range(duplicates):
            out[f"{part_id}_{i}_{j}"] = table[i * n : min((i + 1) * n, len(table))]

    return out


def _ingest_chunk_icu(
    part_id,
    duplicates: int,
    partitions: int,
    icu_chartevents: LazyChunk,
):
    events = icu_chartevents()

    # Join tables
    table = (
        events.drop(columns=["itemid", "storetime", "value", "hadm_id"])
    )

    # Change to rangeindex (saves space) and shuffle table (helps marginal calc.)
    table = (
        table.reset_index(drop=True).rename_axis("chart_id").sample(frac=1, random_state=53)
    )

    n = (table.shape[0] - 1) // partitions + 1

    out = {}
    for i in range(partitions):
        for j in range(duplicates):
            out[f"{part_id}_{i}_{j}"] = table[i * n : min((i + 1) * n, len(table))]

    return out

def _remove_empty_partitions(func):
    """Purges empty partitions returned by func"""
    res = func()
    if not isinstance(res, dict):
        return res

    out = {}
    for name, part in res.items():
        if len(part):
            out[name] = part
    return out


class MimicBillion(TabularView):
    """The mimic icu chart events, with additional columns from patients, icu stays."""

    name = "mimic_billion"
    dataset = "mimic"
    deps = {
        "table": ["patients", "icu_icustays", "icu_chartevents"],
    }
    parameters = get_relative_fn("parameters_billion.yml")

    def ingest(
        self,
        name,
        patients: LazyFrame,
        icu_icustays: LazyFrame,
        icu_chartevents: LazyFrame,
    ):
        assert name == "table"

        # from ...utils import ColumnResampler
        # ColumnResampler(icu_chartevents.sample(), )

        funs = set()
        for part_id, (_patients, stays, events) in LazyFrame.zip(
            patients, icu_icustays, icu_chartevents
        ).items():
            funs.add(partial(_ingest_chunk, part_id, 3, 4, _patients, stays, events))
        return funs

    def filter_table(self, name: str, keys: LazyFrame, **tables: LazyFrame):
        n = keys.shape[0]

        if n < 2000:
            return filter_by_keys_merged(
                keys, tables[name], reset_index=True, drop_index=True
            )

        funcs = filter_by_keys(keys, tables[name], drop_index=True)
        return {partial(_remove_empty_partitions, func) for func in funcs}  # type: ignore


class MimicIcu(View):
    """The mimic icu chart events, with additional columns from patients, icu stays."""

    name = "mimic_icu"
    dataset = "mimic"
    deps = {
        "patients": ["patients"],
        "stays": ["icu_icustays"],
        "events": ["icu_chartevents"],
    }
    trn_deps = {
        "stays": ["patients"],
        "events": ["stays"],
    }
    parameters = get_relative_fn("parameters_icu.yml")

    def ingest(self, name, **tables: LazyFrame):
        match name:
            case "patients":
                return mm_core_transform_patients(tables["patients"]())
            case "stays":
                return tables["icu_icustays"]().dropna(subset=["hadm_id"])
            case "events":
                funs = set()
                for part_id, events in tables["icu_chartevents"].items():
                    funs.add(partial(_ingest_chunk_icu, part_id, 1, 4, events))
                return funs

        assert False, f"Table {name} not part of view {self.name}"

    def filter_table(self, name: str, keys: LazyFrame, **tables: LazyFrame):
        n = keys.shape[0]

        if n < 2000:
            return filter_by_keys_merged(
                keys, tables[name], reset_index=True, drop_index=name == "events"
            )

        funcs = filter_by_keys(keys, tables[name], drop_index=name == "events")
        return {partial(_remove_empty_partitions, func) for func in funcs}  # type: ignore

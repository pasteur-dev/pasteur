"""ACS folktables-task views matching the PrivPGD paper setup.

PrivPGD (ICML 2024, https://github.com/jaabmar/private-pgd) trains five separate
folktables tasks on California ACS 2018 5-Year person-level PUMS data. We
reproduce four of them — Income, Employment, PublicCoverage, TravelTime —
on California 2014–2018 1-Year partitions concatenated, which matches the
2018 5-Year sample window and yields a comparable row count. Each view
exposes the task's feature subset plus a binary `label` derived from the
task's target column, after applying the same row filter folktables uses
(`ACSIncome.preprocess_fn`, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar

from ....utils import LazyFrame, gen_closure, get_relative_fn
from ....utils.data import LazyDataset, LazyPartition
from ....view import TabularView, View

if TYPE_CHECKING:
    import pandas as pd


# PrivPGD's "CA 2018 5-Year" sample = California PUMS data covering 2014–2018.
# We approximate it by concatenating the matching 1-Year state partitions.
_DEFAULT_STATES = ("ca",)
_DEFAULT_YEARS = (2014, 2015, 2016, 2017, 2018)


# ---- per-task filter / label functions (module-level so they pickle cleanly
# across the worker processes pasteur spawns for partition processing) ----

def _filter_income(df):
    return (df["AGEP"] > 16) & (df["PINCP"] > 100) & (df["WKHP"] > 0)


def _label_income(s):
    return s > 50_000


def _filter_employment(df):
    return (df["AGEP"] > 16) & (df["AGEP"] < 90)


def _label_eq_one(s):
    # Employment + PublicCoverage targets are both "code == 1"; ESR / PUBCOV
    # are categoricals so we coerce to string for the comparison.
    return s.astype("string") == "1"


def _filter_public_coverage(df):
    return (df["AGEP"] < 65) & (df["PINCP"] <= 30_000)


def _filter_travel_time(df):
    return (df["AGEP"] > 16) & (df["ESR"].astype("string") == "1")


def _label_travel_time(s):
    return s > 20


def _process_target(
    load: Callable,
    features: list[str],
    target_col: str,
    target_fn: Callable,
    filter_fn: Callable | None,
    needed_cols: list[str],
):
    """Load only the columns we need, filter rows, select features, label."""
    df = load(columns=needed_cols)
    if filter_fn is not None:
        df = df[filter_fn(df)]
    out = df[features]
    out = out.assign(label=target_fn(df[target_col]))
    return out


# Union of all columns the catalog declares per table — every partition the
# acs_person / acs (relational) views emit gets padded to this column set so
# the union-schema parameters.yml validates against any sample year. Year-
# specific columns (RELP/RELSHIPP, JWTR/JWTRNS, TYPE/TYPEHUGQ, YBL/YRBLT) are
# all-NaN in the partitions whose schema generation doesn't carry them.
_PERSON_FULL_COLS = (
    "ST", "PUMA", "PWGTP", "SPORDER", "AGEP", "COW", "SCHL", "MAR", "OCCP",
    "POBP", "POWPUMA", "RELP", "RELSHIPP", "WKHP", "SEX", "RAC1P", "PINCP",
    "PUBCOV", "ESR", "DIS", "ESP", "CIT", "MIG", "MIL", "ANC", "NATIVITY",
    "DEAR", "DEYE", "DREM", "FER", "GCL", "JWMNP", "JWTR", "JWTRNS", "POVPIP",
    "NWLA", "NWAB", "NWAV", "NWLK", "NWRE",
)
_HOUSEHOLD_FULL_COLS = (
    "ST", "PUMA", "NP", "HINCP", "FINCP", "TYPE", "TYPEHUGQ", "BLD", "TEN",
    "VEH", "YBL", "YRBLT", "HHL", "HHT", "HUPAC", "FS", "ACR", "BDSP", "RMSP",
    "VALP", "RNTP", "BROADBND", "ACCESS",
)


def _pad_missing(df, cols):
    import pandas as pd

    missing = [c for c in cols if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = pd.NA
    return df


def _add_state_year(load: Callable, state: str, year: str):
    """Materialize a partition, tag it with `state` (postal abbrev) and `year`
    (int), and pad in any catalog-declared columns the partition's year
    doesn't carry (e.g. RELSHIPP/JWTRNS in pre-2019 partitions)."""
    df = load()
    df = _pad_missing(df, _PERSON_FULL_COLS)
    if "state" not in df.columns:
        df = df.assign(state=state)
    if "year" not in df.columns:
        df = df.assign(year=int(year))
    return df


def _process_household(load: Callable, state: str, year: str):
    """Materialize a household partition keyed by SERIALNO."""
    df = load()
    df = _pad_missing(df, _HOUSEHOLD_FULL_COLS)
    if "state" not in df.columns:
        df = df.assign(state=state)
    if "year" not in df.columns:
        df = df.assign(year=int(year))
    if df.index.name != "SERIALNO":
        df = df.set_index("SERIALNO")
    return df


def _subset(lf: LazyFrame, target_pids: frozenset[str]) -> LazyFrame:
    """Return a LazyFrame containing only the partitions in `target_pids`."""
    if not lf.partitioned:
        return lf
    parts = {pid: p for pid, p in lf.items() if pid in target_pids}
    return LazyDataset(lf.merged_load, parts)


class _AcsTaskView(TabularView):
    """Base class for the four PrivPGD folktables-task views."""

    dataset = "acs"
    deps = {"table": ["person"]}

    _features: ClassVar[list[str]] = []
    _target_col: ClassVar[str] = ""
    _filter_extra_cols: ClassVar[list[str]] = []
    _target_fn: ClassVar[Callable[["pd.Series"], "pd.Series"]] = staticmethod(
        lambda s: s
    )
    _filter_fn: ClassVar[Callable[["pd.DataFrame"], "pd.Series"] | None] = None

    def __init__(
        self,
        states: tuple[str, ...] = _DEFAULT_STATES,
        years: tuple[int, ...] = _DEFAULT_YEARS,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._states = tuple(s.lower() for s in states)
        self._years = tuple(years)
        self._targets = frozenset(
            f"{s}_{y}" for s in self._states for y in self._years
        )

    @property
    def _needed_cols(self) -> list[str]:
        return sorted(set(self._features) | {self._target_col} | set(self._filter_extra_cols))

    def query(self, name: str, **tables: LazyFrame):
        assert name == "table"
        person = tables["person"]

        if not person.partitioned:
            return _process_target(
                lambda **kw: person(**kw),
                self._features,
                self._target_col,
                self._target_fn,
                self._filter_fn,
                self._needed_cols,
            )

        keys = set(person.keys())
        missing = self._targets - keys
        assert not missing, (
            f"View {self.name} expects partitions {sorted(missing)} in "
            f"acs.person; available: {sorted(keys)[:5]}…"
        )

        return {
            pid: gen_closure(
                _process_target,
                fun,
                self._features,
                self._target_col,
                self._target_fn,
                self._filter_fn,
                self._needed_cols,
            )
            for pid, fun in person.items()
            if pid in self._targets
        }

    def split_keys(self, keys, req_splits, splits, random_state):
        return super().split_keys(
            _subset(keys, self._targets), req_splits, splits, random_state
        )

    def filter_table(self, name: str, keys: LazyFrame, **tables: LazyFrame):
        return super().filter_table(
            name,
            _subset(keys, self._targets),
            **{k: _subset(v, self._targets) for k, v in tables.items()},
        )


class AcsIncomeView(_AcsTaskView):
    """ACSIncome: predict person income > 50k from 10 demographic/work features."""

    name = "acs_income"
    parameters = get_relative_fn("parameters_income.yml")

    _features = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP",
                 "SEX", "RAC1P"]
    _target_col = "PINCP"
    _filter_extra_cols = ["AGEP", "PINCP", "WKHP"]
    _target_fn = staticmethod(_label_income)
    _filter_fn = staticmethod(_filter_income)


class AcsEmploymentView(_AcsTaskView):
    """ACSEmployment: predict employment status (ESR == 1) from 16 features."""

    name = "acs_employment"
    parameters = get_relative_fn("parameters_employment.yml")

    _features = ["AGEP", "SCHL", "MAR", "RELP", "DIS", "ESP", "CIT", "MIG",
                 "MIL", "ANC", "NATIVITY", "DEAR", "DEYE", "DREM", "SEX", "RAC1P"]
    _target_col = "ESR"
    _filter_extra_cols = ["AGEP"]
    _target_fn = staticmethod(_label_eq_one)
    _filter_fn = staticmethod(_filter_employment)


class AcsPublicCoverageView(_AcsTaskView):
    """ACSPublicCoverage: predict public health insurance (PUBCOV == 1)."""

    name = "acs_public_coverage"
    parameters = get_relative_fn("parameters_public_coverage.yml")

    _features = ["AGEP", "SCHL", "MAR", "SEX", "DIS", "ESP", "CIT", "MIG",
                 "MIL", "ANC", "NATIVITY", "DEAR", "DEYE", "DREM", "PINCP",
                 "ESR", "ST", "FER", "RAC1P"]
    _target_col = "PUBCOV"
    _filter_extra_cols = ["AGEP", "PINCP"]
    _target_fn = staticmethod(_label_eq_one)
    _filter_fn = staticmethod(_filter_public_coverage)


class AcsTravelTimeView(_AcsTaskView):
    """ACSTravelTime: predict commute > 20 min (JWMNP > 20) for employed adults."""

    name = "acs_travel_time"
    parameters = get_relative_fn("parameters_travel_time.yml")

    # Uses 2018-era columns: RELP (not RELSHIPP), JWTR (not JWTRNS).
    _features = ["AGEP", "SCHL", "MAR", "SEX", "DIS", "ESP", "MIG", "RELP",
                 "RAC1P", "PUMA", "ST", "CIT", "OCCP", "JWTR", "POWPUMA",
                 "POVPIP"]
    _target_col = "JWMNP"
    _filter_extra_cols = ["AGEP", "ESR"]
    _target_fn = staticmethod(_label_travel_time)
    _filter_fn = staticmethod(_filter_travel_time)


# Each year is split into this many equal-row chunks so the grouped-by-year
# view emits ~6× as many partitions, parallelizing per-partition processing
# without going back to the original 459-way (state, year) partitioning.
_CHUNKS_PER_YEAR = 6


def _slice_chunk(df, chunk_idx: int, n_chunks: int):
    """Deterministic row-range slice of df. Boundaries are computed as
    ceil(n / n_chunks) * chunk_idx so the same cuts apply in the keys
    pipeline and the table pipeline (same n, same chunk_idx, same boundaries)."""
    n = len(df)
    chunk_size = -(-n // n_chunks)  # ceil division
    start = chunk_idx * chunk_size
    end = min(n, start + chunk_size)
    return df.iloc[start:end]


def _concat_year_for_view(state_funs, year: str, chunk_idx: int, n_chunks: int):
    """Concatenate all states' person partitions for one year, then slice the
    requested chunk. Pads the union schema and tags each row with `state`
    (per-source) and `year` (constant). States are loaded in the order they
    were given; the caller is expected to pass them sorted alphabetically so
    the fresh `id` index matches what `_concat_keys_for_year` produces from
    the keys side."""
    import pandas as pd

    parts = []
    for state, fun in state_funs:
        df = fun()
        df = _pad_missing(df, _PERSON_FULL_COLS)
        if "state" not in df.columns:
            df = df.assign(state=state)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    if "year" not in df.columns:
        df = df.assign(year=int(year))
    df.index = df.index.astype("int64").rename("id")
    return _slice_chunk(df, chunk_idx, n_chunks)


def _concat_keys_for_year(loaders, chunk_idx: int, n_chunks: int):
    """Mirror `_concat_year_for_view` for the keys (no-column) DataFrame so
    keys.id and table.id agree row-by-row after the per-year regrouping."""
    import pandas as pd

    df = pd.concat([ld() for ld in loaders], ignore_index=True)
    df.index = df.index.astype("int64").rename("id")
    return _slice_chunk(df, chunk_idx, n_chunks)


def _is_state_year_pid(pid: str) -> bool:
    """True for `{state}_{year}` source pids (e.g. `ca_2018`); False for the
    regrouped `{year}_{chunk}` pids we emit (e.g. `2018_0`).  Used by
    `_group_by_year` to stay idempotent when called twice on the same lf."""
    first = pid.partition("_")[0]
    # state codes are alphabetic (e.g. `ca`, `ny`); years are 4-digit numbers
    return not (len(first) == 4 and first.isdigit())


def _group_by_year(lf: LazyFrame, table_view: bool) -> LazyFrame:
    """Re-partition a `{state}_{year}`-keyed LazyFrame into ``_CHUNKS_PER_YEAR``
    chunks per year (pids `{year}_{chunk}`). `table_view=True` builds the
    per-row state column and pads the union schema; otherwise just
    concatenates index-only key frames.

    Idempotent: if the pids are already in `{year}_{chunk}` form (e.g.
    `filter_table` receives keys that `split_keys` has already regrouped),
    pass through."""
    if not lf.partitioned:
        return lf
    sample = next(iter(lf.keys()))
    if not _is_state_year_pid(sample):
        return lf
    by_year: dict[str, list] = {}
    for pid, p in lf.items():
        state, year = pid.rsplit("_", 1)
        by_year.setdefault(year, []).append((state, p))
    new_parts = {}
    for year, items in by_year.items():
        items.sort(key=lambda sp: sp[0])  # alphabetical, deterministic
        for chunk_idx in range(_CHUNKS_PER_YEAR):
            pid = f"{year}_{chunk_idx}"
            if table_view:
                new_parts[pid] = LazyPartition(
                    _concat_year_for_view,
                    None,
                    items,
                    year,
                    chunk_idx,
                    _CHUNKS_PER_YEAR,
                )
            else:
                loaders = [p for _, p in items]
                new_parts[pid] = LazyPartition(
                    _concat_keys_for_year, None, loaders, chunk_idx, _CHUNKS_PER_YEAR
                )
    return LazyDataset(lf.merged_load, new_parts)


class AcsPersonView(TabularView):
    """Full per-person table, regrouped to one partition per year.

    States within a year are concatenated (alphabetical order) into a single
    partition with a fresh `id` index. `state` and `year` columns survive on
    every row. Reduces 459 → ~9 partitions and trims the per-partition write
    overhead. `split_keys` / `filter_table` are overridden to perform the same
    regrouping on the keys side so the `id`s line up."""

    name = "acs_person"
    dataset = "acs"
    deps = {"table": ["person"]}
    parameters = get_relative_fn("parameters_person.yml")

    def query(self, name, **tables: LazyFrame):
        assert name == "table"
        person = tables["person"]
        regrouped = _group_by_year(person, table_view=True)
        if not regrouped.partitioned:
            return regrouped()
        return {pid: regrouped[pid] for pid in regrouped.keys()}

    def split_keys(self, keys, req_splits, splits, random_state):
        return super().split_keys(
            _group_by_year(keys, table_view=False),
            req_splits,
            splits,
            random_state,
        )

    def filter_table(self, name, keys: LazyFrame, **tables: LazyFrame):
        return super().filter_table(
            name,
            _group_by_year(keys, table_view=False),
            **tables,
        )


class AcsRelationalView(View):
    """Two-table relational view: household (one row per SERIALNO) plus the
    person table (multiple rows per SERIALNO) for hierarchical/relational
    synthesizers like Mare. Partitioned the same way as the dataset; both
    tables share the `{state}_{year}` partition key so per-partition
    SERIALNO uniqueness is preserved."""

    name = "acs"
    dataset = "acs"
    deps = {"household": ["household"], "person": ["person"]}
    trn_deps = {"person": ["household"]}
    parameters = get_relative_fn("parameters_relational.yml")

    def query(self, name, **tables: LazyFrame):
        match name:
            case "household":
                src = tables["household"]
                if not src.partitioned:
                    return _process_household(lambda: src(), "unknown", "0")
                return {
                    pid: gen_closure(
                        _process_household, fun, *pid.rsplit("_", 1)
                    )
                    for pid, fun in src.items()
                }
            case "person":
                src = tables["person"]
                if not src.partitioned:
                    return _add_state_year(lambda: src(), "unknown", "0")
                return {
                    pid: gen_closure(_add_state_year, fun, *pid.rsplit("_", 1))
                    for pid, fun in src.items()
                }
            case other:
                raise AssertionError(f"Table {other!r} not part of view {self.name}")

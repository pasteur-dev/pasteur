"""ACS PUMS 1-Year dataset.

Downloads US Census ACS Public Use Microdata Sample (PUMS) zip archives, one
person and one household per state per year, for the years configured. The
2018→2019 schema rename (RELP→RELSHIPP, JWTR→JWTRNS) is handled by typing
both names in the catalog and using `AcsCSVDataset` to filter to whichever
columns are actually present in each file."""

from __future__ import annotations


import pandas as pd
from kedro_datasets.pandas import CSVDataset
from kedro.io.core import PROTOCOL_DELIMITER

import logging
import os
import re
from typing import TYPE_CHECKING, Callable
from zipfile import BadZipFile, ZipFile

from ....dataset import Dataset
from ....utils import (
    LazyChunk,
    LazyFrame,
    RawSource,
    gen_closure,
    get_relative_fn,
    to_chunked,
)
from ....utils.progress import process_in_parallel

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

STATES = [
    "al",
    "ak",
    "az",
    "ar",
    "ca",
    "co",
    "ct",
    "de",
    "fl",
    "ga",
    "hi",
    "id",
    "il",
    "in",
    "ia",
    "ks",
    "ky",
    "la",
    "me",
    "md",
    "ma",
    "mi",
    "mn",
    "ms",
    "mo",
    "mt",
    "ne",
    "nv",
    "nh",
    "nj",
    "nm",
    "ny",
    "nc",
    "nd",
    "oh",
    "ok",
    "or",
    "pa",
    "ri",
    "sc",
    "sd",
    "tn",
    "tx",
    "ut",
    "vt",
    "va",
    "wa",
    "wv",
    "wi",
    "wy",
    "pr",
]

# 2020 1-Year was not released (Census paused due to COVID).
DEFAULT_YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]

# Inside the zip, year >= 2017 uses psam_{p|h}{FIPS}.csv (numeric state code);
# year <= 2016 uses ss{yy}{p|h}{state}.csv (letter abbreviation).
_CSV_INSIDE = re.compile(
    r"^(?:psam_|ss\d{2})([ph])([a-z]{2}|\d{2})\.csv$", re.IGNORECASE
)

# FIPS state code → 2-letter postal abbreviation. Used to normalize
# psam_p06.csv → p_ca.csv so the catalog regex sees uniform names regardless
# of year.
_FIPS_TO_ABBR = {
    "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co",
    "09": "ct", "10": "de", "11": "dc", "12": "fl", "13": "ga", "15": "hi",
    "16": "id", "17": "il", "18": "in", "19": "ia", "20": "ks", "21": "ky",
    "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn",
    "28": "ms", "29": "mo", "30": "mt", "31": "ne", "32": "nv", "33": "nh",
    "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd", "39": "oh",
    "40": "ok", "41": "or", "42": "pa", "44": "ri", "45": "sc", "46": "sd",
    "47": "tn", "48": "tx", "49": "ut", "50": "vt", "51": "va", "53": "wa",
    "54": "wv", "55": "wi", "56": "wy", "72": "pr",
}


def _extract(loc: str, fn: str, dst_year: str) -> None:
    if not fn.endswith(".zip"):
        return
    os.makedirs(dst_year, exist_ok=True)
    src_zip = os.path.join(loc, fn)
    try:
        zf = ZipFile(src_zip, "r")
    except BadZipFile:
        logger.warning(
            "Skipping corrupt/incomplete zip %s (delete it and re-run "
            "`pasteur download acs` to redownload).",
            src_zip,
        )
        return
    with zf:
        for member in zf.namelist():
            m = _CSV_INSIDE.match(os.path.basename(member))
            if not m:
                continue
            survey, state = m.group(1).lower(), m.group(2).lower()
            # 2017+ uses FIPS digits; map back to postal abbreviation.
            if state.isdigit():
                state = _FIPS_TO_ABBR.get(state, state)
            out_path = os.path.join(dst_year, f"{survey}_{state}.csv")
            if os.path.exists(out_path):
                continue
            with zf.open(member) as src, open(out_path, "wb") as dst:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    dst.write(buf)


def _add_id(load: Callable) -> "pd.DataFrame":
    df = load().reset_index(drop=True)
    df.index = df.index.astype("int64").rename("id")
    return df


class AcsDataset(Dataset):
    name = "acs"
    deps = {"person": ["person"], "household": ["household"]}
    key_deps = ["person"]

    folder_name = "acs"
    catalog = get_relative_fn("catalog.yml")

    def __init__(
        self,
        years: list[int] | None = None,
        states: list[str] | None = None,
        **_,
    ) -> None:
        super().__init__(**_)
        self._years = list(years) if years is not None else list(DEFAULT_YEARS)
        self._states = [s.lower() for s in (states or STATES)]

        self.raw_sources = RawSource(
            files=[
                f"https://www2.census.gov/programs-surveys/acs/data/pums/"
                f"{year}/1-Year/csv_{survey}{state}.zip"
                for year in self._years
                for survey in ("p", "h")
                for state in self._states
            ],
            # Preserve `<year>/1-Year/csv_*.zip` so files don't collide on
            # basename across years.
            keep_dirs=2,
            desc=(
                f"ACS PUMS 1-Year ({min(self._years)}-{max(self._years)}) "
                "(US Census public domain). "
                "https://www.census.gov/programs-surveys/acs/microdata.html"
            ),
        )

    def bootstrap(self, raw: str, dst: str) -> None:
        os.makedirs(dst, exist_ok=True)
        jobs = []
        # On-disk layout from the downloader: <raw>/<year>/1-Year/csv_*.zip
        for year in sorted(os.listdir(raw)):
            if not year.isdigit():
                continue
            year_dir = os.path.join(raw, year)
            if not os.path.isdir(year_dir):
                continue
            dst_year = os.path.join(dst, year)
            for horizon in os.listdir(year_dir):
                horizon_dir = os.path.join(year_dir, horizon)
                if not os.path.isdir(horizon_dir):
                    continue
                for fn in os.listdir(horizon_dir):
                    if fn.endswith(".zip"):
                        if "2014" in horizon_dir and "nh.zip" in fn:
                            # 2014 household survey is corrupted, skip it
                            continue
                        jobs.append(
                            {"loc": horizon_dir, "fn": fn, "dst_year": dst_year}
                        )
        if not jobs:
            logger.warning(
                "No ACS zip files found under %s. Run `pasteur download acs` first.",
                raw,
            )
            return
        process_in_parallel(_extract, jobs, {}, desc="Unzipping ACS")

    def ingest(self, name: str, **tables: LazyFrame):
        src = tables[name]
        assert (
            src.partitioned
        ), f"acs.{name} expected partitioned (one CSV per state-year)"
        return {pid: gen_closure(_add_id, fun) for pid, fun in src.items()}

    @to_chunked
    def keys(self, person: LazyChunk):
        # Keys are household-level (one row per SERIALNO) so that splits and
        # chunkings keep every person of a household together. Views that
        # care about person-level uniqueness still get unique-per-person
        # filtering via person.SERIALNO column matching keys.index.
        df = person()
        serialnos = df["SERIALNO"].drop_duplicates()
        return pd.DataFrame(index=pd.Index(serialnos.values, name="SERIALNO"))


class AcsCSVDataset(CSVDataset):
    def load(self) -> pd.DataFrame:
        load_path = str(self._get_load_path())
        if self._protocol != "file":
            load_path = f"{self._protocol}{PROTOCOL_DELIMITER}{load_path}"

        load_args = dict(self._load_args)
        usecols = load_args.pop("usecols", None)
        dtype = load_args.pop("dtype", None)
        parse_dates = load_args.pop("parse_dates", None)

        if usecols is not None or dtype is not None or parse_dates is not None:
            header_args = {"nrows": 0}
            if self._protocol == "file":
                header = pd.read_csv(load_path, **header_args).columns
            else:
                header = pd.read_csv(
                    load_path, storage_options=self._storage_options, **header_args
                ).columns
            present = set(header)

            if usecols is not None:
                load_args["usecols"] = [c for c in usecols if c in present]
            if dtype is not None:
                load_args["dtype"] = {k: v for k, v in dtype.items() if k in present}
            if parse_dates is not None:
                load_args["parse_dates"] = [c for c in parse_dates if c in present]

        if self._protocol == "file":
            return pd.read_csv(load_path, **load_args)
        return pd.read_csv(
            load_path, storage_options=self._storage_options, **load_args
        )

import logging

import pandas as pd

from ....dataset import Dataset, split_keys
from ....utils import get_relative_fn, gen_closure
from ....utils.progress import piter
from typing import Callable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ....kedro.dataset import PatternDataSet

logger = logging.getLogger(__name__)

LazyFrame = dict[str, Callable[..., pd.DataFrame]]

def ingest_base(base: LazyFrame, base1: LazyFrame, base2: LazyFrame) -> LazyFrame:
    import pandas as pd

    old_pids = list(base.keys())
    new_pids = list(base1.keys())
    assert set(base2.keys()) == set(new_pids), "There's a missmatch of base2 and base1 files."

    def process_old(pid: str):
        chunk = base[pid]()

        # Process new spec units
        spec_unit = chunk["spec_unit"].astype(str)
        get_spec_idx = lambda i: spec_unit.str.slice(i, i + 1).replace("", pd.NA).astype("category")
        spec_units = pd.DataFrame({f"spec_unit_{i + 1}": get_spec_idx(i) for i in range(5)})

        # Drop denormalized facility columns
        return chunk.drop(columns=[
            'fac_acute_care_ind',
            'fac_long_term_ac_ind',
            'fac_other_ltc_ind',
            'fac_peds_ind',
            'fac_psych_ind',
            'fac_rehab_ind',
            'fac_snf_ind',
            'fac_teaching_ind',
            "spec_unit"
        ]).join(spec_units)

    def process_new(pid: str):
        chunk1 = base1[pid]()
        chunk2 = base2[pid]()

        return chunk1.drop(
            columns=[
                "filler_space",
                "apr_grouper_error_code",
                "apr_grouper_version_nbr",
                "ms_grouper_error_code",
                "ms_grouper_version_nbr",
            ]
        ).join(chunk2.drop(columns=["filler_space"]))
    
    return {
        **{pid: gen_closure(process_old, pid) for pid in old_pids},
        **{pid: gen_closure(process_new, pid) for pid in new_pids}
    }

class TexasDataset(Dataset):
    name = "texas"
    deps = {
        "base": ["base", "base1", "base2"],
        "facility": ["facility", "base"],
        "charges": ["charges"],
    }
    key_deps = ["base", "base1", "base2"]

    folder_name = "texas"
    catalog = get_relative_fn("catalog.yml")

    def bootstrap(self, location: str, bootstrap: str):
        import os
        from zipfile import ZipFile

        os.makedirs(bootstrap, exist_ok=True)
        for fn in piter(os.listdir(location), leave=False):
            if not (fn.endswith("_tab") or fn.endswith("-tab-delimited")):
                continue

            with ZipFile(os.path.join(location, fn), "r") as zf:
                logger.info(f"Extracting {fn}...")
                zf.extractall(bootstrap)

    def ingest(self, name: str, **tables: LazyFrame):
        if name == "base":
            return ingest_base(**tables)
        return pd.DataFrame()

    def keys(self, ratios: dict[str, float], random_state: int, **tables: pd.DataFrame):
        return split_keys(tables[""], ratios, random_state)

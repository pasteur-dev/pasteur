from ....view import View
from ....utils import get_relative_fn, to_chunked, LazyChunk, LazyFrame
import pandas as pd

class TexasChargesView(View):
    name = "texas_charges"
    dataset = "texas"
    tabular = True

    deps = {"table": ["charges"]}
    parameters = get_relative_fn("./parameters_charges.yml")

    @to_chunked
    def ingest(self, name: str, charges: LazyChunk):
        return charges()


class TexasBillionView(View):
    name = "texas_billion"
    dataset = "texas"
    tabular = True

    deps = {"table": ["charges"]}
    parameters = get_relative_fn("./parameters_billion.yml")

    def ingest(self, name: str, charges: LazyFrame):
        from ...utils import ColumnResampler
        
        sample = charges.sample()
        res_rev = ColumnResampler(sample["revenue_code"], height=300)
        res_proc = ColumnResampler(sample["hcpcs_procedure_code"], height=2950)

        return self._ingest(name, charges, res_rev, res_proc)  # type: ignore

    @to_chunked
    def _ingest(self, name: str, charges: LazyChunk, res_rev, res_proc):
        a = charges().drop(columns=["modifier_3", "modifier_4"])
        # Limit domain of rev code and hcpcs
        a = a.assign(
            revenue_code=res_rev.resample(a), hcpcs_procedure_code=res_proc.resample(a)
        )
        return (
            pd.concat([a, a, a])[:25_000_000]
            .reset_index(drop=True)
            .rename_axis("charge_id")
        )


class TexasBaseView(View):
    name = "texas_base"
    dataset = "texas"
    tabular = True

    pid_pattern = ""  # "20(?:06|07|11|15)"

    deps = {"table": ["base"]}
    parameters = get_relative_fn("./parameters_base.yml")

    @to_chunked
    def ingest(self, name: str, base: LazyChunk):
        return base()

    # parameters = get_relative_fn("parameters.yml")

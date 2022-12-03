from ....view import View
from ....utils import get_relative_fn, to_chunked, LazyChunk


class TexasChargesView(View):
    name = "texas_charges"
    dataset = "texas"
    tabular = True

    deps = {"table": ["charges"]}
    parameters = get_relative_fn("./parameters_charges.yml")

    @to_chunked
    def ingest(self, name: str, charges: LazyChunk):
        return charges()


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

from ....view import View
from ....utils import get_relative_fn

class TexasChargesView(View):
    name = "texas_charges"
    dataset = "texas"
    tabular = True

    deps = {"table": ["charges"]}
    parameters = get_relative_fn("./parameters_charges.yml")

    def ingest(self, name, charges):
        return charges

class TexasBaseView(View):
    name = "texas_base"
    dataset = "texas"
    tabular = True

    pid_pattern = "" #"20(?:06|07|11|15)"

    deps = {"table": ["base"]}
    parameters = get_relative_fn("./parameters_base.yml")

    def ingest(self, name, base):
        return base

    
    
    # parameters = get_relative_fn("parameters.yml")
from ....view import View
from ....utils import get_relative_fn

class TabTexasView(View):
    name = "tab_texas_charges"
    dataset = "texas"
    tabular = True

    pid_pattern = "" #"20(?:06|07|11|15)"

    deps = {"table": ["charges"]}
    parameters = get_relative_fn("./parameters.yml")

    def ingest(self, name, charges):
        import re
        return {pid: fun for pid, fun in charges.items() if re.search(self.pid_pattern, pid)}

    
    
    # parameters = get_relative_fn("parameters.yml")
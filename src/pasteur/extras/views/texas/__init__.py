from ....view import TabularView
from ....utils import get_relative_fn

class TabTexasView(TabularView):
    name = "tab_texas"
    dataset = "texas"
    # parameters = get_relative_fn("parameters.yml")
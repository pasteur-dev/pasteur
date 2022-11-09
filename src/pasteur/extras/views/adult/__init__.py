from ....view import TabularView
from ....utils import get_relative_fn

class TabAdultView(TabularView):
    name = "tab_adult"
    dataset = "adult"
    parameters_fn = get_relative_fn("catalog.yml")
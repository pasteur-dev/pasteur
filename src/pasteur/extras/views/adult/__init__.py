from ....view import TabularView
from ....utils import get_relative_fn

class TabAdultView(TabularView):
    name = "tab_adult"
    dataset = "adult"
    parameters = get_relative_fn("parameters.yml")
from ....view import TabularView
from ....utils import get_relative_fn

class TabAdultView(TabularView):
    name = "tab_adult"
    dataset = "adult"
    parameters = get_relative_fn("parameters.yml")
    # Adult is too small and when running multiple shuffles we end up
    # with different transformer mappings which breaks distr metrics
    fit_global = True

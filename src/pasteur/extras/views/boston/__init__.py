from ....view import TabularView
from ....utils import get_relative_fn

class BostonView(TabularView):
    name = "boston"
    dataset = "boston"
    parameters = get_relative_fn("parameters.yml")
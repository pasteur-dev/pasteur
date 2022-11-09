from ...view import TabularView
from ..datasets import adult as _


class TabAdultView(TabularView):
    name = "tab_adult"
    dataset = "adult"

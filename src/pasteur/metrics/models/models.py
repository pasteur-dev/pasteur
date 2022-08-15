from abc import ABC, abstractmethod
import pandas as pd
from ...utils import find_subclasses


class BaseModel(ABC):
    name = None
    size_limit = None
    x_trn_type = None
    y_trn_type = None
    y_col_types = None

    def __init__(self, random_state: int):
        self.random_state = random_state

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        pass

    @abstractmethod
    def score(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        pass


class SklearnModel(BaseModel):
    cls: type = None
    base_args = {}

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        self.model = self.cls(**self.base_args, random_state=self.random_state)
        self.model.fit(x, y)

    def score(self, x: pd.DataFrame, y: pd.DataFrame):
        return self.model.score(x, y)


class SklearnClassifierModel(SklearnModel):
    x_trn_type = "num"
    y_trn_type = "idx"


class SklearnRegressionModel(SklearnModel):
    x_trn_type = "num"
    y_trn_type = "num"
    y_col_types = ["numerical"]


class SGDClassifierSklearn(SklearnClassifierModel):
    name = "sgd_clsr"
    base_args = {"loss": "log_loss"}

    @property
    def cls(self):
        from sklearn.linear_model import SGDClassifier

        return SGDClassifier


class SvmClassifierSklearn(SklearnClassifierModel):
    name = "svm_clsr"
    size_limit = 10000

    @property
    def cls(self):
        from sklearn.svm import LinearSVC

        return LinearSVC


class DecisionTreeClassifierSklearn(SklearnClassifierModel):
    name = "tree_clsr"

    @property
    def cls(self):
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier


class RandomForestClassifierSklearn(SklearnClassifierModel):
    name = "forest_clsr"
    base_args = {"min_samples_leaf": 0.0001}

    @property
    def cls(self):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier


def get_models():
    return find_subclasses(BaseModel)


def get_required_types():
    types = []
    for cls in get_models().values():
        if cls.x_trn_type not in types:
            types.append(cls.x_trn_type)
        if cls.y_trn_type not in types:
            types.append(cls.y_trn_type)

    return types

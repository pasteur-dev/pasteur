from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestClassifier,
)
from ...utils import find_subclasses


class BaseModel(ABC):
    name = None
    x_trn_type = None
    y_trn_type = None
    x_col_types = None
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
    cls: type[ClassifierMixin | RegressorMixin] = None

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        self.model = self.cls(random_state=self.random_state)
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


class LogisticRegressionSklearn(SklearnClassifierModel):
    name = "logistic_regr"
    cls = LogisticRegression


class SvmClassifierSklearn(SklearnClassifierModel):
    name = "svm_clsr"
    cls = SVC


class DecisionTreeClassifierSklearn(SklearnClassifierModel):
    name = "tree_clsr"
    cls = DecisionTreeClassifier


class RandomForestClassifierSklearn(SklearnClassifierModel):
    name = "forest_clsr"
    cls = RandomForestClassifier


class GradientBoostingClassifierSklearn(SklearnClassifierModel):
    name = "gradboost_clsr"
    cls = GradientBoostingClassifier


class SklearnRegressionModel(SklearnRegressionModel):
    name = "linear_regr"
    cls = LinearRegression


class SklearnRegressionModel(SklearnRegressionModel):
    name = "svm_regr"
    cls = SVC


class SklearnRegressionModel(SklearnRegressionModel):
    name = "decisiontree_regr"
    cls = DecisionTreeRegressor


class SklearnRegressionModel(SklearnRegressionModel):
    name = "forest_regr"
    cls = RandomForestRegressor


class SklearnRegressionModel(SklearnRegressionModel):
    name = "gradboost_regr"
    cls = GradientBoostingRegressor


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

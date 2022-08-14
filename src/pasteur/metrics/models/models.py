from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin, ClassifierMixin

from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
)
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
    cls: type[ClassifierMixin | RegressorMixin] = None
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
    cls = SGDClassifier
    base_args = {"loss": "log_loss"}


class SvmClassifierSklearn(SklearnClassifierModel):
    name = "svm_clsr"
    cls = LinearSVC
    size_limit = 10000


class DecisionTreeClassifierSklearn(SklearnClassifierModel):
    name = "tree_clsr"
    cls = DecisionTreeClassifier


class RandomForestClassifierSklearn(SklearnClassifierModel):
    name = "forest_clsr"
    cls = RandomForestClassifier
    base_args = {"min_samples_leaf": 0.0001}


class GradientBoostingClassifierSklearn(SklearnClassifierModel):
    name = "gradboost_clsr"
    cls = HistGradientBoostingClassifier

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):

        # Drop rare columns to avoid erroring out on histograms
        cnts = y[y.columns[0]].value_counts()
        drop_cats = list(cnts[cnts < 10].index)
        idx = ~y[y.columns[0]].isin(drop_cats)
        x, y = x[idx], y[idx]

        return super().fit(x, y)


# class LinearRegressionModel(SklearnRegressionModel):
#     name = "linear_regr"
#     cls = LinearRegression


# class SvmRegressionModel(SklearnRegressionModel):
#     name = "svm_regr"
#     cls = SVC


# class DecisionTreeRegressionModel(SklearnRegressionModel):
#     name = "decisiontree_regr"
#     cls = DecisionTreeRegressor


# class RandomForestRegressionModel(SklearnRegressionModel):
#     name = "forest_regr"
#     cls = RandomForestRegressor


# class GradientBoostingRegressionModel(SklearnRegressionModel):
#     name = "gradboost_regr"
#     cls = GradientBoostingRegressor


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

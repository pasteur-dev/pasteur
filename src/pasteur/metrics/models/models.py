from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ...utils import find_subclasses

if TYPE_CHECKING:
    import pandas as pd


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


class XGBoostlassifierModel(BaseModel):
    name = "xgb_clsr"
    x_trn_type = "num"
    y_trn_type = "idx"

    def __init__(self, random_state: int, num_round: int = 10):
        super().__init__(random_state)
        self.num_round = num_round

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        import numpy as np
        import xgboost as xgb

        dtrain = xgb.DMatrix(x, label=y)
        self._bst = xgb.train(
            {"objective": "multi:softmax", "num_class": np.max(y.to_numpy()) + 1},
            dtrain,
            self.num_round,
        )

    def score(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        import numpy as np
        import xgboost as xgb

        deval = xgb.DMatrix(x, label=y)
        return np.mean(self._bst.predict(deval) == y.to_numpy().T)


# class LightGBMClassifierModel(BaseModel):
#     name = "gbm_clsr"
#     x_trn_type = "num"
#     y_trn_type = "idx"

#     def fit(self, x: pd.DataFrame, y: pd.DataFrame):
#         import lightgbm as lgb

#         lgb.register_logger(logging.root)

#         train = lgb.Dataset(x, y)
#         self._gbm = lgb.train({}, train)

#     def score(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
#         import lightgbm as lgb

#         return self._gbm.eval(lgb.Dataset(x, y), "")


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

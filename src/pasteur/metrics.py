from typing import List
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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

from .metadata import TableMeta


class CalcTableMetrics:
    MULTI_PROCESS = True

    MODELS = [
        ("lr", LogisticRegression, LinearRegression),
        ("svm", SVC, SVC),
        ("tree", DecisionTreeClassifier, DecisionTreeRegressor),
        ("forest", RandomForestClassifier, RandomForestRegressor),
        ("gb", GradientBoostingClassifier, GradientBoostingRegressor),
        # ("bayes", GaussianNB, None),
    ]

    def __init__(
        self,
        meta: TableMeta,
        wrk: pd.DataFrame,
        alg: pd.DataFrame,
        dev: pd.DataFrame,
        random_state=None,
        ratio: float = 0.2,
    ):
        self.meta = meta

        self.wrk = wrk
        self.alg = alg
        self.dev = dev

        self.random_state = random_state
        self.ratio = ratio

    def _fit_data(
        self, target_col: str, train: pd.DataFrame, *tests: List[pd.DataFrame]
    ):
        columns = []

        for name, col in self.meta.cols.items():
            if name == target_col:
                continue

            if col.is_id():
                continue
            elif col.is_cat():
                columns.append(
                    (name, OneHotEncoder(handle_unknown="infrequent_if_exist"), [name])
                )
            else:
                columns.append((name, StandardScaler(), [name]))

        trans = ColumnTransformer(
            columns, remainder="drop", verbose_feature_names_out=False
        )

        train_t = trans.fit_transform(train)
        test_t = [trans.transform(test) for test in tests]
        return train_t, *test_t

    def _test_alg(self, args):
        target, type, train, test, model, clf_l, clf_r = args
        x_train, x_test, x_wrk, x_dev = self._fit_data(
            target, train, test, self.wrk, self.dev
        )
        y_train, y_test, y_wrk, y_dev = (
            train[target],
            test[target],
            self.wrk[target],
            self.dev[target],
        )

        clf_c = clf_l if self.meta[target].is_cat() else clf_r

        if clf_c is None:
            return

        clf = clf_c(random_state=self.random_state)
        clf.fit(x_train, y_train)

        res_train = clf.score(x_train, y_train)
        res_dev = clf.score(x_dev, y_dev)

        if type == "alg":
            res_test = clf.score(x_test, y_test)
            res_wrk = clf.score(x_wrk, y_wrk)
        else:
            res_test = np.NAN
            res_wrk = np.NAN

        return (model, type, target, res_train, res_test, res_wrk, res_dev)

    def calculate(self):
        targets = [*self.meta.targets, *self.meta.sensitive]
        jobs = []

        for model, clf_l, clf_r in self.MODELS:
            for type, data in [("wrk", self.wrk), ("alg", self.alg)]:
                train, test = train_test_split(
                    data, test_size=self.ratio, random_state=self.random_state
                )

                for target in targets:
                    jobs.append((target, type, train, test, model, clf_l, clf_r))

        from tqdm import tqdm

        if self.MULTI_PROCESS:
            from tqdm.contrib.concurrent import thread_map

            target_res = thread_map(self._test_alg, jobs, tqdm_class=tqdm)
        else:
            target_res = tqdm(map(self._test_alg, jobs), total=len(jobs))

        target_res = pd.DataFrame(
            target_res,
            columns=[
                "model",
                "data",
                "target",
                "train_results",
                "test_results",
                "wrk_results",
                "dev_results",
            ],
        )
        return target_res

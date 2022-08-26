import mlflow
import pandas as pd
from ..mlflow import gen_html_table


def mlflow_log_model_results(name: str, res: pd.DataFrame):
    if not mlflow.active_run():
        return

    if len(res) == 0:
        return

    res = res.copy()

    res["privacy_leak"] = res["synth_test_orig"] - res["synth_test_real"]
    res["synth_penalty"] = res["orig_test_real"] - res["synth_test_real"]
    style = (
        res.style.format(lambda x: f"{100*x:.1f}%")
        .background_gradient(axis=0)
        .applymap(
            lambda x: "color: transparent; background-color: transparent"
            if pd.isnull(x)
            else ""
        )
    )

    mlflow.log_text(
        gen_html_table(style),
        f"models/{name}.html" if name != "table" else "models.html",
    )
    # mlflow.log_text(res.to_csv(), f"logs/_raw/models/{name}.csv")


def mlflow_log_model_closure(name: str):
    def closure(res: pd.DataFrame):
        return mlflow_log_model_results(name, res)

    closure.__name__ = f"log_{name}_model_results"
    return closure

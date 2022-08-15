import mlflow
import pandas as pd

# Taken from jupyter, with sans serif added, font-size 12px -> 18px
BASE_STYLE = """<style type="text/css">
  table {
    font-family: sans-serif;
    border: none;
    border-collapse: collapse;
    border-spacing: 0;
    color: black;
    font-size: 18px;
    table-layout: fixed;
  }
  thead {
    border-bottom: 1px solid black;
    vertical-align: bottom;
  }
  tr, th, td {
    text-align: right;
    vertical-align: middle;
    padding: 0.5em 0.5em;
    line-height: normal;
    white-space: normal;
    max-width: none;
    border: none;
  }
  th {
    font-weight: bold;
  }
  tbody tr:nth-child(odd) {
    background: #f5f5f5;
  }
  tbody tr:hover {
    background: rgba(66, 165, 245, 0.2);
  }
</style>
"""


def mlflow_log_model_results(name: str, res: pd.DataFrame):
    if not mlflow.active_run():
        return

    res = res.copy()

    res["privacy_leak"] = res["synth_test_orig"] - res["synth_test_real"]
    res["synth_penalty"] = res["orig_test_real"] - res["synth_test_real"]
    html = (
        res.style.format(lambda x: f"{100*x:.1f}%")
        .background_gradient(axis=0)
        .applymap(
            lambda x: "color: transparent; background-color: transparent"
            if pd.isnull(x)
            else ""
        )
        .to_html()
    )

    mlflow.log_text(BASE_STYLE + html, f"metrics/models/{name}.html")
    mlflow.log_text(res.to_csv(), f"metrics/_raw/models/{name}.csv")


def mlflow_log_model_closure(name: str):
    def closure(res: pd.DataFrame):
        return mlflow_log_model_results(name, res)

    closure.__name__ = f"log_{name}_model_results"
    return closure

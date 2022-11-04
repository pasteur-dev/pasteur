import pandas as pd
import mlflow
from ..mlflow import color_dataframe, gen_html_table, mlflow_log_artifacts

FONT_SIZE = "13px"


def log_kl_mlflow(table: str, ref_name: str, log_artifacts: bool = True, **splits: pd.DataFrame):
    if log_artifacts:
        mlflow_log_artifacts("metrics", "kl", table, **splits)

    kl_formatters = {"kl_norm": {"precision": 3}}
    style = color_dataframe(
        splits,
        idx=["col_j"],
        cols=["col_i"],
        vals=["kl_norm"],
        formatters=kl_formatters,
        split_ref=ref_name,
    )

    fn = f"distr/kl.html" if table == "table" else f"distr/{table}_kl.html"
    mlflow.log_text(gen_html_table(style, FONT_SIZE), fn)


def log_cs_mlflow(table: str, ref_name: str, log_artifacts: bool = True, **splits: pd.DataFrame):
    if log_artifacts:
        mlflow_log_artifacts("metrics", "cs", table, **splits)

    cs_formatters = {
        "X^2": {"precision": 3},
        "p": {"formatter": lambda x: f"{100*x:.1f}"},
    }
    style = color_dataframe(
        splits,
        idx=["col"],
        cols=[],
        vals=["X^2", "p"],
        formatters=cs_formatters,
        split_ref=ref_name,
    )

    fn = f"distr/cs.html" if table == "table" else f"distr/{table}_cs.html"
    mlflow.log_text(gen_html_table(style, FONT_SIZE), fn)

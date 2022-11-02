from io import BytesIO

import pandas as pd
from matplotlib.figure import Figure
from pandas.io.formats.style import Styler

from .styles import use_style

# Taken from jupyter, with sans serif added
BASE_TABLE_STYLE = """<style type="text/css">
  table {
    font-family: sans-serif;
    border: none;
    border-collapse: collapse;
    border-spacing: 0;
    color: black;
    font-size: %s;
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

BASE_TXT_STYLE = """<style type="text/css">
  pre {
    font-family: monospace;
    border: none;
    border-collapse: collapse;
    border-spacing: 0;
    color: rgba(0,0,0,.85);
    font-size: %s;
    table-layout: fixed;
  }
</style>
"""

UTF8_META = '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />'
ARTIFACT_DIR = "_raw"


def gen_html_figure_container(viz: dict[str, Figure]):
    import base64

    style = """
    <style>
        .flex {
            display: flex;
            flex-wrap: wrap;
        }
    </style>
    """

    header = """
    <div class="flex">
    """

    footer = """
    </div>
    """

    img_html = (
        lambda name, x: '<img class="'
        + name
        + '" src="data:image/png;base64,'
        + x
        + '">'
    )

    imgs = []
    for name, fig in viz.items():
        with BytesIO() as buff:
            fig.savefig(buff, format="png")

            buff.seek(0)
            bytes = buff.read()

        bytes_base64 = base64.b64encode(bytes)
        enc = bytes_base64.decode()

        img = img_html(name, enc)
        imgs.append(img)

    return (
        "<html><head>"
        + style
        + "</head><body>"
        + header
        + "\n".join(imgs)
        + footer
        + "</body></html>"
    )


def gen_html_table(table: Styler | pd.DataFrame | str, font_size: str = "18px") -> str:
    return (BASE_TABLE_STYLE % font_size) + (
        table if isinstance(table, str) else table.to_html()
    )


def load_matplotlib_style():
    use_style("mlflow")


def color_dataframe(
    df: pd.DataFrame | dict[str, pd.DataFrame],
    idx: list[str],
    cols: list[str],
    vals: list[str],
    split_ref="tst",
    split_col="split",
    cmap="BrBG",
    cmap_ref="Purples",
    diff_reverse=True,
    formatters: dict[str, dict] | None = None,
):
    """Creates a pivot table with `idx`, `cols`, `vals` fed into the `pandas.pivot()`,
    with an additional column level based on `split_col`.

    The columns that have `split_col` equal to `split_ref` have a `background_color()`
    applied on their values with cmap `cmap_ref`.
    The other columns have a symmetrical cmap `cmap` applied to them based on
    their normalized difference with `split_ref`.

    With `diff_reverse`, the cmap applied on the other columns is flipped.
    That way, the colors used for positive/negative can be flipped.

    `formatters` contains a dict of col to `format()` parameters.
    A col can be one of any in `vals`. A `subset` is calculated based on the
    `col` name and it is placed along with the other parameters in the `format()`
    function.
    """

    # If df is a dictionary, merge into one dataframe with an extra `split_col`
    # column with value the key of the dict
    if isinstance(df, dict):
        dfs = [d.assign(**{split_col: n}) for n, d in df.items()]
        df = pd.concat(dfs)
    else:
        df = df.copy()

    # Rename the split column so it's always first
    old_ref = split_ref
    split_ref = f".{split_ref}"
    df.loc[df[split_col] == old_ref, split_col] = split_ref

    pt = (
        df.pivot(index=idx, columns=[*cols, split_col], values=vals)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    pts = pt.style

    if formatters:
        for col, form in formatters.items():
            pts = pts.format(
                subset=(
                    slice(None),
                    (col, *[slice(None) for _ in range(len(cols) + 1)]),
                ),
                **form,
            )

    # Apply background style to ref columns
    for col in vals:
        pts = pts.background_gradient(
            axis=None,
            subset=(
                slice(None),
                (col, *[slice(None) for _ in range(len(cols))], split_ref),
            ),
            cmap=cmap_ref,
        )

    # Apply background to non-ref columns
    # It is based in the difference between expected value to resulting value
    # red = too low
    # white = same, good
    # white = too high
    df_ref = (
        df[df[split_col] == split_ref]
        .pivot(index=idx, columns=cols, values=vals)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    splits = df[split_col].unique()
    for split in splits:
        if split == split_ref:
            continue

        df_split = (
            df[df[split_col] == split]
            .pivot(index=idx, columns=cols, values=vals)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        df_diff = df_split - df_ref
        if diff_reverse:
            df_diff = -df_diff
        df_norm = df_diff / df_diff.abs().max(axis=0) / 2 + 0.5

        pts = pts.background_gradient(
            axis=None,
            subset=(
                slice(None),
                (*[slice(None) for _ in range(len(cols) + 1)], split),
            ),
            gmap=df_norm.to_numpy(),
            vmin=0,
            vmax=1,
            cmap=cmap,
        )

    return pts


def mlflow_log_as_str(path: str, obj, font_size: str = "16px"):
    from html import escape

    import mlflow

    s = f"{UTF8_META}{BASE_TXT_STYLE % font_size}<pre>{escape(str(obj))}</pre>"
    if mlflow.active_run():
        mlflow.log_text(s, path + ".html")


def mlflow_log_artifacts(*prefix: str, **args):
    import pickle
    from os.path import join
    from tempfile import TemporaryDirectory

    import mlflow

    with TemporaryDirectory() as dir:
        for name, val in args.items():
            fn = join(dir, name + ".pkl")
            with open(fn, "wb") as f:
                pickle.dump(val, f)

        mlflow.log_artifacts(dir, join(ARTIFACT_DIR, *prefix))
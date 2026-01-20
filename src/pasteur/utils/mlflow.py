"""Mlflow utility functions.

@TODO: refactor and clean the functions provided by this module."""

from io import BytesIO
from typing import TYPE_CHECKING

import pandas as pd

from .styles import use_style

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Taken from jupyter, with sans serif added
BASE_TABLE_STYLE = """<style type="text/css">
  body {
    font-family: sans-serif;
  }
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
_SAVE_HTML = True


def gen_html_figure_container(viz: dict[str, "Figure"]):
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


def gen_html_table(table, font_size: str = "18px") -> str:
    table_html = ""
    if isinstance(table, str):
        table_html = table
    elif isinstance(table, dict):
        for name, t in table.items():
            if name:
                table_html += f"<h2>{name.capitalize()}</h2>"
            table_html += t.to_html()
    else:
        table_html = table.to_html()

    return (BASE_TABLE_STYLE % font_size) + (
        table if isinstance(table, str) else table_html
    )


def load_matplotlib_style():
    use_style("mlflow")


def color_dataframe(
    df: pd.DataFrame | dict[str, pd.DataFrame] | list[dict],
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
        splits = list(df.keys())
        df = pd.concat(dfs)
    elif isinstance(df, list):
        df = pd.DataFrame(df)
        splits = list(df[split_col].unique())
    else:
        df = df.copy()
        splits = [split_ref]
        df = df.assign(split_col=split_ref)

    # Pivot but do not sort split column
    pt = (
        pd.pivot_table(
            df, index=idx, columns=[*cols, split_col], values=vals, sort=False
        )
        .sort_index(axis="index")
        .sort_index(
            axis="columns", level=list(range(len(cols) + 1)), sort_remaining=False
        )
    )
    pts = pt.style

    if formatters:
        for col, form in formatters.items():
            pts = pts.format(
                subset=(
                    slice(None),
                    (col, *[slice(None) for _ in range(len(cols) + 1)]),
                ),  # type: ignore
                **form,
            )

    # Apply background style to ref columns
    for col in vals:
        pts = pts.background_gradient(
            axis=None,
            subset=(
                slice(None),
                (col, *[slice(None) for _ in range(len(cols))], split_ref),
            ),  # type: ignore
            cmap=cmap_ref,
        )

    # Apply background to non-ref columns
    # It is based in the difference between expected value to resulting value
    # blue = too low
    # white = same, good
    # copper = too high

    # Get difference of each split to ref column
    pt_ref = pd.pivot_table(
        df[df[split_col] == split_ref], index=idx, columns=cols, values=vals
    )
    pt_diffs = {}
    for split in splits:
        if split == split_ref:
            continue

        pt_split = pd.pivot_table(
            df[df[split_col] == split], index=idx, columns=cols, values=vals
        )

        pt_diff = pt_split - pt_ref
        if diff_reverse:
            pt_diff = -pt_diff
        pt_diffs[split] = pt_diff

    # Find max difference between all columns
    pt_max = pd.concat(pt_diffs).abs().groupby(level=-1).max().max(axis="index")

    # Apply styling based in difference
    for split in splits:
        if split == split_ref:
            continue

        pt_norm = pt_diffs[split] / pt_max / 2 + 0.5
        pts = pts.background_gradient(
            axis=None,
            subset=(
                slice(None),
                (*[slice(None) for _ in range(len(cols) + 1)], split),
            ),  # type: ignore
            gmap=pt_norm.to_numpy(),
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


def mlflow_log_figures(path: str, viz: "Figure | dict[str, Figure]"):
    import matplotlib.pyplot as plt
    import mlflow

    if not mlflow.active_run():
        return

    if isinstance(viz, dict):
        if _SAVE_HTML:
            html = gen_html_figure_container(viz)
            mlflow.log_text(html, f"{path}.html")
        else:
            for i, (n, v) in enumerate(viz.items()):
                mlflow.log_figure(v, f"{path}_{i}_{n}.png")

        for v in viz.values():
            plt.close(v)
    else:
        mlflow.log_figure(viz, f"{path}.png")
        plt.close(viz)


def mlflow_log_hists(table: str, name: str, viz: "Figure | dict[str, Figure]"):
    path_prefix = "histograms/"
    if table != "table":
        path_prefix += f"{table}/"

    if not isinstance(name, str):
        name = "_".join(name)
    name = name.lower()

    full_name = f"{path_prefix}{name}"
    mlflow_log_figures(full_name, viz)


def mlflow_log_perf(**runs: dict[str, float]):
    import math

    import mlflow

    df = pd.DataFrame(runs).reset_index(names="node")
    node_obj = df["node"].str.split(".")

    node_type = node_obj.apply(lambda x: x[0])
    node_view = node_obj.apply(lambda x: x[1] if x[0] == "nodes" else pd.NA)  # type: ignore
    node_pkg = node_obj.apply(lambda x: x[2] if x[0] == "nodes" and len(x) == 4 else pd.NA)  # type: ignore
    node_fun = node_obj.apply(
        lambda x: (x[3] if len(x) == 4 else x[2]) if x[0] == "nodes" else x[1]
    )
    node_df = pd.concat(
        {"node": node_type, "view": node_view, "package": node_pkg, "fun": node_fun},
        axis=1,
    )

    time_df = df.drop(columns=["node"]).map(
        lambda x: (
            f"{int(x // 3600):02d}:{int((x // 60) % 60):02d}:{int(x % 60):02d}.{int((x % 1) * 1000):03d}"
            if not math.isnan(x)
            else "N/A"
        )
    )
    perf_df = pd.concat([node_df, time_df], axis=1).set_index(
        ["node", "view", "package", "fun"]
    ).pivot_table(
        index=["node", "fun"],
        values=list(runs.keys()),
        aggfunc="first",
    )

    mlflow.log_text(gen_html_table(perf_df), "perf.html")

def mlflow_log_energy(**runs: dict[str, pd.DataFrame]):
    import mlflow
    import matplotlib.pyplot as plt
    from pasteur.utils.styles import use_style

    use_style("mlflow")
    plt.rcParams["font.family"] = "monospace"
    
    def compute_energy_kwh(df):
        df_sorted = df.sort_values("timestamp").copy()
        df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"])
        
        power_w = pd.to_numeric(df_sorted["power.draw.average [W]"], errors="coerce")
        dt_seconds = df_sorted["timestamp"].diff().dt.total_seconds()
        default_interval = dt_seconds.median()
        if pd.isna(default_interval):
            default_interval = 0
        dt_hours = dt_seconds.fillna(default_interval) / 3600.0
        energy_kwh = (power_w * dt_hours).sum() / 1000.0
        energy_avg = energy_kwh / (dt_hours.sum())

        power_w_smoothed = power_w.copy()
        power_w_smoothed.index = df_sorted["timestamp"]
        power_w = power_w_smoothed.rolling("10s", min_periods=1).mean().reset_index(drop=True)

        util = pd.to_numeric(df_sorted["utilization.gpu [%]"], errors="coerce")
        util_smoothed = util.copy()
        util_smoothed.index = df_sorted["timestamp"]
        util_rolling = util_smoothed.rolling("10s", min_periods=1).mean().reset_index(drop=True)
        util_avg = util.mean()

        return df_sorted["timestamp"], power_w, energy_kwh, util_rolling, util_avg, energy_avg

    # Plot with separate subplots for clarity
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    start = -1
    end = -1
    for pretty, data in runs.items():
        for dtype, samp in data.items():
            r_start = samp["timestamp"].min()
            if start == -1 or r_start < start:
                start = r_start
            r_end = samp["timestamp"].max()
            if end == -1 or r_end > end:
                end = r_end
    
    use_hours = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 3600.0 > 2.0
    
    max_len = max(map(len, ["ref", *runs.keys()]))

    for pretty, data in runs.items():
        for dtype, samp in data.items():
            is_ref = "ref" in dtype.lower()
            is_eval = "gpu_eval" in dtype.lower()
            
            ts, power, kwh, util, util_avg, energy_avg = compute_energy_kwh(samp)
            wall = (ts - pd.to_datetime(start)).dt.total_seconds() / (3600.0 if use_hours else 60.0)

            name = f"{'real data' if is_ref else pretty:>{max_len}s} { 'Eval.' if is_eval else " Gen."}"
            ax.plot(wall, util, label=f"{name} Util. (avg. {util_avg:.1f}%)")
            ax2.plot(wall, power, label=f"{name} Power (avg. {1000*energy_avg:.0f}W, {kwh:.2f} kWh)")
    
    ax.set_ylabel("GPU Utilization (%)")
    ax.legend(loc="lower center")
    ax.grid(True, linestyle="--", alpha=0.5)

    ax2.set_ylabel("Power draw (W)")
    ax2.set_xlabel("Wall Time (hours)" if use_hours else "Wall Time (minutes)")
    ax2.legend(loc="lower center")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    mlflow_log_figures("energy_usage", fig)
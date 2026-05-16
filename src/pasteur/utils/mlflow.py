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


# CSS / control / script bundle that turns an inlined-SVG page into a
# pannable, scroll-zoomable surface.  Shared by graph SVGs and metric
# multiplots so both flavours have the same navigation affordances.
_ZOOM_CSS = (
    "html,body{margin:0;padding:0;background:#fff}"
    "body{cursor:grab;min-height:100vh}"
    "html.dragging,html.dragging body{cursor:grabbing!important;"
    "user-select:none}"
    "html.dragging *{cursor:grabbing!important}"
    "svg{display:block}"
    "#zoom-ctl{position:fixed;top:8px;right:8px;z-index:1000;"
    "display:flex;gap:4px;background:#fff;border:1px solid #ccc;"
    "border-radius:4px;padding:3px;"
    "font:13px/1 -apple-system,Segoe UI,sans-serif}"
    "#zoom-ctl button{width:26px;height:26px;cursor:pointer;"
    "border:1px solid #ccc;background:#fafafa;border-radius:3px;"
    "padding:0;font:600 14px/1 inherit}"
    "#zoom-ctl button:hover{background:#eee}"
    "#zoom-ctl span{align-self:center;min-width:36px;text-align:center;"
    "color:#666;font-size:11px}"
)

_ZOOM_CTL = (
    '<div id="zoom-ctl">'
    '<button title="Zoom out (or scroll down)" onclick="zoomBy(0.8)">−</button>'
    '<span id="zoom-lvl">100%</span>'
    '<button title="Zoom in (or scroll up)" onclick="zoomBy(1.25)">+</button>'
    '<button title="Reset zoom" onclick="resetZoom()">⤢</button>'
    "</div>"
)

# Cursor-anchored wheel zoom + drag-to-pan, applied to every <svg> in
# the document so it works for both single-SVG and stitched multi-SVG
# pages.
_ZOOM_JS = (
    "<script>(function(){"
    "var svgs=Array.from(document.querySelectorAll('svg'));"
    "if(!svgs.length)return;"
    "var bases=svgs.map(function(s){"
    "var w=s.getAttribute('width')||'';"
    "var h=s.getAttribute('height')||'';"
    "return{w:parseFloat(w),h:parseFloat(h),"
    "unit:(w.match(/[a-z%]+$/i)||['px'])[0]};"
    "});"
    "var scale=1;var lvl=document.getElementById('zoom-lvl');"
    "function apply(){"
    "svgs.forEach(function(s,i){"
    "s.setAttribute('width',(bases[i].w*scale)+bases[i].unit);"
    "s.setAttribute('height',(bases[i].h*scale)+bases[i].unit);"
    "});"
    "lvl.textContent=Math.round(scale*100)+'%';"
    "}"
    "function zoomAt(factor,px,py){"
    "var oldScale=scale;"
    "scale=Math.max(0.05,Math.min(20,scale*factor));"
    "if(scale===oldScale)return;"
    "apply();"
    "var r=scale/oldScale;"
    "window.scrollBy(px*(r-1),py*(r-1));"
    "}"
    "window.zoomBy=function(f){"
    "zoomAt(f,window.scrollX+window.innerWidth/2,"
    "window.scrollY+window.innerHeight/2);"
    "};"
    "window.resetZoom=function(){scale=1;apply();};"
    # Only intercept wheel events that look like a zoom gesture
    # (ctrl/⌘+wheel, or trackpad pinch — browsers signal pinch with
    # ctrlKey).  Plain two-finger swipes fall through so the page
    # scrolls naturally.
    "document.addEventListener('wheel',function(e){"
    "if(e.target.closest('#zoom-ctl'))return;"
    "if(!(e.ctrlKey||e.metaKey))return;"
    "e.preventDefault();"
    "var f=Math.exp(-e.deltaY*0.0015);"
    "zoomAt(f,e.pageX,e.pageY);"
    "},{passive:false});"
    "var drag=null;"
    "document.addEventListener('mousedown',function(e){"
    "if(e.button!==0)return;"
    "if(e.target.closest('#zoom-ctl'))return;"
    "drag={x:e.clientX,y:e.clientY,"
    "sx:window.scrollX,sy:window.scrollY};"
    "document.documentElement.classList.add('dragging');"
    "e.preventDefault();"
    "});"
    "document.addEventListener('mousemove',function(e){"
    "if(!drag)return;"
    "window.scrollTo(drag.sx-(e.clientX-drag.x),"
    "drag.sy-(e.clientY-drag.y));"
    "});"
    "function endDrag(){"
    "if(!drag)return;"
    "drag=null;"
    "document.documentElement.classList.remove('dragging');"
    "}"
    "document.addEventListener('mouseup',endDrag);"
    "document.addEventListener('mouseleave',endDrag);"
    "})();</script>"
)


def strip_svg_preamble(svg) -> str:
    """Strip the <?xml ?> preamble and any DOCTYPE declarations from
    an SVG so it can be safely inlined into an HTML body."""
    if isinstance(svg, bytes):
        svg = svg.decode("utf-8")
    svg = svg.lstrip()
    if svg.startswith("<?xml"):
        svg = svg.split("?>", 1)[1].lstrip()
    while svg.startswith("<!DOCTYPE") or svg.startswith("<!doctype"):
        svg = svg.split(">", 1)[1].lstrip()
    return svg


def wrap_zoom_html(body: str, title: str, extra_css: str = "") -> str:
    """Wrap ``body`` (which contains one or more inlined <svg> blocks)
    in an HTML page with zoom controls, cursor-anchored wheel zoom,
    and drag-to-pan."""
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{title}</title>"
        f"<style>{_ZOOM_CSS}{extra_css}</style></head><body>"
        f"{_ZOOM_CTL}{body}{_ZOOM_JS}</body></html>"
    )


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


class mlflow_log_folder:
    def __init__(self, subdir: str | None = None):
        self.tmp = None
        self.subdir = subdir

    def __enter__(self):
        from tempfile import TemporaryDirectory

        self.tmp = TemporaryDirectory()
        self.dir = self.tmp.__enter__()
        return self.dir

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.tmp:
            return

        import mlflow
        from os.path import join

        mlflow.log_artifacts(self.dir, self.subdir)

        return self.tmp.__exit__(exc_type, exc_value, traceback)


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
    perf_df = (
        pd.concat([node_df, time_df], axis=1)
        .set_index(["node", "view", "package", "fun"])
        .pivot_table(
            index=["node", "fun"],
            values=list(runs.keys()),
            aggfunc="first",
        )
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
        power_w = (
            power_w_smoothed.rolling("10s", min_periods=1).mean().reset_index(drop=True)
        )

        util = pd.to_numeric(df_sorted["utilization.gpu [%]"], errors="coerce")
        util_smoothed = util.copy()
        util_smoothed.index = df_sorted["timestamp"]
        util_rolling = (
            util_smoothed.rolling("10s", min_periods=1).mean().reset_index(drop=True)
        )
        util_avg = util.mean()

        return (
            df_sorted["timestamp"],
            power_w,
            energy_kwh,
            util_rolling,
            util_avg,
            energy_avg,
        )

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
    total_energy = 0.0

    for pretty, data in runs.items():
        color = None

        for dtype, samp in data.items():
            is_ref = "ref" in dtype.lower()
            is_eval = "gpu_eval" in dtype.lower()

            ts, power, kwh, util, util_avg, energy_avg = compute_energy_kwh(samp)
            wall = (ts - pd.to_datetime(start)).dt.total_seconds() / (
                3600.0 if use_hours else 60.0
            )

            REF_COLOR = "#999999"

            name = f"{'ref' if is_ref else pretty:>{max_len}s} { 'Eval' if is_eval else ' Gen'}"
            (line,) = ax.plot(
                wall,
                util,
                # label=f"{name} ({util_avg:.1f}%)",
                color=REF_COLOR if is_ref else color,
                linestyle="--" if is_eval else "-",
            )
            ax2.plot(
                wall,
                power,
                label=f"{name} ({util_avg:.1f}%, {1000*energy_avg:.0f}W, "
                + (f"{kwh:.2f} kWh)" if kwh > 1 else f"{1000*kwh:3.0f} Wh)"),
                color=REF_COLOR if is_ref else color,
                linestyle="--" if is_eval else "-",
            )

            total_energy += kwh

            if not is_ref:
                color = line.get_color()  # Reuse color for both plots

    ax.set_ylabel("GPU Utilization (%)")
    # ax.legend()  # (loc="lower center")
    # ax.grid(True, linestyle="--", alpha=0.5)

    ax2.set_ylabel(f"Power draw (W, total {total_energy:.2f} kWh)")
    ax2.set_xlabel("Wall Time (hours)" if use_hours else "Wall Time (minutes)")
    ax2.legend(ncols=2, loc="lower center")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    mlflow_log_figures("energy_usage", fig)

import pandas as pd
import mlflow


def color_dataframe(
    df: pd.DataFrame | dict[str, pd.DataFrame],
    idx: list[str],
    cols: list[str],
    vals: list[str],
    ref_split="tst",
    split_col="split",
    cmap="BrBG",
    cmap_ref="Purples",
    diff_reverse=True,
    formatters: dict[str, dict] | None = None,
):
    if isinstance(df, dict):
        dfs = [d.assign(split=n) for n, d in df.items()]
        df = pd.concat(dfs)

    pt = (
        df.pivot(index=idx, columns=[*cols, split_col], values=vals)
        .sort_index(0)
        .sort_index(1)
    )
    pts = pt.style

    if formatters:
        for col, form in formatters.items():
            pts = pts.format(
                subset=(
                    slice(None),
                    (col, *[slice(None) for _ in range(len(cols) + 1)]),
                ),
                **form
            )

    # Apply background style to ref columns
    for col in vals:
        pts = pts.background_gradient(
            axis=None,
            subset=(
                slice(None),
                (col, *[slice(None) for _ in range(len(cols))], ref_split),
            ),
            cmap=cmap_ref,
        )

    # Apply background to non-ref columns
    # It is based in the difference between expected value to resulting value
    # red = too low
    # white = same, good
    # white = too high
    df_ref = (
        df[df[split_col] == ref_split]
        .pivot(index=idx, columns=cols, values=vals)
        .sort_index(0)
        .sort_index(1)
    )
    splits = df[split_col].unique()
    for split in splits:
        if split == ref_split:
            continue

        df_split = (
            df[df[split_col] == split]
            .pivot(index=idx, columns=cols, values=vals)
            .sort_index(0)
            .sort_index(1)
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


def log_kl_mlflow(table: str, ref_name: str, **splits: pd.DataFrame):
    pass

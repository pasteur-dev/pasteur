def _gen_sdgym_tables(dir: str):
    import os
    from zipfile import ZipFile

    files = os.listdir(dir)
    names = {
        fn.split("_v")[0].split(".zip")[0].lower(): fn
        for fn in files
        if fn.endswith(".zip")
    }

    tables = []
    views = {}
    for ds_name, ds_fn in names.items():
        with ZipFile(os.path.join(dir, ds_fn), "r") as zip:
            ds_tables = []
            for file in zip.filelist:
                fn = file.filename
                if not fn.endswith(".csv"):
                    continue

                name = fn.split("/")[-1][:-4].lower()
                full_name = f"{ds_name}.{name}"
                ds_tables.append(full_name)

            if len(ds_tables) == 1:
                ds_tables = [f"{ds_name}.table"]

            tables.extend(ds_tables)
            views[f"sdgym_{ds_name}"] = ds_tables
    return views, tables


def _reload_sdgym_tables(dir: str = "raw/sdgym", out_fn: str = "_sdgym.py"):
    assert out_fn
    views, tables = _gen_sdgym_tables(dir)
    with open(out_fn, "w") as f:
        f.write(f"sdgym_views={repr(views)}\nsdgym_tables={repr(tables)}")


if __name__ == "__main__":
    """If this script is ran as as a module, it updates the sdgym constants based on
    the currently downloaded datasets.
    
    The path to the sdgyms zips can be passed as a parameter.
    
    Kedro requires pipeline structures to be predefined, so tables and views have 
    to be hardcoded"""
    import sys
    import os

    if len(sys.argv) < 2:
        path = "raw/sdgym"
    else:
        path = sys.argv[1]
    
    curr_path = os.path.dirname(os.path.realpath(__file__))
    _reload_sdgym_tables(path, os.path.join(curr_path, "sdgym_const.py"))
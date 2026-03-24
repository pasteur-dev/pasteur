# Pasteur

Pasteur is a framework for privacy-preserving synthetic data generation built
on top of Kedro. It supports tabular and relational datasets with configurable
transformation, encoding, synthesis, and evaluation pipelines.

## Environment

Set `AGENT=1` to switch Pasteur to agent-friendly output (disables Rich
tracebacks, progress bars, and interactive formatting; uses plain logging with
full stack traces instead).

```bash
AGENT=1 python -m pasteur <command>
```

## CLI Commands

Run with `python -m pasteur <command>`. Key commands:

- `download --accept <dataset>` — download raw data from source
- `ingest_dataset <dataset>` — ingest raw data into dataset tables (parquet)
- `ingest_view <view>` — ingest dataset tables into denormalized view tables
- `bootstrap <dataset>` — preprocess downloaded datasets that require it
- `sweep <view>.<synth> ...` — run synthesis sweep with parameter grid
- `pipe <pipeline>` — run a specific pipeline

## Project Structure

```
conf/                       Kedro config (base + local overrides)
  base/locations.yml        Default data/raw paths
  local/locations.yml       Local overrides for data paths
raw/                        Downloaded raw data
data/                       Processed data (ds/, view/, synth/, reporting/)
src/
  project/settings.py       Kedro settings, module registration (PASTEUR_MODULES)
  pasteur/
    dataset.py              Dataset base class
    view.py                 View base class
    extras/
      __init__.py           get_recommended_modules() — registers all datasets/views
      datasets/             Dataset implementations
        rfel/               CTU Relational Learning Repository datasets
        adult/              Adult dataset
        mimic/              MIMIC-IV dataset
        eicu/               eICU dataset
        ...
      views/                View implementations (denormalized dataset projections)
        rfel/               RFEL views + parameters_*.yml files
        ...
    kedro/
      pipelines/            Pipeline definitions (dataset.py, views.py, main.py)
      runner/               Custom parallel runner
    utils/
      download.py           Download utilities (wget, s3, relational.fel)
```

## Adding a New Dataset

A dataset has three parts: a **Dataset** class (downloads + ingests raw data),
a **View** class (denormalizes into the shape used by synthesis), and a
**parameters YAML** (declares field types). Look at existing implementations
under `src/pasteur/extras/datasets/` and `src/pasteur/extras/views/` for
reference.

1. **Dataset class** — subclass `Dataset` (or a helper like `RfelDataset`).
   Defines tables, primary keys, raw sources, and a `keys()` method returning
   the top-level entity index. The first table in the `tables` dict is used
   for key generation.
2. **View class** — subclass `View` (or a helper like `RfelView`). Maps view
   tables to dataset table dependencies via `deps`. The `ingest()` method
   joins, denormalizes, and casts types. Every child table should carry the
   top-level key. Cast IDs to `pd.Int64Dtype()`.
3. **Parameters file** — `parameters_<short_name>.yml` next to the view.
   Declares each table's primary key and field types: `id`,
   `id:<table>.<field>` (foreign key), `categorical`, `ordinal`, `numerical`,
   `date`, `datetime`, `time`. Append `?` for nullable.
4. **Register** — import and add both to `get_recommended_datasets()` in
   `src/pasteur/extras/__init__.py`.
5. **Download, ingest, verify**:
   ```bash
   AGENT=1 python -m pasteur download --accept <dataset>
   AGENT=1 python -m pasteur ingest_dataset <dataset>
   AGENT=1 python -m pasteur ingest_view <view>
   ```
   Data lands in `<base_location>/ds/<dataset>/tables/` and
   `<base_location>/view/<view>/tables/`. Check `conf/local/locations.yml`
   for actual paths.

"""Data layer for LITMUS: model discovery and entity generation.

Handles scanning the Kedro catalog for available views/models and
generating entities from synth models or real data.
"""

import json
import logging
import random
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_metadata_params(
    data_dir: Path, view: str, alg: str, version: str
) -> dict:
    """Load algorithm params from the metadata pickle for a given version."""
    meta_path = data_dir / "view" / view / "metadata.pkl" / version / "metadata.pkl"
    if not meta_path.exists():
        return {}
    try:
        import pickle

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        # Merge base alg params with overrides
        params = dict(meta.algs.get(alg, {}))
        params.update(meta.alg_override)
        return params
    except Exception:
        logger.debug(f"Could not load metadata for {view}/{alg}/{version}", exc_info=True)
        return {}


def _prettify_model_versions(
    data_dir: Path, view: str, alg: str, versions: list[str]
) -> list[dict]:
    """Build display names for model versions using differing params.

    Uses the same logic as MLflow's prettify_run_names: only show
    params that differ between versions. Params are shown as
    key=value pairs. Nested dicts are flattened.
    """
    from pasteur.utils.parser import dict_to_flat_params

    if not versions:
        return []

    # Load params for each version
    all_params = {}
    for v in versions:
        raw = _load_metadata_params(data_dir, view, alg, v)
        all_params[v] = dict_to_flat_params(raw)

    # Find params that differ across versions
    ref = all_params[versions[0]]
    differing_keys = set()
    for v in versions[1:]:
        for k in set(list(ref.keys()) + list(all_params[v].keys())):
            if ref.get(k) != all_params[v].get(k):
                differing_keys.add(k)

    result = []
    for v in versions:
        params = all_params[v]
        # Build display parts from differing params only
        parts = []
        for k in sorted(differing_keys):
            val = params.get(k)
            if val is None:
                continue
            # Use short key (last segment after .)
            short_key = k.split(".")[-1] if "." in k else k
            parts.append(f"{short_key}={val}")

        name = " ".join(parts) if parts else ""
        overrides = {
            k.split(".")[-1] if "." in k else k: params[k]
            for k in sorted(differing_keys)
            if k in params
        }
        result.append({
            "version": v,
            "name": name,
            "overrides": overrides,
        })

    return result


def discover_models(data_dir: str | Path) -> dict:
    """Scan the synth data directory for available view/algorithm/version combos.

    Each model version includes a prettified name derived from params
    that differ between versions (matching MLflow naming logic).
    """
    data_dir = Path(data_dir)
    synth_dir = data_dir / "synth"
    result: dict = {"views": {}}

    if not synth_dir.exists():
        logger.warning(f"Synth directory not found: {synth_dir}")
        return result

    for view_dir in sorted(synth_dir.iterdir()):
        if not view_dir.is_dir():
            continue

        view_name = view_dir.name
        view_info: dict = {"models": {}}

        for alg_dir in sorted(view_dir.iterdir()):
            if not alg_dir.is_dir():
                continue

            alg_name = alg_dir.name
            model_pkl_dir = alg_dir / "model.pkl"

            if not model_pkl_dir.exists():
                continue

            # List version directories (timestamps)
            # Only include versions that have actual output data (bst files)
            bst_dir = alg_dir / "bst"
            raw_versions = []
            for version_dir in sorted(
                model_pkl_dir.iterdir(), reverse=True
            ):
                if not version_dir.is_dir():
                    continue
                if not (version_dir / "model.pkl").exists():
                    continue
                # Check that synth output data exists for this version
                v_name = version_dir.name
                if bst_dir.exists():
                    has_data = any(
                        (table_dir / v_name).is_dir()
                        and any((table_dir / v_name).iterdir())
                        for table_dir in bst_dir.iterdir()
                        if table_dir.is_dir()
                    )
                    if not has_data:
                        logger.info(
                            f"Skipping {view_name}/{alg_name}/{v_name}: no output data"
                        )
                        continue
                raw_versions.append(v_name)

            if raw_versions:
                view_info["models"][alg_name] = _prettify_model_versions(
                    data_dir, view_name, alg_name, raw_versions
                )

        if view_info["models"]:
            result["views"][view_name] = view_info

    return result


class EntityGenerator:
    """Generates entities from synth models or real data using the Kedro catalog."""

    def __init__(self, ctx):
        """Initialize with a loaded Kedro context.

        Args:
            ctx: KedroContext from session.load_context()
        """
        self.ctx = ctx
        self.catalog = ctx.catalog
        self._data_cache: dict[str, Any] = {}
        self._encoder_cache: dict[str, Any] = {}
        self._mapping_cache: dict[str, Any] = {}
        self._real_data_cache: dict[str, dict] = {}
        # Background entity precomputation
        self._entity_pools: dict[str, list[dict]] = {}
        self._precompute_started: set[str] = set()
        self._precompute_lock = threading.Lock()

    def _load_json_encoder(self, view: str):
        """Load the JSON encoder for a view, with caching."""
        if view in self._encoder_cache:
            return self._encoder_cache[view]

        dataset_name = f"{view}.enc.json"
        logger.info(f"Loading JSON encoder: {dataset_name}")
        encoder = self.catalog.load(dataset_name)
        self._encoder_cache[view] = encoder
        return encoder

    def _get_mapping(self, view: str):
        """Get the table mapping for entity processing, with caching."""
        if view in self._mapping_cache:
            return self._mapping_cache[view]

        from pasteur.extras.encoders import create_table_mapping, get_top_table

        encoder = self._load_json_encoder(view)
        meta = encoder.get_metadata()
        top_table = meta["top_table"]
        mapping = create_table_mapping(
            top_table, meta["relationships"], meta["attrs"], meta["ctx_attrs"]
        )
        self._mapping_cache[view] = {
            "mapping": mapping,
            "top_table": top_table,
            "relationships": meta["relationships"],
            "attrs": meta["attrs"],
            "ctx_attrs": meta["ctx_attrs"],
        }
        return self._mapping_cache[view]

    def get_view_meta(self, view: str) -> dict:
        """Return table order and date reference info for the frontend."""
        mapping_info = self._get_mapping(view)
        encoder = self._load_json_encoder(view)

        # Table order from encoder attrs
        table_order = list(encoder.attrs.keys())

        # Extract date reference dates from transformers
        date_refs: dict[str, dict[str, str]] = {}
        for table in table_order:
            trn_name = f"{view}.trn.{table}"
            try:
                trn = self.catalog.load(trn_name)
                for col_name, transformer in trn.transformers.items():
                    if hasattr(transformer, "ref") and hasattr(transformer, "year_name"):
                        ref_ts = transformer.ref
                        date_refs.setdefault(table, {})[col_name] = ref_ts.isoformat()
            except Exception:
                logger.debug(f"Could not load transformer for {table}", exc_info=True)

        return {
            "table_order": table_order,
            "top_table": mapping_info["top_table"],
            "date_refs": date_refs,
        }

    def _load_real_data(self, view: str):
        """Load reverse-transformed real data tables for a view, with caching."""
        if view in self._real_data_cache:
            return self._real_data_cache[view]

        # Load the encoded real data from the JSON encoder output
        # The working split data is at {view}.wrk.json
        # But we need decoded tables + ids for process_entity
        # Load bst (baseline transformed), ctx, and ids for the working split
        mapping_info = self._get_mapping(view)
        relationships = mapping_info["relationships"]

        # Find all tables from the relationships
        all_tables = set(relationships.keys())
        for children in relationships.values():
            all_tables.update(children)

        tables = {}
        ids = {}
        ctx: dict[str, dict] = {}

        for t in all_tables:
            try:
                ds_name = f"{view}.wrk.bst_{t}"
                data = self.catalog.load(ds_name)
                if callable(data):
                    data = data()
                tables[t] = data
            except Exception:
                logger.debug(f"Could not load {ds_name}", exc_info=True)
                tables[t] = __import__("pandas").DataFrame()

            try:
                ds_name = f"{view}.wrk.ids_{t}"
                data = self.catalog.load(ds_name)
                if callable(data):
                    data = data()
                ids[t] = data
            except Exception:
                logger.debug(f"Could not load ids for {t}", exc_info=True)

        # Load ctx tables
        for ctx_name in mapping_info["ctx_attrs"]:
            ctx[ctx_name] = {}
            for t in all_tables:
                try:
                    ds_name = f"{view}.wrk.ctx_{t}"
                    data = self.catalog.load(ds_name)
                    if callable(data):
                        # Multi-partition: materialize
                        data = data()
                    if t in data:
                        ctx[ctx_name][t] = data[t]
                except Exception:
                    logger.debug(f"Could not load ctx {ctx_name}/{t}", exc_info=True)

        self._real_data_cache[view] = {
            "tables": tables,
            "ids": ids,
            "ctx": ctx,
        }
        return self._real_data_cache[view]

    def _load_synth_data(self, view: str, alg: str, version: str | None = None):
        """Load pre-generated synthetic data (reverse-transformed tables + ids).

        Uses the same approach as real data: load bst, ctx, ids tables
        from the catalog for the given view.alg combination.
        """
        cache_key = f"{view}.{alg}.{version or 'latest'}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        # Early check: verify output data exists on disk before loading
        if version:
            locations = self.ctx.config_loader.get("locations")
            data_dir = Path(locations.get("base", "data"))
            bst_dir = data_dir / "synth" / view / alg / "bst"
            if bst_dir.exists():
                has_data = any(
                    (table_dir / version).is_dir()
                    for table_dir in bst_dir.iterdir()
                    if table_dir.is_dir()
                )
                if not has_data:
                    raise ValueError(
                        f"No output data for {view}/{alg}/{version}"
                    )

        mapping_info = self._get_mapping(view)
        relationships = mapping_info["relationships"]
        all_tables = set(relationships.keys())
        for children in relationships.values():
            all_tables.update(children)

        # Pin version on all datasets we'll load
        if version:
            from kedro.io.core import Version

            for t in all_tables:
                for prefix in [f"{view}.{alg}.bst_", f"{view}.{alg}.ids_", f"{view}.{alg}.ctx_"]:
                    ds_name = f"{prefix}{t}"
                    ds = self.catalog._datasets.get(ds_name)
                    if ds and hasattr(ds, "_version"):
                        ds._version = Version(load=version, save=None)
                        ds._version_cache.clear()

        tables = {}
        ids = {}
        ctx: dict[str, dict] = {}

        for t in all_tables:
            try:
                data = self.catalog.load(f"{view}.{alg}.bst_{t}")
                if callable(data):
                    data = data()
                tables[t] = data
            except Exception:
                logger.debug(f"Could not load synth bst for {t}", exc_info=True)
                tables[t] = __import__("pandas").DataFrame()

            try:
                data = self.catalog.load(f"{view}.{alg}.ids_{t}")
                if callable(data):
                    data = data()
                ids[t] = data
            except Exception:
                logger.debug(f"Could not load synth ids for {t}", exc_info=True)

        for ctx_name in mapping_info["ctx_attrs"]:
            ctx[ctx_name] = {}
            for t in all_tables:
                try:
                    data = self.catalog.load(f"{view}.{alg}.ctx_{t}")
                    if callable(data):
                        data = data()
                    if isinstance(data, dict) and t in data:
                        ctx[ctx_name][t] = data[t]
                    elif not isinstance(data, dict):
                        ctx[ctx_name][t] = data
                except Exception:
                    logger.debug(f"Could not load synth ctx {ctx_name}/{t}", exc_info=True)

        # Verify we actually loaded some data
        top_table = mapping_info["top_table"]
        if top_table not in ids or len(ids[top_table]) == 0:
            logger.warning(
                f"No entity data loaded for {cache_key}, skipping"
            )
            raise ValueError(f"Empty output for {cache_key}")

        result = {"tables": tables, "ids": ids, "ctx": ctx}
        self._data_cache[cache_key] = result
        return result

    def _get_entity_pool(self, cache_key: str) -> list[dict] | None:
        """Return precomputed entity pool if ready, else None."""
        return self._entity_pools.get(cache_key)

    def _start_precompute(
        self, cache_key: str, mapping_info: dict, data: dict
    ):
        """Start background precomputation of all entities for a dataset.

        No-op if already started or finished for this cache_key.
        Thread-safe: uses a per-key lock to prevent duplicate work.
        """
        # Fast path: already done
        if cache_key in self._entity_pools:
            return

        # Acquire the startup lock to check/set the "in progress" flag
        with self._precompute_lock:
            if cache_key in self._entity_pools or cache_key in self._precompute_started:
                return
            self._precompute_started.add(cache_key)

        def _worker():
            import time

            from pasteur.extras.encoders import process_entity

            t0 = time.perf_counter()
            top_table = mapping_info["top_table"]
            ids = data["ids"]
            tables = data["tables"]
            ctx = data["ctx"]

            if top_table not in ids or len(ids[top_table]) == 0:
                self._entity_pools[cache_key] = []
                return

            all_ids = list(ids[top_table].index)
            mapping = mapping_info["mapping"]
            pool = []
            for eid in all_ids:
                try:
                    pool.append(
                        process_entity(top_table, eid, mapping, tables, ctx, ids)
                    )
                except Exception:
                    logger.debug(f"Failed to process entity {eid}", exc_info=True)

            elapsed = time.perf_counter() - t0
            logger.info(
                f"Precomputed {len(pool)} entities for {cache_key} in {elapsed:.2f}s"
            )
            self._entity_pools[cache_key] = pool

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _generate_one(self, mapping_info: dict, data: dict) -> dict:
        """Generate a single entity on-demand (fallback while pool builds)."""
        from pasteur.extras.encoders import process_entity

        top_table = mapping_info["top_table"]
        ids = data["ids"]
        if top_table not in ids or len(ids[top_table]) == 0:
            return {"error": "No data available"}

        all_ids = list(ids[top_table].index)
        return process_entity(
            top_table,
            random.choice(all_ids),
            mapping_info["mapping"],
            data["tables"],
            data["ctx"],
            ids,
        )

    def generate_entity_from_model(
        self, view: str, alg: str, version: str | None = None
    ) -> dict:
        """Pick a random synthetic entity. Serves from cache if ready,
        otherwise generates on-demand and starts background precomputation."""
        mapping_info = self._get_mapping(view)
        synth_data = self._load_synth_data(view, alg, version)
        cache_key = f"synth.{view}.{alg}.{version or 'latest'}"

        # Start background precompute (no-op if already running/done)
        self._start_precompute(cache_key, mapping_info, synth_data)

        # Serve from pool if ready, otherwise generate on-demand
        pool = self._get_entity_pool(cache_key)
        if pool:
            return random.choice(pool)
        return self._generate_one(mapping_info, synth_data)

    def generate_entity_from_real(self, view: str) -> dict:
        """Pick a random real entity. Serves from cache if ready,
        otherwise generates on-demand and starts background precomputation."""
        mapping_info = self._get_mapping(view)
        real_data = self._load_real_data(view)
        cache_key = f"real.{view}"

        self._start_precompute(cache_key, mapping_info, real_data)

        pool = self._get_entity_pool(cache_key)
        if pool:
            return random.choice(pool)
        return self._generate_one(mapping_info, real_data)

    def generate_entity_by_index(
        self, view: str, alg: str | None, version: str | None, index: int
    ) -> dict:
        """Get a specific entity by index. Deterministic across calls.

        Uses the precomputed pool (waits if still building). The index
        is modulo'd by pool size so any large index works.
        """
        if alg is None:
            # Real data
            mapping_info = self._get_mapping(view)
            data = self._load_real_data(view)
            cache_key = f"real.{view}"
        else:
            mapping_info = self._get_mapping(view)
            data = self._load_synth_data(view, alg, version)
            cache_key = f"synth.{view}.{alg}.{version or 'latest'}"

        # Start precompute (no-op if already running/done)
        self._start_precompute(cache_key, mapping_info, data)

        # Try to serve from pool
        pool = self._get_entity_pool(cache_key)
        if pool:
            return pool[index % len(pool)]

        # Pool not ready yet — generate on demand using the index to pick a
        # specific entity ID (deterministic)
        from pasteur.extras.encoders import process_entity

        top_table = mapping_info["top_table"]
        ids = data["ids"]
        if top_table not in ids or len(ids[top_table]) == 0:
            return {"error": "No data available"}

        all_ids = list(ids[top_table].index)
        eid = all_ids[index % len(all_ids)]
        return process_entity(
            top_table, eid, mapping_info["mapping"],
            data["tables"], data["ctx"], ids,
        )

    def generate_entity_json(
        self, view: str, alg: str | None = None, version: str | None = None
    ) -> tuple[dict, str]:
        """Generate an entity and return it as (entity_dict, json_string).

        Args:
            view: View name
            alg: Algorithm name, or None for real data
            version: Specific model version, or None for latest

        Returns:
            Tuple of (entity_dict, pretty_json_string)
        """
        if alg is None:
            entity = self.generate_entity_from_real(view)
            source = "real"
        else:
            entity = self.generate_entity_from_model(view, alg, version)
            source = f"{alg}_{version or 'latest'}"

        json_str = json.dumps(entity, indent=2)
        return entity, json_str, source

    def load_llm_scores(
        self, view: str, alg: str, version: str | None = None
    ) -> tuple[dict[int, int] | None, dict[int, int] | None]:
        """Load LLM evaluator scores for a model version.

        Returns (syn_distribution, ref_distribution) where each is
        {1: count, 2: count, ...} or None if unavailable.
        """
        import pickle
        from pathlib import Path

        data_dir = Path(self.ctx.config_loader.get("locations").get("base", "data"))
        llm_dir = data_dir / "synth" / view / alg / "msr" / "llmeval" / "pre.pkl"

        if not llm_dir.exists():
            return None, None

        if version:
            pkl_path = llm_dir / version / "pre.pkl"
        else:
            versions = sorted(llm_dir.iterdir(), reverse=True)
            if not versions:
                return None, None
            pkl_path = versions[0] / "pre.pkl"

        if not pkl_path.exists():
            return None, None

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            syn = (
                {i + 1: int(data.syn[i]) for i in range(5)}
                if data.syn is not None
                else None
            )
            ref = (
                {i + 1: int(data.ref[i]) for i in range(5)}
                if data.ref is not None
                else None
            )
            return syn, ref
        except Exception:
            logger.debug(f"Could not load LLM scores for {view}/{alg}/{version}", exc_info=True)

        return None, None

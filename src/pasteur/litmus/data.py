"""Data layer for LITMUS: model discovery and entity generation.

Handles scanning the Kedro catalog for available views/models and
generating entities from synth models or real data.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def discover_models(data_dir: str | Path) -> dict:
    """Scan the synth data directory for available view/algorithm/version combos.

    Returns a dict structured as:
    {
        "views": {
            "rfel_sl": {
                "models": {
                    "mare": [
                        {"version": "2026-01-22T09.23.40.216Z"},
                        {"version": "2026-01-21T20.16.57.883Z"},
                    ],
                    "amalgam": [...],
                }
            }
        }
    }
    """
    synth_dir = Path(data_dir) / "synth"
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
            versions = []
            for version_dir in sorted(
                model_pkl_dir.iterdir(), reverse=True
            ):
                if not version_dir.is_dir():
                    continue
                # Verify model.pkl exists in this version
                if (version_dir / "model.pkl").exists():
                    versions.append({"version": version_dir.name})

            if versions:
                view_info["models"][alg_name] = versions

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
        self._model_cache: dict[str, Any] = {}
        self._encoder_cache: dict[str, Any] = {}
        self._mapping_cache: dict[str, Any] = {}
        self._real_data_cache: dict[str, dict] = {}

    def _load_model(self, view: str, alg: str, version: str | None = None):
        """Load a synth model from catalog, with caching."""
        cache_key = f"{view}.{alg}.{version or 'latest'}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        dataset_name = f"{view}.{alg}.model"
        logger.info(f"Loading model: {dataset_name}")

        # If a specific version is requested, set it on the dataset
        if version:
            ds = self.catalog._datasets.get(dataset_name)
            if ds and hasattr(ds, "_version"):
                from kedro.io.core import Version

                ds._version = Version(load=version, save=None)
                ds._version_cache.clear()

        model = self.catalog.load(dataset_name)
        self._model_cache[cache_key] = model
        return model

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

    def generate_entity_from_model(
        self, view: str, alg: str, version: str | None = None
    ) -> dict:
        """Generate a single entity from a synth model.

        Returns a dict representing the entity in human-readable form.
        """
        model = self._load_model(view, alg, version)

        # Generate one entity
        raw_output = model.sample_partition(n=1, i=random.randint(0, 10000))

        # raw_output is dict[str, DataFrame] for simple models
        # or dict[str, dict[str, DataFrame]] for multi-type models
        # We need to decode it through the encoder
        return self._decode_synth_output(view, raw_output)

    def _decode_synth_output(self, view: str, raw_output: dict) -> dict:
        """Decode raw synth output into a human-readable entity dict."""
        from pasteur.extras.encoders import (
            create_table_mapping,
            get_top_table,
            process_entity,
        )

        encoder = self._load_json_encoder(view)

        # Decode through the encoder
        decoded = encoder.decode(raw_output)

        # decoded is a set of closures; materialize them
        # Each closure returns {"ids": {...}, "data": {...}}
        all_data = {}
        for closure in decoded:
            result = closure()
            for key, partitions in result.items():
                if key not in all_data:
                    all_data[key] = {}
                all_data[key].update(partitions)

        # The "data" partitions contain the JSON-encoded entities
        # Each partition value is a DataFrame with stringified dicts
        # Parse the first entity
        if "data" in all_data:
            for _pid, df in all_data["data"].items():
                if len(df) > 0:
                    entity_str = df.iloc[0, 0]
                    if isinstance(entity_str, str):
                        return json.loads(entity_str.replace("'", '"'))
                    return dict(entity_str)

        return {"error": "No entity generated"}

    def generate_entity_from_real(self, view: str) -> dict:
        """Sample a random real entity from the working split.

        Returns a dict representing the entity in human-readable form.
        """
        from pasteur.extras.encoders import get_top_table, process_entity

        mapping_info = self._get_mapping(view)
        real_data = self._load_real_data(view)

        top_table = mapping_info["top_table"]
        ids = real_data["ids"]
        tables = real_data["tables"]
        ctx = real_data["ctx"]

        # Pick a random entity from the top table
        if top_table not in ids or len(ids[top_table]) == 0:
            return {"error": "No real data available"}

        # Get all entity IDs from the top-level table
        all_ids = list(ids[top_table].index)
        entity_id = random.choice(all_ids)

        return process_entity(
            top_table,
            entity_id,
            mapping_info["mapping"],
            tables,
            ctx,
            ids,
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


def fixup_partial_json(data: str) -> str | None:
    """Apply bracket-matching fix-up to make partial JSON valid.

    Ported from Amalgam's _printer() logic. Returns a pretty-printed
    JSON string, or None if the partial cannot be parsed.
    """
    if not data:
        return None

    # Track open brackets to build a closing suffix
    suffix = ""
    for char in data:
        if char == "{":
            suffix += "}"
        elif char == "[":
            suffix += "]"
        elif char == '"':
            if suffix.endswith('"'):
                suffix = suffix[:-1]
            else:
                suffix += '"'
        elif char == "}":
            if suffix.endswith("}"):
                suffix = suffix[:-1]
        elif char == "]":
            if suffix.endswith("]"):
                suffix = suffix[:-1]

    suffix = suffix[::-1]  # reverse

    sdata = data.rstrip()
    if sdata.endswith(","):
        full = data[:-1] + suffix
    elif sdata.endswith(":"):
        full = data + " null" + suffix
    else:
        full = data + suffix

    try:
        obj, _ = json.JSONDecoder().raw_decode(full)
        return json.dumps(obj, indent=2)
    except json.JSONDecodeError:
        return None

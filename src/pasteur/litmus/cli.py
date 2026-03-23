"""Click CLI command for starting the LITMUS server."""

import logging

import click

logger = logging.getLogger(__name__)


def _discover_catalog(ctx) -> dict:
    """Discover available views and models from the Kedro context."""
    from pasteur.module import get_module_dict
    from pasteur.synth import SynthFactory
    from pasteur.view import View

    pasteur_hook = ctx.pasteur
    views = get_module_dict(View, pasteur_hook.modules)
    algorithms = get_module_dict(SynthFactory, pasteur_hook.modules)

    catalog_info: dict = {"views": {}}

    for view_name, view in views.items():
        view_info: dict = {
            "tables": list(view.tables),
            "models": {},
        }

        for alg_name in algorithms:
            # Check if this view-algorithm combination has a model
            model_name = f"{view_name}.{alg_name}.model"
            try:
                if ctx.catalog.exists(model_name):
                    # Try to get version info
                    ds = ctx.catalog._datasets.get(model_name)
                    versions = []
                    if ds and hasattr(ds, "_version"):
                        v = ds._version
                        if v and v.load:
                            versions.append(v.load)

                    view_info["models"][alg_name] = {
                        "algorithm": alg_name,
                        "versions": versions,
                    }
            except Exception:
                logger.debug(
                    f"Could not check model {model_name}", exc_info=True
                )

        if view_info["models"]:
            catalog_info["views"][view_name] = view_info

    return catalog_info


@click.command()
@click.option("--port", "-p", type=int, default=5000, help="Port to serve on.")
@click.option(
    "--host", "-h", type=str, default="127.0.0.1", help="Host to bind to."
)
def litmus(port: int, host: str):
    """Start the LITMUS evaluation server for blinded human evaluation of synthetic data."""
    from kedro.framework.session import KedroSession

    from .app import create_app

    with KedroSession.create(env="base") as session:
        ctx = session.load_context()

        # Discover available views and models
        catalog_info = _discover_catalog(ctx)
        view_count = len(catalog_info.get("views", {}))
        model_count = sum(
            len(v.get("models", {}))
            for v in catalog_info.get("views", {}).values()
        )
        logger.info(
            f"Discovered {view_count} views with {model_count} model(s)"
        )

        # Determine data directory for experiment persistence
        locations = ctx.config_loader.get("locations")
        data_dir = locations.get("base", "data")

        # Store the kedro context for later use by data loading
        app = create_app(data_dir=data_dir, catalog_info=catalog_info)
        app.config["kedro_context"] = ctx

        logger.info(f"Starting LITMUS server on http://{host}:{port}")
        app.run(host=host, port=port, debug=True, use_reloader=False)

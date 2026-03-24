"""Click CLI command for starting the LITMUS server."""

import logging

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option("--port", "-p", type=int, default=5555, help="Port to serve on.")
@click.option(
    "--host", "-h", type=str, default="127.0.0.1", help="Host to bind to."
)
@click.option(
    "--hotreload",
    is_flag=True,
    help="Enable Flask hot-reload (watches for code changes).",
)
def litmus(port: int, host: str, hotreload: bool):
    """Start the LITMUS evaluation server for blinded human evaluation of synthetic data."""
    from kedro.framework.session import KedroSession

    from .app import create_app
    from .data import EntityGenerator, discover_models

    with KedroSession.create() as session:
        ctx = session.load_context()

        # Determine data directory using the same logic as the Pasteur hook
        patterns = getattr(ctx.config_loader, "config_patterns", {})
        if "locations" not in patterns:
            patterns["locations"] = ["location*", "location*/**", "**/location*"]
        locations = ctx.config_loader.get("locations")

        # Allow local overrides without duplicate key errors
        if "hidden_base" in locations:
            locations["base"] = locations.pop("hidden_base")
        if "hidden_raw" in locations:
            locations["raw"] = locations.pop("hidden_raw")

        data_dir = locations.get("base", "data")

        # Discover available views and models from the filesystem
        catalog_info = discover_models(data_dir)
        view_count = len(catalog_info.get("views", {}))
        model_count = sum(
            len(v.get("models", {}))
            for v in catalog_info.get("views", {}).values()
        )
        logger.info(
            f"Discovered {view_count} views with {model_count} algorithm(s)"
        )

        # Create entity generator bound to the Kedro catalog
        generator = EntityGenerator(ctx)

        # Create and start Flask app
        app = create_app(
            data_dir=data_dir,
            catalog_info=catalog_info,
            generator=generator,
        )

        logger.info(f"Starting LITMUS server on http://{host}:{port}")
        app.run(
            host=host,
            port=port,
            debug=True,
            use_reloader=hotreload,
        )

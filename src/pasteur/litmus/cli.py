"""Click CLI command for starting the LITMUS server."""

import logging

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option("--port", "-p", type=int, default=5000, help="Port to serve on.")
@click.option(
    "--host", "-h", type=str, default="127.0.0.1", help="Host to bind to."
)
def litmus(port: int, host: str):
    """Start the LITMUS evaluation server for blinded human evaluation of synthetic data."""
    from kedro.framework.session import KedroSession

    from .app import create_app
    from .data import EntityGenerator, discover_models

    with KedroSession.create(env="base") as session:
        ctx = session.load_context()

        # Determine data directory
        locations = ctx.config_loader.get("locations")
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
        app.run(host=host, port=port, debug=True, use_reloader=False)

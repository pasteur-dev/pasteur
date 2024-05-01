"""Pasteur file for ensuring the package is executable
as `pasteur` and `python -m pasteur`
"""


def main():
    import os

    print(
        f"Overriding kedro config with '{os.path.join(os.curdir, 'conf/base/logging.yml')}'"
    )
    os.environ["KEDRO_LOGGING_CONFIG"] = os.path.join(
        os.curdir, "conf/base/logging.yml"
    )
    from kedro.framework.cli import main as kedro_main

    return kedro_main()


if __name__ == "__main__":
    main()

__all__ = ["main"]

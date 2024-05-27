"""Pasteur file for ensuring the package is executable
as `pasteur` and `python -m pasteur`
"""


def main():
    import os

    log_fn = os.path.join(os.curdir, "conf/base/logging.yml")
    if os.path.exists(log_fn):
        print(f"Overriding kedro config with '{log_fn}'")
        os.environ["KEDRO_LOGGING_CONFIG"] = log_fn
    from kedro.framework.cli import main as kedro_main

    return kedro_main()


if __name__ == "__main__":
    main()

__all__ = ["main"]

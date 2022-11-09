"""Pasteur file for ensuring the package is executable
as `pasteur` and `python -m pasteur`
"""
 
from kedro.framework.cli import main

if __name__ == "__main__":
    main()

__all__ = ["main"]
<h1 align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="./res/logo/logo_text_dark.svg" width="90%">
        <source media="(prefers-color-scheme: light)" srcset="./res/logo/logo_text_light.svg" width="90%">
        <img alt="Pasteur Logo with text. Tagline reads: 'Sanitize Your Data'" src="./res/logo/logo_text_light.svg" width="90%">
    </picture>
</h1>

Pasteur is a system for data synthesis.
This readme is under construction.

## Reproducibility
You can find the experiment files that can be used to reproduce the paper
about Pasteur [here](https://github.com/pasteur-dev/pasteur/tree/paper/notebooks/paper).

## Usage
You can install Pasteur with pip.
```bash
pip install pasteur
```

By default, Pasteur is disabled. To use it with a kedro project, add the
`PASTEUR_MODULES` keyword in your `settings.py` file, which will enable it.
```python
from pasteur.extras import get_recommended_modules

PASTEUR_MODULES = get_recommended_modules()
```

Currently, there does not exist a template project from which to start upon.
This repository is a working Pasteur project and is what was used to develop it.
The module `./src/project` is a kedro project with configs in `./conf` and it is 
the one that was used to develop pasteur.
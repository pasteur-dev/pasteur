<h1 align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pasteur-dev/pasteur/master/res/logo/text_dark.svg" width="90%">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/pasteur-dev/pasteur/master/res/logo/text_light.svg" width="90%">
        <img alt="Pasteur Logo with text. Tagline reads: 'Sanitize Your Data'" src="https://raw.githubusercontent.com/pasteur-dev/pasteur/master/res/logo/text_light.svg" width="90%">
    </picture>
</h1>

[![PyPI package version](https://badge.fury.io/py/pasteur.svg)](https://pypi.org/project/pasteur/)
[![Docs build status](https://readthedocs.org/projects/pasteur/badge/?version=latest)](https://docs.pasteur.dev/)
[![License is Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-informational.svg)](https://opensource.org/license/apache2-0-php/)
[![Python version 3.10+](https://img.shields.io/badge/python-3.10%2B-informational.svg)](https://pypi.org/project/pasteur/)
[![Code style is Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
<!-- [![]()]() -->

Pasteur is a library for managing the end-to-end process of data synthesis.
Gather your raw data and preprocess, synthesize, and evaluate it within a single
project.
Use the tools you're familiar with: numpy, pandas, scikit-learn, scipy or any other.
When your dataset grows, scale to out-of-core data by using Pasteur's parallelization 
and partitioning primitives, without code changes or using different libraries.

Pasteur focuses on providing a common platform for the processing, evaluation and 
sharing of synthetic data.
In the current version, Pasteur can ingest and encode arbitrary multi-table
hierarchical/sequential datasets with a mixture of numerical, categorical, and date values
into a common format for synthesis, through a flexible metadata and encoding system.
Post synthesis, Pasteur can evaluate the resulting data through a multi-table
native, extensible evaluation architecture (with built-in support for basic metrics
such as histograms) and allows for comparison to "ideal" synthetic data, through the
use of a hold-out reference set, which it also creates and manages.
 
Pasteur features built-in support for synthesizing data using PrivBayes, AIM, or MST
(due to the lack of viable multi-table synthesis algorithms).
If not, or if a custom algorithm should be used, it is trivial to add support for
it to Pasteur, by implementing the `Synth` interface.

>
> Pasteur is currently an early research alpha. It is architected to allow multi-modal
> data synthesis (e.g., the combination of hierarchical data with sounds and images)
> and will soon feature a novel synthesis method for hierarchical/event-based data.
>

## Usage
You can install Pasteur with pip.
```bash
pip install pasteur
```

Following, you can create a Pasteur project with:
```bash
pasteur new --starter=pasteur
```

The `pasteur` command is aliased to `kedro`, so you can use them interchangeably.
Within your new project, you can now begin working with Pasteur.
Create a virtual environment to install the project's dependencies.
```bash
# Create a Virtual environment
cd <myproject>
python3.10 -m venv venv # Python 3.10+ required.
source venv/bin/activate
# Freeze your dependencies to allow reproducible installs between colleagues
# and install the default project dependencies.
pip install pip-tools
pip-compile --resolver=backtracking
pip install -r requirements.txt
```
You can now download and synthesize datasets!
```bash
# Download adult and perform 1-time actions (bootstrap; unzipping adult.zip)
pasteur download --accept adult
pasteur bootstrap adult
# Ingest the dataset, then the view that's derived from it and finally run
# synthesis using privbayes
pasteur ingest_dataset adult
pasteur ingest_view tab_adult
pasteur pipe tab_adult.privbayes
```

Access Kedro viz and mlflow to preview runs and quality reports:
```bash
kedro viz
mlflow ui --backend-store-uri data/reporting/flow/ 
```

### Using Jupyter
Use jupyter to iterate on new prototypes.

First, to eliminate path issues depending from where you launch jupyter,
fill in `conf/local/globals.yml with absolute paths for your raw and data directories:
```
raw_location: <project_dir>/raw
base_location: <project_dir>/data
```

Then, create a notebook in the `./notebooks` directory and open it with jupyter
or VS Code.

The first cell of your notebook should contain a variation of the following:
```python
# Star notation allows importing type hints for kedro variables (catalog, pipeline, etc)
# along with the register_kedro function.
from pasteur.kedro.ipython import *

# Import data analysis libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add autoreload to allow iterating on your modules
%load_ext autoreload
%autoreload 2

# Register your project as a `kedro` project to access the catalog
register_kedro()
```

From then on, you may access the `kedro` catalog and experiment with variables.
```python
# Access lazy dataframes and load them by calling them
raw_train = catalog.load('adult.raw@train')()
wrk = catalog.load('tab_adult.wrk.table')()
enc = catalog.load('tab_adult.wrk.idx')['table']()
print(enc.head())

# Access pasteur modules like transformers, encoders, and synth models after synthesis to debug their attributes
print(catalog.load('tab_adult.trn.table').get_attributes())
print(catalog.load('tab_adult.privbayes.model'))

# Unsure of whether one of your modules works?
# Import it and test it in the notebook by pulling intermediary artifacts from the catalog
# Here, we build the transformer for the `workclass` column and apply it to it
# to see if the transformation works and is reversible
from pasteur.extras.transformers import IdxTransformer

data = wrk['workclass']
t = IdxTransformer.get_factory().build()
t.fit(data)

workclass_trn = t.transform(data)
workclass_rev = t.reverse(workclass_trn)
workclass_comp = pd.concat([data, workclass_trn, workclass_rev], axis=1)
print(workclass_comp)
```


## Contributing
To contribute, clone this repository and install the frozen requirements.
```bash
git clone github.com/pasteur-dev/pasteur pasteur

cd pasteur
python3.10 -m venv venv # Python 3.10+ required.
source venv/bin/activate
pip install -r requirements.txt
```

The requirements file installs Pasteur from this repository in an editable
state, so you can begin modifying files.
The requirements file can be regenerated with the following commands, which
will pull the latest version of packages.
To ensure interoperability with other packages, Pasteur does not specify narrow
ranges for supported package versions, which might cause issues for certain version
combinations.
```bash
rm requirements.txt
pip-compile --resolver=backtracking
```

This repository is a Pasteur project used for testing. 
You can start testing Pasteur by running commands.
```bash
pasteur download --accept adult
pasteur ingest_dataset adult
pasteur ingest_view tab_adult
pasteur pipe tab_adult.privbayes
```

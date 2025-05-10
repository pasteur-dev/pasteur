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

Pasteur is a library for managing the end-to-end process of structured data synthesis.
It features the algorithms MARE, PrivBayes, AIM, or MST to produce synthetic data
and contains a variety of evaluation metrics and transformation tools for data.
In addition, a collection of premade datasets is included, focusing on MIMIC-IV.

# Example Usage

## Preparing the work area
On the same directory, clone the following repositories:
```bash
git clone https://github.com/pasteur-dev/pasteur
# For privacy metrics, fork with a small change to allow calling the library
git clone https://github.com/antheas/syntheval
```

Then, install the dependencies in the same virtual environment:
```bash
# In the pasteur folder
cd pasteur
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install -e ../syntheval

# todo: include in the requirements
pip install tabulate
```

You can now place your data in the `raw/` folder of the pasteur directory.

### Testing the Adult Dataset
Adult is a tabular dataset that can be used to test synthesis works.

```bash
# Download and unzip
pasteur download --accept adult
pasteur bootstrap adult

# Ingest the datasets and views created from those
pasteur ingest_dataset adult
pasteur ingest_view tab_adult

# Run synthesis executions
## Normal hyperparameters
pasteur p tab_adult.privbayes

## Normal hyperparameters + run ingest_view, ingest_dataset
pasteur p tab_adult.privbayes --all

## Sweep (privacy budgets 1,2,5,10,100)
pasteur s tab_adult.privbayes -i i="range(5)" alg.etotal="[1,2,5,10,100][i]" alg.theta="[5,5,10,10,25][i]"
```

### Testing the MIMIC-IV Datasets
MIMIC-IV is a large dataset we can partition in a variety of ways to create
synthetic datasets. You need physionet credentials to download the data.

```bash
# Download (you will be prompted for your credentials)
# takes a while to download
pasteur download --accept mimic_iv
# Data is not zipped, no bootstrap needed

# Ingest the MIMIC-IV tables
# Requires a lot of memory per worker. For 64GB of RAM, 5 workers are ok
# Takes ~45min for 5 workers
pasteur ingest_dataset mimic -w 5

# Below are the commands that run the experiments shown in the MARE paper
# With MIMIC-IV.

#
# MIMIC Core
#

pasteur ingest_view mimic_core

# This is a privacy budget sweep. We also change the PrivBayes theta param
# to accoutn for the larger privacy budget.
pasteur s mimic_core.mare -i i="range(5)" alg.etotal="[1,2,5,10,100][i]" alg.theta="[5,5,10,10,25][i]" -p
# This is the ablation study from the MARE paper. It turns on/off components
# of the algorithm to see how they affect the results.
pasteur s mimic_core.mare -i noh='range(4)' alg.etotal="2" alg.theta='5' alg.no_hist='noh == 0 or noh == 2' alg.no_seq='noh == 0 or noh == 1' -p

#
# MIMIC-ICU
#
pasteur iv mimic_icu 

# Same setup as above.
pasteur s mimic_icu.mare -i i="range(5)" alg.etotal="[1,2,5,10,100][i]" alg.theta="[5,5,10,10,25][i]" -p
pasteur s mimic_icu.mare -i noh='range(4)' alg.etotal="2" alg.theta='5' alg.no_hist='noh == 0 or noh == 2' alg.no_seq='noh == 0 or noh == 1' -p

# You can view the resulting experiments with:
mlflow ui --backend-store-uri data/reporting/flow
```
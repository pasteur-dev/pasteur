# **Reproduce** Synthesizing 1 Billion rows: Pasteur, a system for Scalable and Reproducible data synthesis
This readme explains how to reproduce the experiments from the paper with the
above title.
In its directory, you will find the scripts relevant to executing the experiments.

## Prerequisites

### Python
Pasteur **requires Python 3.10+**, its dev packages and we suggest using virtual
environments.
```bash
python -V # Python 3.10.9

# Ubuntu 22.04
sudo apt install python3 python3-dev python3-venv
# Ubuntu 20.04 and lower
# You can use the deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-dev python3.10-venv
```
### Pasteur + Dependencies
```bash
# Clone and cd
git clone https://github.com/pasteur-dev/pasteur
cd pasteur

# Create a virtual environment + dependencies
python3.10 -v venv venv
source venv/bin/activate

# Install the frozen dependecies as they were used in the paper
pip install -r src/requirements.lock 
# Install pasteur
pip install -e --no-deps ./src
```

### AIM (optional)
Run the following commands to install AIM for its experiments.
```bash
# Clone the required repositories
git clone https://github.com/ryan112358/hdmm-1 external/hdmm
git clone https://github.com/ryan112358/private-pgm external/private-pgm

# Add them to the venv
echo $PWD/external/hdmm/src > venv/lib/python3.10/site-packages/hdmm.pth 
echo $PWD/external/private-pgm/src > venv/lib/python3.10/site-packages/pgm.pth
echo $PWD/external/private-pgm/experiments > venv/lib/python3.10/site-packages/pgm_exp.pth

# Install its dependencies
pip install autodp disjoint_set

# Copy a slightly modified aim.py mechanism to private-pgm
cp notebooks/paper/aim.py external/private-pgm/experiments
```

## Experiments

### Downloading the Datasets
The paper contains Views from 2 datasets: View and MIMIC-IV.
We used MIMIC-IV 1.0 for the code of the paper.
Future versions of Pasteur will use the latest version.
You can download the datasets using a built-in downloader and the following commands:
```bash
pasteur download adult
pasteur download mimic_iv_1_0
```
> Warning: MIMIC requires credentialed access from [physionet](https://physionet.org/content/mimiciv/2.2/).
> You will be asked for your account credentials. 
> Downloader source code: [src/pasteur/extras/download.py](../../src/pasteur/extras/download.py)

### Ingestion
After downloading the datasets, run the following commands to ingest them:
```bash
# The main pasteur command is "pipe", or "p"
# It is a shorter form of "kedro pipeline" that runs the pipelines pasteur has generated
# It supports easier overrides, example (don't use DP for baking):
# pasteur pipe mimic_tab_admissions.privbayes alg.e1=None
# Adding --synth will only run synthetic nodes

# "<dataset>.ingest" runs the nodes for ingesting a dataset.
pasteur pipe adult.ingest
pasteur pipe mimic.ingest

# Likewise, "<view>.ingest" runs all the cacheable nodes of a View
# Apart from pipeline.sh, the experiments assume
# the view "ICU Charts" is ingested all its rows.
# In Pasteur, it is named "mimic_billion".
pasteur pipe mimic_billion.ingest
```

### Sidenote
Experiments ran with Jupyter have been commited with their results as used
in the paper in this folder.
They have been cleaned to remove computer specific metadata with [notebooks/utils/nbstripout](../../notebooks/utils/nbstripout) prior to commiting.
This script was designed to make notebooks diffable and making two executions in
different computers appear the same by removing all changing metadata.

### Memory Usage
Section 3.3: Memory Usage contains a table about memory use (Table 1).
It can be reproduced from the files [formats.ipynb](formats.ipynb), and [formats_save.sh](formats_save.sh).
The easiest way to compare naive to optimised encoding was to load the optimized
artifacts and de-optimise them with `df.astype()`.

Initially, `formats.ipynb` was used for the whole table.
But it was found that `pd.to_csv()` was slow and painted a bad picture.
As of this writing, `pandas` uses its own engine for saving CSV files.
When using PyArrow, CSV saves in a comparable time with Parquet files.
`formats_save.sh` uses Pasteur's export function, found in [src/pasteur/kedro/cli.py](../../src/pasteur/kedro/cli.py).
The underlying implementation uses PyArrow (`df.to_parquet()` uses PyArrow as well).
Time from the message `Starting export of...` was used in the paper.

### Marginal Calculations
The marginal calculations experiments (Table 2 and 3) can be reproduced from
[marginals_inmemory.ipynb](marginals_inmemory.ipynb) and [marginals_throughput.ipynb](marginals_throughput.ipynb).

### Parallelization
The synthesis executions table (Table 4) was generated from [pipeline.sh](pipeline.sh).
Run the following commands to reproduce all of the table.
The results have to be extracted manually from the logs.
The system will print out how many marginals are computed during synthesis
and the combined KL measure for the ref(erence) and w(o)rk sets.

```bash
# Assumes you created a virtual environment in venv
# Otherwise...
# export PASTEUR=pasteur

./notebooks/paper/pipeline.sh tab_adult
./notebooks/paper/pipeline.sh mimic_tab_admissions

./notebooks/paper/pipeline.sh mimic_billion 1M
./notebooks/paper/pipeline.sh mimic_billion 10M
./notebooks/paper/pipeline.sh mimic_billion 100M
./notebooks/paper/pipeline.sh mimic_billion 500M
./notebooks/paper/pipeline.sh mimic_billion 1B

./notebooks/paper/pipeline.sh mimic_billion 500Msingle
```


### Miscellaneous
```bash
# The initial version of the paper used the Texas Dataset table charges for 1B rows
# But was scraped in favor of MIMIC due to low data quality.
# In the version of Pasteur related to the paper you can work with it by:
pasteur download texas # <- not available publicly anymore, requires sending a form
pasteur bootstrap texas # <- Omitted in the paper, performs one time tasks (ex. unzipping)

pasteur p texas.ingest
pasteur p texas_billion.ingest

pasteur p texas_billion.privbayes --synth
```
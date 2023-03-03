<h1 align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="./res/logo/logo_text_dark.svg" width="90%">
        <source media="(prefers-color-scheme: light)" srcset="./res/logo/logo_text_light.svg" width="90%">
        <img alt="Pasteur Logo with text. Tagline reads: 'Sanitize Your Data'" src="./res/logo/logo_text_light.svg" width="90%">
    </picture>
</h1>
Pasteur is a library for performing privacy-aware end-to-end data synthesis.
Gather your raw data and preprocess, synthesize, and evaluate it within a single
project.
Use the tools you're familiar with: numpy, pandas, scikit-learn, scipy or any other.
When your dataset grows, scale to out-of-core data by using Pasteur's parallelization 
and partitioning primitives, without code changes or using different libraries.

## Reproducibility
You can find the experiment files that can be used to reproduce the paper
about Pasteur [here](https://github.com/pasteur-dev/pasteur/tree/paper/notebooks/paper).

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
pip-compile src/requirements.txt src/requirements.lock
pip install -r src/requirements.lock
```
You can now download and synthesize datasets!
```bash
pasteur download --accept adult
pasteur p adult.ingest
pasteur p tab_adult.ingest
pasteur p tab_adult.privbayes --synth
```

Access Kedro viz and mlflow to preview runs and quality reports:
```bash
kedro viz
mlflow ui --backend-store-uri data/reporting/flow/ 
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
pasteur p adult.ingest
pasteur p tab_adult.ingest
pasteur p tab_adult.privbayes --synth
```

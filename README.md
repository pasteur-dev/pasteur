Amalgam is implemented in a fork of the pasteur synthesis system. Here is how to run the experiments of the Amalgam paper:

```bash
#!/bin/bash

set -euo pipefail

python -m venv venv
source venv/bin/activate
pip install -e .

# Download the datasets
# These come directly from the database hosted at https://relational.fel.cvut.cz/
# named as CTUR in the paper instead of RFEL, but the same dataset.
pasteur download rfel_ce --accept # CustomerExpenditures
pasteur download rfel_sl --accept # StudentLoans
pasteur download rfel_fn --accept # Financial Dataset
# mimic requires physionet credentials
pasteur  download mimic_iv --accept

# Initial ingest of the views
pasteur iv rfel_ce --all
pasteur iv rfel_sl --all
pasteur iv rfel_fnc --all
pasteur iv mimic_core --all

ARGS="alg.samples=2000 metrics.llmeval.samples=2000 --all"

pasteur s rfel_ce -a mare -a amalgam $ARGS
pasteur s rfel_sl -a mare -a amalgam $ARGS
pasteur s rfel_fnc -a mare -a amalgam $ARGS

pasteur s mimic_core -a mare -a amalgam $ARGS
pasteur s mimic_core.amalgam $ARGS -p \
    -i i="range(4)" \
     "alg.model.repo_id=[\"unsloth/gemma-3-12b-it-GGUF\", \"unsloth/gpt-oss-20b-GGUF\", \"MaziyarPanahi/Meta-Llama-3.1-8B-Instruct-GGUF\", \"Qwen/Qwen3-8B-GGUF\"  ][i]" \
    "alg.model.filename=[\"gemma-3-12b-it-Q4_K_M.gguf\",  \"gpt-oss-20b-Q4_K_M.gguf\",  \"Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf\",        \"Qwen3-8B-Q4_K_M.gguf\"][i]" \
    "_pretty=[\"Gemma 3 12B It\", \"GPT-Oss 20B\", \"Meta Llama 3.1 8B Instruct\", \"Qwen 3 8B\"][i]"

```
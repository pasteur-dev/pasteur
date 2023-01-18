#!/bin/bash
DEF_DATASET=adult
DEF_VIEW=tab_adult
DEF_PARAMS=

DATASET=${DATASET:-$DEF_DATASET}
VIEW=${VIEW:-$DEF_VIEW}
PARAMS=${PARAMS:-$DEF_PARAMS}

# time venv/bin/pasteur p $DATASET.ingest
time venv/bin/pasteur p $VIEW.ingest $PARAMS

mkdir -p data/csv
venv/bin/pasteur export $VIEW.wrk.idx_table data/csv/$VIEW.csv
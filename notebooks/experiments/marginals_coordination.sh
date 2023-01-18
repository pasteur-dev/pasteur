#!/bin/bash

venv/bin/python notebooks/experiments/marginals_coordination.py --parallel   -w 1 -m 1
# venv/bin/python notebooks/experiments/marginals_coordination.py --sequential -w 1 -m 1 &
# venv/bin/python notebooks/experiments/marginals_coordination.py --inmemory   -w 1 -m 1 &

# for job in `jobs -p`
# do
# echo $job
#     wait $job
# done

# venv/bin/python notebooks/experiments/marginals_coordination.py --parallel   -w 32 -m 1
# venv/bin/python notebooks/experiments/marginals_coordination.py --inmemory   -w 32 -m 1
# venv/bin/python notebooks/experiments/marginals_coordination.py --sequential -w 32 -m 1
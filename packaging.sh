#!/bin/bash
# This sciprt is for packaging use.
#

ZIP_NAME="team_11_code.zip"
zip -r --exclude=*__pycache__* \
    ${ZIP_NAME}  \
    ./sim_annealing/sim_anneal.py \
    ./sim_annealing/main.py \
    ./tabu_search \
    ./genetic_algorithm \
    ./benchmark \
    ./benchmark_res \
    ./analysis \
    ./run_benchmark.sh \
    ./create_venv.sh \
    ./README.md

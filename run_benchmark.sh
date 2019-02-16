#!/bin/bash

# benchmark result folder
BEN_RESULT_FOLDER="./benchmark_res"

# use venv environment
source venv/bin/activate

# mkdir the result directory
mkdir -p ./${BEN_RESULT_FOLDER}

# run the script to create benchmark result
# python3 benchmark/benchmark.py -o <output json> <tabu>

# ================================
# run the benchmark in parallel
# ================================
# avoid the background process to run after this script is killed
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
# run test cases for tabu search
python3 benchmark/benchmark.py -o ./${BEN_RESULT_FOLDER}/result_tabu.json -n 10 -it 300 tabu_search &
# run test cases for sim_annealing
python3 benchmark/benchmark.py -o ./${BEN_RESULT_FOLDER}/result_sa.json -n 10 sim_annealing &
# run test cases for genetic_algorithm
python3 benchmark/benchmark.py -o ./${BEN_RESULT_FOLDER}/result_ga.json -n 10 genetic_algorithm &

# wait until all finish
wait

echo ""
echo "-----------------------------------"
echo "All benchmark jobs are finish!"
echo "-----------------------------------"

# AI_Planning_Project
-------------------------------------
- [x] README.txt including brief file descriptions, and guidelines how to run it

## File Description
-------------------------------------
***/create_venv.sh***   
For creating a python virtual environment.

***/run_benchmark.sh***   
For running the benchmark suite.

***/benchmark***   
benchmark suite, the vrp file parsers and the instance files

***/analysis***   
Tool to calculate the result_xxx.json statistics

***/genetic_algorithm***   
source code of Genetic Algorithm

***/sim_annealing***    
source code of Simulated Annealing

***/tabu_search***   
source code of Tabu Search

## How to Run
-------------------------------------
1. Check your environment (Unix-like & Python 3.4+)   
   Recommend to run the source code in EECS Shell servers  
   See: https://intra.ict.kth.se/en/it-service/shell-servers-ict-1.36363

2. Run create_venv.sh
   Notes: if you want to run in other OS (e.g. Windows),   
   please read ```create_venv.sh``` and create a python virtual enviroment to   
   run the source code.

3. run the main.py inside the algorithm folder to see the basic results   
   For example:
>``` bash
>source venv/bin/activate   # activate python virtual environment
>cd tabu_search
>python3 main.py
>```

4. run the shell script ```run_benchmark.sh```  
   Notes: It takes about two hours to run all the cases. 
5. check the result json file in benchmark_result
6. If you would like to make the statistic csv, please source the virtual   
   environment and run the analysis python script.
>```bash
>source venv/bin/activate
>python3 analysis/make_stat_csv.py
>```

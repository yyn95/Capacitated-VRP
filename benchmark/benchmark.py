import os
from os.path import join, abspath, dirname, basename
import glob
import sys
import json
import time
import argparse
from cvrp_parser import CVRPFileParser


def get_benchmark_insts():
    """
    Get a list of instance file of benchmark CVRP problem

    Args:
        None
    Return:
        a list of instance file absolute path
    """
    # set A
    set_a = glob.glob(
        join(dirname(__file__), "instances/Vrp-Set-A/*.vrp"))

    set_b = glob.glob(
        join(dirname(__file__), "instances/Vrp-Set-B/*.vrp"))

    ret = set_a + set_b
    return ret


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark genration tool")

    parser.add_argument("algo")
    parser.add_argument("-it", dest="iter", type=int, default=100)
    parser.add_argument("-o", dest="ouput_json")
    parser.add_argument("-n", dest="num_sample", type=int, default=5)

    args = parser.parse_args()
    # check the input arguments
    if args.iter < 1:
        raise ValueError("Please input iteration greater than 1!")

    return args

def get_loads(routes, demands):

    loads = [0 for _ in routes]
    for i, route in enumerate(routes):
        for n in route:
            loads[i] += p.demands[n]
    return loads


if __name__ == '__main__':

    # for import algorithm
    sys.path.append(join(dirname(__file__), ".."))

    # get arguments
    args = get_args()

    # get benchmark instance
    benchmark_instances = get_benchmark_insts()

    result = dict()
    result['iteration'] = args.iter
    result['algorithm'] = args.algo

    # each instance, we make 5 runs
    num_runs = args.num_sample

    # case selection of which algo
    if args.algo == "tabu_search":
        # Tabu search
        import tabu_search.tabu as tabu
        tabu.MAX_ITER = args.iter

        for inst_ in benchmark_instances:
            p = CVRPFileParser(inst_)
            p.parse()

            inst_result = dict()
            inst_result['cost'] = []
            inst_result['proc_time'] = []
            inst_result['sol'] = []
            inst_result['loads'] = []
            for _ in range(num_runs):

                stime = time.process_time()
                solution, cost = tabu.tabu_search(
                    capacity=p.capacity, distances=p.distances, demands=p.demands)
                etime = time.process_time()

                # save down the cost and cpu time
                inst_result['cost'].append(cost)
                inst_result['proc_time'].append(etime - stime)
                inst_result['sol'].append(repr(solution))
                inst_result['loads'].append(repr(get_loads(solution, p.demands)))

            result[basename(inst_)] = inst_result
    elif args.algo == "sim_annealing":
        from sim_annealing.sim_anneal import SimulatedAnnealing
        SimulatedAnnealing.MAX_DURATION = 60
        for inst_ in benchmark_instances:
            print(inst_)
            p = CVRPFileParser(inst_)
            p.parse()

            inst_result = dict()
            inst_result['cost'] = []
            inst_result['proc_time'] = []
            inst_result['sol'] = []
            inst_result['loads'] = []
            for _ in range(num_runs):

                stime = time.process_time()
                sa = SimulatedAnnealing(
                    dimension=p.dimension, coordinates=p.coordinates,
                    capacity=p.capacity, distances=p.distances,
                    demands=p.demands)

                solution, cost = sa.solve()

                etime = time.process_time()

                # save down the cost and cpu time
                inst_result['cost'].append(cost)
                inst_result['proc_time'].append(etime - stime)
                inst_result['sol'].append(repr(solution['routes']))
                inst_result['loads'].append(repr(get_loads(solution['routes'], p.demands)))

            result[basename(inst_)] = inst_result

    elif args.algo == "genetic_algorithm":
        import genetic_algorithm.GA_VRP as GA_VRP
        GA_VRP.MAX_ITER = args.iter

        for inst_ in benchmark_instances:
            p = CVRPFileParser(inst_)
            p.parse()

            inst_result = dict()
            inst_result['cost'] = []
            inst_result['proc_time'] = []
            inst_result['sol'] = []
            inst_result['loads'] = []
            while len(inst_result['cost']) < num_runs:  # take until num_runs samples
                # ============================
                # PUT PARAMETER INTO MODULE
                # ============================
                GA_VRP.mod_param_config(
                    city_num=p.dimension,
                    coordinates=p.coordinates,
                    capacity=p.capacity,
                    distances=p.distances,
                    demands=p.demands)

                try:
                    stime = time.process_time()
                    population = GA_VRP.initialization()
                    population = GA_VRP.main_GA(population)
                    best_chromo = population[0]
                    best_trips = best_chromo.trips
                    best_distances = best_chromo.fitness
                    etime = time.process_time()

                    # save down the cost and cpu time
                    inst_result['cost'].append(best_distances)
                    inst_result['proc_time'].append(etime - stime)
                    inst_result['sol'].append(repr(best_trips))
                    inst_result['loads'].append(repr(get_loads(best_trips, p.demands)))
                except:
                    pass

            result[basename(inst_)] = inst_result
    else:
        raise("Unrecognized algorithm!")

    # save the result dictionary to json format for further analysis
    with open(args.ouput_json, 'w') as fp:
        json.dump(result, fp, indent=4)

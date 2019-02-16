#!/usr/bin/env python3
import math
import time
# from tabu import tabu_search
import tabu
# import matplotlib.pyplot as plt
import sys
from os.path import join, dirname
# import CVRPFileParser
sys.path.append(join(dirname(__file__), "../benchmark"))
from cvrp_parser import CVRPFileParser

if __name__ == '__main__':


    # p = CVRPFileParser('../benchmark/instances/Vrp-Set-A/A-n32-k5.vrp')
    p = CVRPFileParser('../benchmark/instances/Vrp-Set-A/A-n38-k5.vrp')
    p.parse()
    capacity = p.capacity
    distances = p.distances
    demands = p.demands
    coordinates = p.coordinates
    print(capacity)
    print(distances)
    print(demands)

    tabu.MAX_ITER = 200

    time_begin = time.process_time()
    solution, distance = tabu.tabu_search(capacity=capacity, distances=distances, demands=demands)
    time_end = time.process_time()

    load = [0 for _ in solution]
    for i, r in enumerate(solution):
        for n in r:
            load[i] += demands[n]

    print('Solution: ', solution)
    print('Distance: ', distance)
    print('load:', load)
    print('CPU time: %.6f' % (time_end - time_begin))

    # for route in solution:
    #     x = [coordinates[city][0] for city in route]
    #     y = [coordinates[city][1] for city in route]
    #     x.append(x[0])
    #     y.append(y[0])

    #     plt.plot(x, y)
    #     plt.scatter(x, y)

    # plt.show()

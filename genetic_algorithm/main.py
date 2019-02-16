import GA_VRP
import time
import matplotlib.pyplot as plt
import math
import sys
from os.path import join, dirname
# import CVRPFileParser
sys.path.append(join(dirname(__file__), "../benchmark"))
from cvrp_parser import CVRPFileParser


if __name__ == '__main__':

    # p = CVRPFileParser('../benchmark/instances/Vrp-Set-A/A-n32-k5.vrp')
    p = CVRPFileParser('../benchmark/instances/Vrp-Set-B/B-n34-k5.vrp')
    p.parse()
    # ============================
    # PUT PARAMETER INTO MODULE
    # ============================
    GA_VRP.mod_param_config(
        city_num=p.dimension,
        coordinates=p.coordinates,
        capacity=p.capacity,
        distances=p.distances,
        demands=p.demands)

    #GA
    time_begin = time.process_time()

    population = GA_VRP.initialization()
    population = GA_VRP.main_GA(population)
    best_chromo = population[0]
    best_trips = best_chromo.trips
    best_distances = best_chromo.fitness

    time_end = time.process_time()
    print('Solution: ', best_trips)
    print('Distance: ', best_distances)
    print('CPU time: %.6f' % (time_end - time_begin))
    load = [0 for _ in best_trips]
    for i, r in enumerate(best_trips):
        for n in r:
            load[i] += p.demands[n]
    print('load:', load)

    for route in best_trips:
        x = [GA_VRP.coordinates[city][0] for city in route]
        y = [GA_VRP.coordinates[city][1] for city in route]

        # add the starting point to the front
        x.insert(0, GA_VRP.coordinates[0][0])
        y.insert(0, GA_VRP.coordinates[0][1])

        # add the starting point to the end
        x.append(GA_VRP.coordinates[0][0])
        y.append(GA_VRP.coordinates[0][1])

        plt.plot(x, y)
        plt.scatter(x, y)
    plt.show()


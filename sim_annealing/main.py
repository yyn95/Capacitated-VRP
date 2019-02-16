from sim_anneal import SimulatedAnnealing
import sys
from os.path import join, dirname
# import CVRPFileParser
sys.path.append(join(dirname(__file__), "../benchmark"))
from cvrp_parser import CVRPFileParser

if __name__ == '__main__':


    p = CVRPFileParser('../benchmark/instances/Vrp-Set-A/A-n38-k5.vrp')
    # p = CVRPFileParser('../benchmark/instances/Vrp-Set-A/A-n34-k5.vrp')
    # p = CVRPFileParser('../benchmark/instances/Vrp-Set-A/A-n32-k5.vrp')
    p.parse()

    sa = SimulatedAnnealing(dimension=p.dimension, coordinates=p.coordinates,
                            capacity=p.capacity, distances=p.distances,
                            demands=p.demands)
    # sa.test_get_fessible_sol()
    sa.solve()
    # sa.test_move_transform()

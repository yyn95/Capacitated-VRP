"""
Author:
    * zhenghli@kth.se
    * chlin3@kth.se

Reference:
    * Short introduction to Simulated Annealing:
        https://am207.github.io/2017/wiki/lab4.html
    * Harmanani, Haidar M., et al. "A Simulated Annealing Algorithm for the
      Capacitated Vehicle Routing Problem." CATA. 2011.
"""
import numpy as np
import copy
import time
from itertools import tee

def pairwise(iterable):
    """
    Return an iterator generates two consecutive elements from the iterable

    Example:
    ```
    for p in pairwise(range(10)):
        print(p)

    # the output will be
    # (0, 1)
    # (1, 2)
    # (2, 3)
    # (3, 4)
    # (4, 5)
    # (5, 6)
    # (6, 7)
    # (7, 8)
    # (8, 9)
    ```
    Args:
        iterable (iterable): a iterable object
    Return
        iterator of a paired tuple
    """
    # create two iterators
    first, second = tee(iterable)
    # make the second iterator step one step
    next(second, None)

    return zip(first,second)


class SimulatedAnnealing:
    """
    A solver for running the simulated annealing algorithm to CVRP probelm
    """

    # ==============================
    # Parameters of SA algorithm
    # =============================
    alpha = 0.92  # Temperature reduction multiplier
    beta = 1.005   # Iteration multiplier
    init_cool_it = 5   # cooling iteration until next parameter update, M_0 in Harmanani
    init_temp = 8000   # initial temperature
    MAX_ITER = 90000
    MAX_DURATION = 60
    freez_temp = 0.001

    # ===============
    # internal param
    # ===============
    # for compute the neigborhood of existing solution
    max_trial = 50   # the number of trials when inserting back to the routes
    num_swap = 2     # the number of swap in move trans. and replace highest average trans.

    def __init__(self, dimension, coordinates, capacity, distances, demands, num_veh=5):
        """
        Constructor of this solver
        Args:
            dimension (int): the dimension of the problem, which means the number
                             of nodes of CVRP problem
            coordinates (list of list): the list of x,y pair, which indicate the
                                        city coordinate
            capacity (int): the capacity of vechicles in CVRP problem
            demands (list): the demands of the nodes
            num_veh (Optional[int]): the number of vechiles could be used in this
                                     problem

        """
        self._dim = dimension
        self._coord = np.array(coordinates)
        self._dist_mat = np.array(distances)
        # each vaehicle capacity
        self._cap = capacity
        self._demands = np.array(demands)
        self._num_veh = num_veh

    def solve(self):
        """
        Sovle the problem which characterized by the class construction

        Return:
            return a tuple of the solution dictionary and the cost of the sol
        """
        # set initial guess
        cur_sol = self.get_fessible_sol()
        cur_cost = self.get_cost_of_sol(cur_sol)

        # set best sol
        best_sol = copy.deepcopy(cur_sol)
        best_cost = cur_cost

        # initalize parameters
        temp = self.init_temp
        cool_iter = self.init_cool_it  # cool_iter: M in Harmanani
        total_iter = 0

        endtime = time.time() + self.MAX_DURATION

        for iter_ in range(self.MAX_ITER):
            if iter_ > 0:
                # update cooling parameters
                temp *= self.alpha
                cool_iter = int(np.ceil(cool_iter * self.beta))

            print("Temperature", temp, "Length", cool_iter)

            # run through the iterations for each iteration
            for it in range(cool_iter):

                # keep track of total proposals
                total_iter += 1

                # get a new proposal and calculate its energy
                new_sol = self.get_neighbor(cur_sol)

                # compute the cost of the new route
                new_sol_cost = self.get_cost_of_sol(new_sol)

                # compute the change of cost fucntion
                delta = new_sol_cost - cur_cost

                if ((delta < 0) or (np.exp(-delta/temp) > np.random.random())):   # if the new route is better
                    # update sol
                    cur_sol = new_sol
                    # update cost
                    cur_cost = new_sol_cost

                    # record down if the updated route is the best
                    if new_sol_cost < best_cost:
                        best_sol = cur_sol
                        best_cost = new_sol_cost

            print('best sol:', best_sol)
            print('total cost of best route: ', best_cost)

            # Stop iteration citeria 1: arrived at the freezing temperature
            if 0 < temp < self.freez_temp:
                break
            # Stop iteration citeria 2: exceed the maximum iteration
            if total_iter > self.MAX_ITER:
                break
            # Stop iteration citeria 3: exceed the run duration
            if time.time() > endtime:
                break

        return best_sol, int(best_cost)

    def get_neighbor(self, sol):
        """
        Get the neighborhood of the current route

        Args:
            route: the input route
        Return:
            the neighborhood of the input route
        """
        # copy the solution since it is a mutable object (dict)
        ret = copy.deepcopy(sol)
        # run move transform
        ret = self._move_transform(ret)
        # run replace highest average transformation
        ret = self._replace_highest_value(ret)

        # inserts the five selected customers in the route with the resulting
        # minimum cost. We here use 2-opt
        ret = self._2opt_transform(ret)

        return ret

    def _move_transform(self, sol):
        """
        Ramdomly <self.num_swap> swap nodes in routes except the deport and the
        end node of the shortest <self.num_swap> edges on the routes

        Args:
            sol (dict): the solution you would like to transform
        Return:
            The transformed result if success, return the input if cannot perform
            transformation.
        """

        # we try to remove the number of swap from the route and insert them
        # back to routes randomly
        ret = copy.deepcopy(sol)

        routes = ret['routes']
        load = ret['load']

        # build the exclusion list first
        excl_nodes = [0]
        edge_dist_on_routes = dict()
        for r in routes:
            for s_idx, e_idx in pairwise(r):
                edge_dist_on_routes[(s_idx, e_idx)] = self._dist_mat[s_idx, e_idx]

        # sort the dictionary by value, record down the second element of key
        for key, value in sorted(edge_dist_on_routes.items(), key=lambda kv: kv[1]):
            if len(excl_nodes) >= self.num_swap: break
            excl_nodes.append(key[1])

        # select nodes to be swapped
        swap_nodes = list()
        while len(swap_nodes) < self.num_swap:
            r_idx = np.random.randint(0, len(routes))

            if (len(routes[r_idx])-1 == 1):
                n_idx = 1
            else:
                n_idx = np.random.randint(1, len(routes[r_idx])-1)

            node = routes[r_idx][n_idx]
            if node not in excl_nodes:
                swap_nodes.append(node)
                # remove the node from the route
                del routes[r_idx][n_idx]
                # remove the load
                load[r_idx] -= self._demands[node]

        # insert swap_nodes back to the route
        ntrial = 0
        while (swap_nodes and ntrial <= self.max_trial): # stop if exceed maximun trails
            node = swap_nodes[0]
            # check the constrain
            r_idx = np.random.randint(0, len(routes))
            dem = self._demands[node]
            if (dem + load[r_idx] <= self._cap):
                # pick a location to insert this node
                if (len(routes[r_idx])-1 == 1):
                    n_idx = 1
                else:
                    n_idx = np.random.randint(1, len(routes[r_idx])-1)

                routes[r_idx].insert(n_idx, node)
                load[r_idx] += self._demands[node]
                swap_nodes.pop(0)  # remove the node in swap since inserted
            ntrial += 1

        # if there is a non-empty swap_nodes, it means the back insertion fails
        # return the input solution back
        if swap_nodes:
            return sol

        return ret

    def _replace_highest_value(self, sol):
        """
        The replace highest average transformation calculates the average
        distance of every pair of customers in the graph.

        Args:
            sol (dict): the solution you would like to transform
        Return:
            The transformed result if success, return the input if cannot perform
            transformation.
        """
        ret = copy.deepcopy(sol)
        routes = ret['routes']
        load = ret['load']

        # calculate the avg distacne of every vertex
        avg_dist = dict()
        for r in routes:
            for i in range(1, len(r)-1):
                d = (self._dist_mat[r[i-1], r[i]] + self._dist_mat[r[i], r[i+1]]) * 0.5
                avg_dist[r[i]] = d

        # pick num_swap nodes with the highest avg distance
        swap_nodes = list()
        for k, v in sorted(avg_dist.items(), key=lambda kv: kv[1], reverse=True):
            if len(swap_nodes) >= self.num_swap: break
            swap_nodes.append(k)

        # remove them from routes
        for i, r in enumerate(routes):
            for n in r.copy():
                if n in swap_nodes:
                    r.remove(n)
                    load[i] -= self._demands[n]

        # insert swap_nodes back to the route
        ntrial = 0
        while (swap_nodes and ntrial <= self.max_trial):  # stop if exceed maximun trails
            node = swap_nodes[0]
            # check the constrain
            r_idx = np.random.randint(0, len(routes))
            dem = self._demands[node]
            if (dem + load[r_idx] <= self._cap):
                # pick a location to insert this node
                if (len(routes[r_idx])-1 == 1):
                    n_idx = 1
                else:
                    n_idx = np.random.randint(1, len(routes[r_idx])-1)
                routes[r_idx].insert(n_idx, node)
                load[r_idx] += dem
                swap_nodes.pop(0)  # remove the node in swap since inserted
            ntrial += 1

        # if there is a non-empty swap_nodes, it means the back insertion fails
        # return the input solution back
        if swap_nodes:
            return sol

        return ret

    def _2opt_transform(self, sol):
        for r in sol['routes']:
            r[1:-2] = self._run_2opt(r[1:-2])
        return sol


    @staticmethod
    def _swap_2opt(route, i, k):
        """
        Swaps the endpoints of two edges by reversing a section of nodes
        to eliminate crossovers

        Args:
            route(list): route to apply 2-opt
            i (int): the start index of the portion of the route to be reversed
            k (int): the end index of theportion of route to be reversed

        Return:
            The new route created with a the 2-opt swap

        """
        assert 0 <= i < (len(route)-1)
        assert i < k <= (len(route)-1)
        ret = list(route)
        ret[i:k+1] = list(reversed(route[i:k+1]))
        assert len(ret) == len(route)
        return ret

    def _run_2opt(self, route):
        """
        Improves an existing route using the 2-opt swap

        Args:
        route (list): route to improve

        Return:
            the best route found
        """
        # A flag for indicating if has improved
        has_improved = True
        best_route = list(route)
        best_cost = self.get_cost_of_route(route)
        while has_improved:
            has_improved = False
            for i in range(len(best_route)-1):
                for k in range(i+1, len(best_route)):
                    new_route = self._swap_2opt(best_route, i, k)
                    new_cost = self.get_cost_of_route(new_route)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_route = new_route
                        has_improved = True
                        # A improvement found, return to the top of the while loop
                        break
                if has_improved:
                    break
        assert len(best_route) == len(route)
        return best_route


    def get_fessible_sol(self):
        """
        Initialize and return a fessible solution to the problem accroding
        to the input parameters

        Return:
            a fessible solution as the initial guess of the probelm
        """
        # constain the sol only with the cap

        routes = [[0] for _ in range(self._num_veh)]
        load = np.zeros(self._num_veh)

        # loop over all customer nodes
        for i, dem in enumerate(self._demands):
            # assume the first city is depot
            if i == 0 : continue

            for veh in range(self._num_veh):
                if load[veh] + dem > self._cap:
                    # next vehicle
                    continue

                # load the demand to the vehicle
                routes[veh].append(i)
                load[veh] += dem
                break  # go to next city/demand

        # ask the vehicle to go back to depot
        for r in routes:
            r.append(0)

        return {'load': load, 'routes': routes}

    def get_cost_of_sol(self, sol):
        """
        Get cost of the current solution

        Args:
            sol (dict): a dictionary representing the current solution guess
                        of the problem
        Return:
            the cost of the input solution.
        """

        ret = 0

        for r in sol['routes']:
            ret += self.get_cost_of_route(r)
        return ret

    def get_cost_of_route(self, route):
        """
        Return the cost of a route

        Args:
            route (list): the list of the node index

        Return:
            The cost of the input route
        """
        ret = 0
        for s_idx, e_idx in pairwise(route):
            ret += self._dist_mat[s_idx, e_idx]
        return ret

    # ===================================================
    # belows are testing routines
    # ===================================================
    def test(self):
        print("fessible_sol:")
        fg = self.get_fessible_sol()
        print(fg)

        fg_cost = self.get_cost_of_sol(fg)
        print(fg_cost)

        np.random.seed(100)
        g = self._move_transform(fg)
        print(self.get_cost_of_sol(g))
        print(g)

        g = self._replace_highest_value(g)
        print(self.get_cost_of_sol(g))
        print(g)

        g = self._2opt_transform(g)
        print(self.get_cost_of_sol(g))
        print(g)

    def test_move_transform(self):
        g = self.get_fessible_sol()
        for _ in range(100000):
            g = self.get_neighbor(g)

    def test_get_fessible_sol(self):
        print("fessible_sol:")
        fg = self.get_fessible_sol()
        print(fg)

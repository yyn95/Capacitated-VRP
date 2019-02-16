# coding: utf-8
"""
***Tabu Search***

A Tabu Search Heuristic for the Vehicle Routing Problem
Author(s): Michel Gendreau, Alain Hertz and Gilbert Laporte
Source: Management Science, Vol. 40, No. 10 (Oct., 1994), pp. 1276-1290 Published by: INFORMS
Stable URL: https://www.jstor.org/stable/2661622

New Insertion and Postoptimization Procedures for the Traveling Salesman Problem
Author(s): Michel Gendreau, Alain Hertz and Gilbert Laporte
Source: Operations Research, Vol. 40, No. 6 (Nov. - Dec., 1992), pp. 1086-1094 Published by: INFORMS
Stable URL: https://www.jstor.org/stable/171722

"""
import random
from copy import deepcopy

MAX_ITER = 100
NOUT_LOOP = 3


def tabu_search(capacity, distances, demands, num_veh=5):
    def _search(possible_solution):
        penalty = 1.0
        feasible_check = list()
        tabu_list, tabu_list_size = list(), 10
        current_solution = deepcopy(possible_solution)
        best_solution = deepcopy(current_solution)
        best_distance = _distance(best_solution)
        best_evaluation = _evaluation(best_solution, penalty)
        possible_neighbours = list(range(1, len(demands)))

        us_usable = True
        for iteration in range(MAX_ITER):
            # all possible operation
            possible_moves = list()
            current_distance = _distance(current_solution)
            current_evaluation = _evaluation(current_solution, penalty)
            # make sure at least one city of each route to be checked in each iteration
            for route_r in current_solution:
                if len(route_r) == 1:
                    continue
                city_r = route_r[random.randrange(1, len(route_r))]
                index_route_r = current_solution.index(route_r)

                for route_s in current_solution:
                    index_route_s = current_solution.index(route_s)
                    if index_route_s == index_route_r:
                        continue
                    else:
                        neighbours = _neighbours(city_r, possible_neighbours, p=max(5, len(current_solution[index_route_r])))
                        if len(route_s) > 1 and not list(set(neighbours).intersection(set(route_s))):
                            continue
                        new_route_r = deepcopy(route_r)
                        # remove current city from old route
                        new_route_r.remove(city_r)
                        # insert current city to new route
                        new_route_s = _geni(route_s, [city_r])

                        new_solution = deepcopy(current_solution)
                        new_solution[index_route_r] = new_route_r
                        new_solution[index_route_s] = new_route_s
                        new_distance = _distance(new_solution)
                        new_evaluation = _evaluation(new_solution, penalty)
                        new_feasible = _feasible(new_solution)
                        new_tabu = [city_r, index_route_r]
                        if [city_r, index_route_s] not in tabu_list:
                            # when operation not in the tabu list
                            possible_moves.append([new_solution, new_distance, new_evaluation, new_feasible, new_tabu])
                        elif new_distance < best_distance and new_feasible:
                            # or better distance for feasible solution (special amnesty of tabu search)
                            possible_moves.append([new_solution, new_distance, new_evaluation, new_feasible, new_tabu])
                        elif new_evaluation < best_evaluation and not new_feasible:
                            # or better evaluation for feasible solution (special amnesty of tabu search)
                            possible_moves.append([new_solution, new_distance, new_evaluation, new_feasible, new_tabu])

            if len(possible_moves) > 0:
                # select the operation with best evaluation
                new_solution, new_distance, new_evaluation, new_feasible, new_tabu = sorted(possible_moves, key=lambda x: x[2])[0]
                feasible_check.append(new_feasible)
                if new_evaluation > current_evaluation and _feasible(current_solution) and us_usable:
                    us_usable = False
                    for index_current_route in range(len(current_solution)):
                        current_solution[index_current_route] = _us(current_solution[index_current_route])
                else:
                    us_usable = True
                    current_solution = new_solution
                    # put this original route into tabu list
                    tabu_list.append(new_tabu)

                    if new_feasible and new_distance < best_distance:
                        best_solution, best_distance = new_solution, new_distance
                    # validity of tabu list
                    if len(tabu_list) > tabu_list_size:
                        tabu_list.pop(0)

            if iteration % 10 == 0:
                if all(feasible_check):
                    penalty /= 2
                if all(map(lambda x: not x, feasible_check)):
                    penalty *= 2

            print('Iteration ', iteration)
            print('\tCurrent solution ', current_solution)
            print('\tCurrent distance ', _distance(current_solution))

        return best_solution, best_distance

    def _feasible(solution):
        return all((sum(demands[city] for city in route) <= capacity) for route in solution)

    def _score(route):
        # distance for each route
        distance = distances[route[len(route) - 1]][route[0]]
        for i in range(1, len(route)):
            distance += distances[route[i - 1]][route[i]]
        return distance

    def _distance(solution):
        # distance for each solution (one solution contains one or more routes)
        return sum(_score(route) for route in solution)

    def _evaluation(solution, penalty):
        # evaluation for each solution (one solution contains one or more routes)
        return _distance(solution) + sum(max(0, sum(demands[city] for city in route) - capacity) for route in solution) * penalty

    def _neighbours(city, tour, p=5):
        # find at most p nearest cities in tour
        references = [(candidate, distances[city][candidate]) for candidate in tour]
        if len(references) > 0:
            neighbours = sorted(references, key=lambda x: x[1])
            neighbours = list(map(lambda x: x[0], neighbours))
            if len(neighbours) > p:
                return neighbours[:p]
            return neighbours
        return list()

    def _geni(old_route, off_route):
        # one great method to insert cities from off_route to new_route
        new_route = deepcopy(old_route)
        if len(new_route) == 0:
            if len(off_route) > 2:
                new_route.extend(off_route[:3])
                off_route = off_route[3:]
            else:
                return new_route + off_route
        elif len(new_route) == 1:
            if len(off_route) > 1:
                new_route.extend(off_route[:2])
                off_route = off_route[2:]
            else:
                return new_route + off_route
        elif len(new_route) == 2:
            if len(off_route) > 0:
                new_route.extend(off_route[:1])
                off_route = off_route[1:]
            else:
                return new_route + off_route

        while off_route:
            current_city, off_route = off_route[0], off_route[1:]

            references = list()
            for city_i in _neighbours(current_city, new_route):
                index_city_i = new_route.index(city_i)
                for city_j in _neighbours(current_city, new_route):
                    index_city_j = new_route.index(city_j)
                    for city_k in _neighbours(current_city, new_route):
                        index_city_k = new_route.index(city_k)

                        # the first possible type of insertion
                        if index_city_i < index_city_j < index_city_k:
                            temp_route = list()
                            temp_route.extend(new_route[:index_city_i + 1])
                            temp_route.append(current_city)
                            temp_route.extend(new_route[index_city_i + 1:index_city_j + 1][::-1])
                            temp_route.extend(new_route[index_city_j + 1:index_city_k + 1][::-1])
                            temp_route.extend(new_route[index_city_k + 1:])
                            references.append([temp_route, _score(temp_route)])

                        # the second possible type of insertion
                        for city_l in _neighbours(current_city, new_route):
                            index_city_l = new_route.index(city_l)
                            if index_city_i + 1 < index_city_l < index_city_j + 1 < index_city_k:
                                temp_route = list()
                                temp_route.extend(new_route[:index_city_i + 1])
                                temp_route.append(current_city)
                                temp_route.extend(new_route[index_city_l:index_city_j + 1][::-1])
                                temp_route.extend(new_route[index_city_j + 1:index_city_k])
                                temp_route.extend(new_route[index_city_i + 1:index_city_l][::-1])
                                temp_route.extend(new_route[index_city_k:])
                                references.append([temp_route, _score(temp_route)])
            if references:
                refer_scores = list([reference[1] for reference in references])
                refer_route = references[refer_scores.index(min(refer_scores))][0]
                new_route = refer_route
        return new_route

    def _easy_geni(old_route):
        # easy adaptor of geni for initial route for TSP
        if len(old_route) > 3:
            off_route = deepcopy(old_route)
            random.shuffle(off_route)
            new_route = off_route[:3]
            off_route = off_route[3:]
            return _geni(new_route, off_route)
        return old_route

    def _us(old_route):
        # one great method to reinsert cities in route
        if len(old_route) > 3:
            new_route = deepcopy(old_route)
            index_old_city = 0
            while index_old_city < len(old_route):
                current_city = old_route[index_old_city]
                index_city_i = new_route.index(current_city)
                new_score = _score(new_route)

                escape, proceed = False, True
                if not 0 < index_city_i < len(new_route) - 1:
                    index_old_city += 1
                    continue
                for city_j in _neighbours(new_route[index_city_i + 1], new_route):
                    if escape:
                        break
                    index_city_j = new_route.index(city_j)

                    for city_k in _neighbours(new_route[index_city_i - 1], new_route):
                        if escape:
                            break
                        index_city_k = new_route.index(city_k)

                        # the first possible type of reinsertion
                        if index_city_i < index_city_k < index_city_j:
                            temp_route = list()
                            temp_route.extend(new_route[:index_city_i])
                            temp_route.extend(new_route[index_city_i + 1:index_city_k + 1][::-1])
                            temp_route.extend(new_route[index_city_k + 1:index_city_j + 1][::-1])
                            temp_route.extend(new_route[index_city_j + 1:])
                            temp_route = _geni(temp_route, [current_city])

                            if _score(temp_route) < new_score:
                                new_route = temp_route
                                escape, proceed = True, False

                        # the second possible type of reinsertion
                        if index_city_k + 1 >= len(new_route):
                            continue
                        for index_city_l in _neighbours(new_route[index_city_k + 1], new_route):
                            if escape:
                                break
                            if index_city_i + 1 < index_city_j < index_city_l + 1 < index_city_k + 1:
                                temp_route = list()
                                temp_route.extend(new_route[:index_city_i])
                                temp_route.extend(new_route[index_city_l + 1:index_city_k + 1][::-1])
                                temp_route.extend(new_route[index_city_i + 1:index_city_j][::-1])
                                temp_route.extend(new_route[index_city_j:index_city_l + 1])
                                temp_route.extend(new_route[index_city_k + 1:])
                                temp_route = _geni(temp_route, [current_city])

                                if _score(temp_route) < new_score:
                                    new_route = temp_route
                                    escape, proceed = True, False
                if proceed:
                    index_old_city += 1
            return new_route
        return old_route

    cities = list(range(len(demands)))
    pivot = random.randrange(1, len(demands))
    # one trick
    cities = [0] + cities[pivot:] + cities[1:pivot]
    # initial solution for TSP
    cities = _easy_geni(cities)
    # better solution for TSP
    cities = _us(cities)
    pivot = cities.index(0)
    # make sure depot is at the index 0
    cities = cities[pivot:] + cities[:pivot]
    # get all customer cities
    cities = cities[1:]
    # initial_solution = list()
    # generate initial solution for VRP with solution for TSP
    # while cities:
    #     extra = capacity
    #     for index in range(len(cities)):
    #         extra -= demands[index]
    #         if extra < 0:
    #             initial_solution.append([0] + list(cities[:index]))
    #             cities = cities[index:]
    #             break
    #     else:
    #         initial_solution.append([0] + cities)
    #         cities = []
    # solution_size = len(initial_solution)
    # initial_solution.extend([[0] for _ in range(len(demands) - 1 - solution_size)])
    init_routes = [[0] for _ in range(num_veh)]
    load = [0 for _ in range(num_veh)]

    # loop over all customer nodes
    for i, dem in enumerate(demands):
        # assume the first city is depot
        if i == 0 : continue

        for veh in range(num_veh):
            if load[veh] + dem > capacity:
                # next vehicle
                continue

            # load the demand to the vehicle
            init_routes[veh].append(i)
            load[veh] += dem
            break  # go to next city/demand

    guess_route = init_routes

    for w in range(NOUT_LOOP):
      guess_route, cost =  _search(guess_route)
    return guess_route, cost



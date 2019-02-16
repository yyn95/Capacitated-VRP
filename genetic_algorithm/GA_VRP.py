import numpy as np
import math
import random
import itertools
from numpy.linalg import norm
import copy


# =======================
# Param Configuration
# =======================
# input:
#   N: number of customers
N = None
#   coordinates: N + 1, index for customers from 1 to N, index for depot is 0
coordinates = None
#   W: vehicle capacity
W = None
#   distances: distances matrix, (N+1)*(N+1), index 0 for depot
distances = None
#   demands: N + 1,index for customers(same with the order of coordinates), index for depot is 0
demands = None
#   L: upper cost limit(rough estimation)
L = None
# ===================================
# other Param Configuration for GA
# ===================================
#   customers indexed according to the order in coordinates
customers = None
#   population size for each iteration
population_size = 30
#   parameter for is_spaced judgement (typlical 0.5 or 1)
delta = 1
#   store for keeping population spaced
U = None
#   initiliaztion times for generating the spaced population
tao = 50
#   posibilities for mutation (if <: mutate)
Pm = 0.05
#   maximum number of productive iterations(number of crossovers not yield a clone)
max_alpha = 500
#   maximal number of iterations without improving the best solution
max_beta = 200
#   low_bound: low cost limit(rough estimation)
low_bound = 500
#   MAX_ITER
MAX_ITER = 100

def mod_param_config(**kwargs):
    """
    Module parameter configuration

    Args:
        N: number of customers
        coordinates: N + 1, index for customers from 1 to N, index for depot is 0
        W: vehicle capacity
        distances: distances matrix, (N+1)*(N+1), index 0 for depot
        demands: N + 1,index for customers(same with the order of coordinates), index for depot is 0
    """
    global N, coordinates, W, distances, demands, L, customers, U

    N = kwargs['city_num'] - 1
    coordinates = kwargs['coordinates']
    W = kwargs['capacity']
    distances = kwargs['distances']
    demands = kwargs['demands']
    # calucalte L
    L = 0
    for i in distances:
        for j in i:
            L += j

    # set customers
    customers = np.arange(1, N+1).tolist()
    # store for keeping population spaced
    U = np.zeros(int(L/delta),int)

# ===========================
# Params for GA_restarts
# ===========================
# the number of chromosomes get replaced each iteration
num_replace = 8
restart_Pm = 0.1
restart_maxalpha = 200
restart_maxbeta = 200

# INFINITY
INF = 99999

class chromosomes:
    """
    define chromosomes class: each chromosomes represents a solution:
    sequence: a list store the order of customers being visited, length = N
    P: precursor of each node in chromosomes(the first item is 0 for node depot), length = N+1
    V: the least cost of the path from 0 to each node j(the first item is 0 for node depot), length = N+1
    trips: list of trips, each trip is a list of the customers in visited order
    fitness: total cost(distances) of this solution
    """
    def __init__(self, sequence):
        self.sequence = sequence

    def split(self):
        self.V = np.zeros(N+1,int)
        self.V[1:] = INF
        self.P = np.zeros(N+1,int)
        for i in range(1,N+1):
            load = 0
            cost = 0
            j = i
            while True:
                load = load + demands[self.sequence[j-1]]
                if i == j:
                    cost = distances[0][self.sequence[j-1]] + distances[self.sequence[j-1]][0]
                else:
                    cost = cost - distances[self.sequence[j-2]][0] + distances[self.sequence[j-2]][self.sequence[j-1]] + distances[self.sequence[j-1]][0]
                if load <= W and cost <= L:
                    if self.V[i-1] + cost < self.V[j]:
                        self.V[j] = self.V[i-1] + cost
                        self.P[j] = i - 1
                    j = j + 1
                if (j > N or load > W or cost > L):
                    break
        return self.V,self.P

    def solution(self):
        self.trips = []
        for i in range(0,N):
            self.trips.append([])
        t = 0
        j = N
        self.split()
        while True:
            t = t + 1
            i = self.P[j]
            for k in range(i+1,j+1):
                self.trips[t].append(self.sequence[k-1])
            j = i
            if i == 0:
                break
        self.trips = [i for i in self.trips if i != []]
        return self.trips

    def get_fitness(self):
        total_cost = 0
        self.solution()
        for trip in self.trips:
            if trip:
                total_cost = total_cost + distances[0][trip[0]]
                if len(trip) > 1:
                    for j in range(0,len(trip)-1):
                        total_cost = total_cost + distances[trip[j]][trip[j+1]]
                total_cost = total_cost + distances[trip[-1]][0]
        self.fitness = total_cost
        return self.fitness


#define GA
#calcilate total distances of one trip
def trip_distance(trip):
    total_cost = 0
    total_cost = total_cost + distances[0][trip[0]]
    if len(trip) > 1:
        for j in range(0,len(trip)-1):
            total_cost = total_cost + distances[trip[j]][trip[j+1]]
    total_cost = total_cost + distances[trip[-1]][0]
    return total_cost

#calculate loads of one trip
def trip_load(trip):
    total_load = 0
    for j in range(0,len(trip)):
        total_load = total_load + demands[trip[j]]
    return total_load

#decending
def decsort_population(population):
    sortpopulation = sorted(population,key = lambda chromosomes: chromosomes.fitness,reverse = True)
    return sortpopulation

#ascending
def ascsort_population(population):
    sortpopulation = sorted(population,key = lambda chromosomes: chromosomes.fitness)
    return sortpopulation

#intialization
#calculate  Clarke_White solution
def Clarke_White(num_customer, Mdistances, demands, vehicle_capacity):
    #generate the savings list,each item is [i,j,saving]
    savings = []
    for i in range(1,num_customer + 1):
        for j in range(i + 1,num_customer + 1):
            saving = Mdistances[0][i] + Mdistances[0][j] - Mdistances[i][j]
            savings.append([i,j,saving])
    savings = sorted(savings, key = lambda i: i[2],reverse = True)
    trips = [[i] for i in list(range(0,num_customer + 1))] #first 0 is depot
    indicator = np.arange(0,num_customer + 1) #the trip index of each customer,first 0 for depot
    for saving in savings:
        #print(trips)
        if indicator[saving[0]] != indicator[saving[1]]:
            if vehicle_capacity >= trip_load(trips[indicator[saving[0]]]) + trip_load(trips[indicator[saving[0]]]):
                #both i and j are the first customer of the trip, add reversed i trip in front of j trip, change j trip to []
                if saving[0] == trips[indicator[saving[0]]][0] and saving[1] == trips[indicator[saving[1]]][0]:
                    trips[indicator[saving[0]]].reverse()
                    trips[indicator[saving[0]]] = trips[indicator[saving[0]]] + trips[indicator[saving[1]]]
                    temp = indicator[saving[1]]
                    for customer in trips[temp]:
                        indicator[customer] = indicator[saving[0]]
                    trips[temp] = []
                #i is the first customer while j is the last, add i trip in the end of j trip, change i trip to []
                elif saving[0] == trips[indicator[saving[0]]][0] and saving[1] == trips[indicator[saving[1]]][-1]:
                    trips[indicator[saving[1]]] = trips[indicator[saving[1]]] + trips[indicator[saving[0]]]
                    temp = indicator[saving[0]]
                    for customer in trips[temp]:
                        indicator[customer] = indicator[saving[1]]
                    trips[temp] = []
                #i is the last customer while j is the first, add j trip in the end of i trip, change j trip to []
                elif saving[0] == trips[indicator[saving[0]]][-1] and saving[1] == trips[indicator[saving[1]]][0]:
                    trips[indicator[saving[0]]] = trips[indicator[saving[0]]] + trips[indicator[saving[1]]]
                    temp = indicator[saving[1]]
                    for customer in trips[temp]:
                        indicator[customer] = indicator[saving[0]]
                    trips[temp] = []
                #both i and j are the last customer of the trip, add reversed j trip in the end of i trip, change j trip to []
                elif saving[0] == trips[indicator[saving[0]]][-1] and saving[1] == trips[indicator[saving[1]]][-1]:
                    trips[indicator[saving[1]]].reverse()
                    trips[indicator[saving[0]]] = trips[indicator[saving[0]]] + trips[indicator[saving[1]]]
                    temp = indicator[saving[1]]
                    for customer in trips[temp]:
                        indicator[customer] = indicator[saving[0]]
                    trips[temp] = []
        #print(trips)
    trips = [i for i in trips[1:] if i != []] #ignore depot 0
    sequence = []
    for trip in trips:
        sequence = sequence + trip
    CW = chromosomes(sequence)
    CW.get_fitness()
    return CW

#calculate Gillett_Miller solution
def vector_angle(vector1, vector2):
    cos_sim = np.inner(vector1, vector2)/(norm(vector1)*norm(vector2))
    return math.acos(cos_sim)

#coordinates: list of depot and customer coordinates, 0 for depot
def Gillett_Miller(coordinates, Mdistances, demands, vehicle_capacity):
    depot = np.array(coordinates[0])
    first_customer = random.randint(1,N)
    other_customers = list(range(1,first_customer)) + list(range(first_customer + 1, N + 1))
    angles = []
    angles.append((first_customer,0))
    for customer in other_customers:
        angle = vector_angle(np.array(coordinates[first_customer]) - depot,np.array(coordinates[customer]) - depot)
        angles.append((customer,angle))
    angles = sorted(angles,key = lambda i: i[1])
    sequence = [j[0] for j in angles]
    GM = chromosomes(sequence)
    GM.get_fitness()
    return GM

#initialization for the main phase GA
def initialization(population_size=population_size):
    population = []
    #init1 = chromosomes(np.random.permutation(customers))
    #init1.get_fitness()
    init1 = Clarke_White(N, distances, demands, W)
    population.append(init1)
    U[int(init1.fitness/delta)] = 1
    init2 = Gillett_Miller(coordinates, distances, demands, W)
    if U[int(init2.fitness/delta)] == 0:
        population.append(init2)
        U[int(init2.fitness/delta)] = 1
        population_size = population_size - 1
    for i in range(0, population_size - 1):
        for j in range(0,tao):
            s = chromosomes(np.random.permutation(customers).tolist())
            s.get_fitness()
            if U[int(s.fitness/delta)] == 0:
                population.append(s)
                U[int(s.fitness/delta)] = 1
                break
    population = ascsort_population(population)
    return population

#randomly select two cross points, and randomly choose one child
def crossover(chromosomes1,chromosomes2):
    positions = sorted(random.sample(list(range(0,N)),2))
    #print(positions)
    otherpos = list(range(positions[1]+1,N)) + list(range(0,positions[0]))
    #if choose_child == 1:
    C = np.zeros(N,int).tolist()
    C[positions[0]:positions[1]+1] = chromosomes1.sequence[positions[0]:positions[1]+1]
    temp = [i for i in chromosomes2.sequence[positions[1]+1:N] + chromosomes2.sequence[0:positions[1]+1] if i not in C[positions[0]:positions[1]+1]]
    for i in otherpos:
        C[i] = temp.pop(0)
    child1 = chromosomes(C)
    child1.get_fitness()
    #if choose_child == 2:
    C = np.zeros(N,int).tolist()
    C[positions[0]:positions[1]+1] = chromosomes2.sequence[positions[0]:positions[1]+1]
    temp = [i for i in chromosomes1.sequence[positions[1]+1:N] + chromosomes1.sequence[0:positions[1]+1] if i not in C[positions[0]:positions[1]+1]]
    for i in otherpos:
        C[i] = temp.pop(0)
    child2 = chromosomes(C)
    child2.get_fitness()
    return child1,child2

#return the mutation chromosomes
def mutation(chromo):
    pairs = itertools.permutations(list(range(0,N+1)), 2)
    least_cost = chromo.fitness #current_least_cost
    cost = chromo.fitness
    best_mutation = copy.deepcopy(chromo.trips)
    for pair in pairs:
        trips = copy.deepcopy(best_mutation)
        for i in range(0,len(trips)):
            if pair[0] in trips[i]:
                index0 = (i,trips[i].index(pair[0]))
            if pair[1] in trips[i]:
                index1 = (i,trips[i].index(pair[1]))
        for method in range(1,10):
            trips = copy.deepcopy(best_mutation)
            #M1
            if method == 1:
                if pair[0] > 0:
                    if pair[1] > 0 and index0[0] != index1[0]:
                        trips[index0[0]].remove(pair[0])
                        trips[index1[0]].insert(index1[1] + 1,pair[0])
                    elif pair[1] > 0 and index0[0] == index1[0]:
                        if index0[1] < index1[1]:
                            trips[index1[0]].insert(index1[1] + 1,pair[0])
                            trips[index0[0]].pop(index0[1])
                        elif index0[1] > index1[1]:
                            trips[index0[0]].pop(index0[1])
                            trips[index1[0]].insert(index1[1] + 1,pair[0])
                    elif pair[1] == 0:
                        trips.append([pair[0]])
                    trips = [i for i in trips if i != []]
                    cost = 0
                    for trip in trips:
                        if trip_load(trip) <= W:
                            cost = cost + trip_distance(trip)
                        else:
                            cost = chromo.fitness
                            continue
                    if cost < least_cost:
                        least_cost = cost
                        best_mutation = copy.deepcopy(trips)
                        break
            #M2
            if method == 2:
                if pair[0] > 0 and pair[1] > 0:
                    if pair[0] != trips[index0[0]][-1]:
                        x = trips[index0[0]][index0[1] + 1]
                        if index0[0] != index1[0]:
                            trips[index0[0]].pop(index0[1] + 1)
                            trips[index0[0]].pop(index0[1])
                            trips[index1[0]].insert(index1[1] + 1,pair[0])
                            trips[index1[0]].insert(index1[1] + 2,x)
                        else:
                            if index0[1] < index1[1]:
                                trips[index1[0]].insert(index1[1] + 1,pair[0])
                                trips[index1[0]].insert(index1[1] + 2,x)
                                trips[index0[0]].pop(index0[1] + 1)
                                trips[index0[0]].pop(index0[1])
                            if index0[1] > index1[1]:
                                trips[index0[0]].pop(index0[1] + 1)
                                trips[index0[0]].pop(index0[1])
                                trips[index1[0]].insert(index1[1] + 1,pair[0])
                                trips[index1[0]].insert(index1[1] + 2,x)
                        trips = [i for i in trips if i != []]
                        cost = 0
                        for trip in trips:
                            if trip_load(trip) <= W:
                                cost = cost + trip_distance(trip)
                            else:
                                cost = chromo.fitness
                                continue
                        if cost < least_cost:
                            least_cost = cost
                            best_mutation = copy.deepcopy(trips)
                            break
                elif pair[0] > 0 and pair[1] == 0:
                    if pair[0] != trips[index0[0]][-1]:
                        trips.append([pair[0],trips[index0[0]][index0[1] + 1]])
                        trips[index0[0]].pop(index0[1] + 1)
                        trips[index0[0]].pop(index0[1])
                    trips = [i for i in trips if i != []]
                    cost = 0
                    for trip in trips:
                        if trip_load(trip) <= W:
                            cost = cost + trip_distance(trip)
                        else:
                            cost = chromo.fitness
                            continue
                    if cost < least_cost:
                        least_cost = cost
                        best_mutation = copy.deepcopy(trips)
                        break
            #M3
            if method == 3:
                if pair[0] > 0 and pair[1] > 0:
                    if pair[0] != trips[index0[0]][-1]:
                        x = trips[index0[0]][index0[1] + 1]
                        if index0[0] != index1[0]:
                            trips[index0[0]].pop(index0[1] + 1)
                            trips[index0[0]].pop(index0[1])
                            trips[index1[0]].insert(index1[1] + 1,x)
                            trips[index1[0]].insert(index1[1] + 2,pair[0])
                        else:
                            if index0[1] < index1[1]:
                                trips[index1[0]].insert(index1[1] + 1,x)
                                trips[index1[0]].insert(index1[1] + 2,pair[0])
                                trips[index0[0]].pop(index0[1] + 1)
                                trips[index0[0]].pop(index0[1])
                            if index0[1] > index1[1]:
                                trips[index0[0]].pop(index0[1] + 1)
                                trips[index0[0]].pop(index0[1])
                                trips[index1[0]].insert(index1[1] + 1,x)
                                trips[index1[0]].insert(index1[1] + 2,pair[0])
                        trips = [i for i in trips if i != []]
                        cost = 0
                        for trip in trips:
                            if trip_load(trip) <= W:
                                cost = cost + trip_distance(trip)
                            else:
                                cost = chromo.fitness
                                continue
                        if cost < least_cost:
                            least_cost = cost
                            best_mutation = copy.deepcopy(trips)
                            break
                elif pair[0] > 0 and pair[1] == 0:
                    if pair[0] != trips[index0[0]][-1]:
                        trips.append([trips[index0[0]][index0[1] + 1],pair[0]])
                        trips[index0[0]].pop(index0[1] + 1)
                        trips[index0[0]].pop(index0[1])
                    trips = [i for i in trips if i != []]
                    cost = 0
                    for trip in trips:
                        if trip_load(trip) <= W:
                            cost = cost + trip_distance(trip)
                        else:
                            cost = chromo.fitness
                            continue
                    if cost < least_cost:
                        least_cost = cost
                        best_mutation = copy.deepcopy(trips)
                        break

            #M4
            if method == 4:
                if pair[0] > 0 and pair[1] > 0:
                    trips[index0[0]][index0[1]] = pair[1]
                    trips[index1[0]][index1[1]] = pair[0]
                    trips = [i for i in trips if i != []]
                    cost = 0
                    for trip in trips:
                        if trip_load(trip) <= W:
                            cost = cost + trip_distance(trip)
                        else:
                            cost = chromo.fitness
                            continue
                    if cost < least_cost:
                        least_cost = cost
                        best_mutation = copy.deepcopy(trips)
                        break
            #M5
            if method == 5:
                if pair[0] > 0 and pair[1] > 0:
                    if pair[0] != trips[index0[0]][-1]:
                        x = trips[index0[0]][index0[1] + 1]
                        if x != pair[1]:
                            trips[index0[0]][index0[1]] = pair[1]
                            trips[index1[0]][index1[1]] = pair[0]
                            if index0[0] != index1[0]:
                                trips[index1[0]].insert(index1[1] + 1,x)
                                trips[index0[0]].pop(index0[1] + 1)
                            else:
                                if index0[1] < index1[1]:
                                    trips[index1[0]].insert(index1[1] + 1,x)
                                    trips[index0[0]].pop(index0[1] + 1)
                                if index0[1] > index1[1]:
                                    trips[index0[0]].pop(index0[1] + 1)
                                    trips[index1[0]].insert(index1[1] + 1,x)
                            trips = [i for i in trips if i != []]
                            cost = 0
                            for trip in trips:
                                if trip_load(trip) <= W:
                                    cost = cost + trip_distance(trip)
                                else:
                                    cost = chromo.fitness
                                    continue
                            if cost < least_cost:
                                least_cost = cost
                                best_mutation = copy.deepcopy(trips)
                                break
            #M6
            if method == 6:
                if pair[0] > 0 and pair[1] > 0:
                    if pair[0] != trips[index0[0]][-1] and pair[1] != trips[index1[0]][-1]:
                        x = trips[index0[0]][index0[1] + 1]
                        if x != pair[1]:
                            trips[index0[0]][index0[1]] = pair[1]
                            trips[index0[0]][index0[1] + 1] = trips[index1[0]][index1[1] + 1]
                            trips[index1[0]][index1[1]] = pair[0]
                            trips[index1[0]][index1[1] + 1] = x
                            trips = [i for i in trips if i != []]
                            cost = 0
                            for trip in trips:
                                if trip_load(trip) <= W:
                                    cost = cost + trip_distance(trip)
                                else:
                                    cost = chromo.fitness
                                    continue
                            if cost < least_cost:
                                least_cost = cost
                                best_mutation = copy.deepcopy(trips)
                                break
            #M7
            if method == 7:
                if pair[0] > 0 and pair[1] > 0:
                    if index0[0] == index1[0]:
                        if pair[0] != trips[index0[0]][-1]:
                            trips[index1[0]][index1[1]] = trips[index0[0]][index0[1] + 1]
                            trips[index0[0]][index0[1] + 1] = pair[1]
                        elif pair[0] == trips[index0[0]][-1]:
                            trips[index0[0]].append(pair[1])
                            trips.append(trips[index1[0]][index1[1] + 1:])
                            trips[index1[0]] = trips[index1[0]][:index1[1]]
                trips = [i for i in trips if i != []]
                cost = 0
                for trip in trips:
                    if trip_load(trip) <= W:
                        cost = cost + trip_distance(trip)
                    else:
                        cost = chromo.fitness
                        continue
                if cost < least_cost:
                    least_cost = cost
                    best_mutation = copy.deepcopy(trips)
                    break
            #M8
            if method == 8:
                if pair[0] > 0 and pair[1] > 0:
                    if index0[0] != index1[0]:
                        if pair[0] != trips[index0[0]][-1]:
                            trips[index1[0]][index1[1]] = trips[index0[0]][index0[1] + 1]
                            trips[index0[0]][index0[1] + 1] = pair[1]
                        elif pair[0] == trips[index0[0]][-1]:
                            trips[index0[0]].append(pair[1])
                            trips.append(trips[index1[0]][index1[1] + 1:])
                            trips[index1[0]] = trips[index1[0]][:index1[1]]
                        trips = [i for i in trips if i != []]
                        cost = 0
                        for trip in trips:
                            if trip_load(trip) <= W:
                                cost = cost + trip_distance(trip)
                            else:
                                cost = chromo.fitness
                                continue
                        if cost < least_cost:
                            least_cost = cost
                            best_mutation = copy.deepcopy(trips)
                            break
            #M9
            if method == 9:
                if pair[0] > 0 and pair[1] > 0:
                    if index0[0] != index1[0]:
                        if pair[0] != trips[index0[0]][-1] and pair[1] != trips[index1[0]][-1]:
                            x = trips[index0[0]][index0[1] + 1]
                            trips[index0[0]][index0[1] + 1] = trips[index1[0]][index1[1] + 1]
                            trips[index1[0]].pop(index1[1] + 1)
                            trips[index1[0]].insert(index1[1],x)
                        elif pair[0] == trips[index0[0]][-1] and pair[1] != trips[index1[0]][-1]:
                            trips[index0[0]].append(trips[index1[0]][index1[1] + 1])
                            trips[index1[0]].pop(index1[1] + 1)
                            trips.append(trips[index1[0]][index1[1]:])
                            trips[index1[0]] = trips[index1[0]][:index1[1]]
                        elif pair[0] != trips[index0[0]][-1] and pair[1] == trips[index1[0]][-1]:
                            trips[index1[0]].insert(index1[1],trips[index0[0]][index0[1] + 1])
                            temp = trips[index0[0]][index0[1] + 1:]
                            trips[index0[0]] = trips[index0[0]][:index0[1] + 1]
                            temp.pop(0)
                            trips.append(temp)
                        elif pair[0] == trips[index0[0]][-1] and pair[1] == trips[index1[0]][-1]:
                            trips.append([pair[1]])
                            trips[index1[0]] = trips[index1[0]][:index1[1]]
                        trips = [i for i in trips if i != []]
                        cost = 0
                        for trip in trips:
                            if trip_load(trip) <= W:
                                cost = cost + trip_distance(trip)
                            else:
                                cost = chromo.fitness
                                continue
                        if cost < least_cost:
                            least_cost = cost
                            best_mutation = copy.deepcopy(trips)
                            break
    if best_mutation != chromo.trips:
        seq = []
        for trip in best_mutation:
                seq = seq + trip
        M = chromosomes(seq)
        M.get_fitness()
        return M
    else:
        return chromo

#the main phase of GA with only one chromosomes replaced each iteration
def main_GA(population):
    alpha = 0
    beta = 0
    for _ in range(MAX_ITER):
        parents = []
        for i in range(0,2):
            populist = [j for j in population if j not in parents]
            parent = random.sample(populist,2)
            if parent[0].fitness < parent[1].fitness:
                parents.append(parent[0])
            else:
                parents.append(parent[1])
        children = crossover(parents[0],parents[1])
        choose_child = random.sample([1,2],1)
        child = children[choose_child[0] - 1]
        k = random.randint(int(len(population)/2),len(population) - 1)
        if random.random() < Pm:
            M = mutation(child)
            if U[int(M.fitness/delta)] == 0 or int(M.fitness/delta) == int(population[k].fitness/delta):
                child = M
        if U[int(child.fitness/delta)] == 0 or int(child.fitness/delta) == int(population[k].fitness/delta):
            alpha = alpha + 1
            if child.fitness < population[0].fitness:
                beta = 0
            else:
                beta = beta + 1
            U[int(population[k].fitness/delta)] = 0
            population[k] = child
            U[int(child.fitness/delta)] = 1
            population = ascsort_population(population)
        print("alpha: %d" % alpha)
        print("beta: %d" % beta)
        print(population[0].fitness)
        if alpha == max_alpha or beta == max_beta or population[0].fitness <= low_bound:
            break
    return population

#the restart phase of GA
def restart_GA(population):
    alpha = 0
    beta = 0
    while True:
        count_replace = 0
        count_iteration = 0
        #each restart replace num_replace chromosomes
        while True:
            #print("alpha: %d" % alpha)
            #print("beta: %d" % beta)
            count_iteration =  count_iteration + 1
            new_chromosomes = []
            while len(new_chromosomes) < num_replace:
                chromo = chromosomes(np.random.permutation(customers).tolist())
                chromo.get_fitness()
                if U[int(chromo.fitness/delta)] == 0:
                    new_chromosomes.append(chromo)
            #print("new: 3")
            new_chromosomes = ascsort_population(new_chromosomes)
            for new_chromo in new_chromosomes:
                if new_chromo.fitness < population[-1].fitness:
                    if new_chromo.fitness < population[0].fitness:
                        beta = 0
                    else:
                        beta = beta + 1
                    U[int(population[-1].fitness/delta)] = 0
                    population[-1] = new_chromo
                    U[int(new_chromo.fitness/delta)] = 1
                    population =  ascsort_population(population)
                    count_replace = count_replace + 1
                    alpha = alpha + 1
                else:
                    populist = [j for j in (population + new_chromosomes) if j != new_chromo]
                    best_child = population[-1]
                    for parent in populist:
                        children = crossover(new_chromo,parent)
                        child = children[0] if children[0].fitness < children[1].fitness else children[1]
                        #mutation
                        if random.random() < restart_Pm:
                            M = mutation(child)
                            if U[int(M.fitness/delta)] == 0:
                                child = M
                        if child.fitness < best_child.fitness:
                            best_child = child
                    alpha = alpha + len(populist)
                    if best_child.fitness < max(new_chromosomes[-1].fitness,population[-1].fitness):
                        if best_child.fitness < population[0].fitness:
                            beta = 0
                        else:
                            beta = beta + 1
                        U[int(population[-1].fitness/delta)] = 0
                        population[-1] = best_child
                        U[int(best_child.fitness/delta)] = 1
                        population =  ascsort_population(population)
                        count_replace = count_replace + 1
            if count_iteration >= 5 and count_replace >= 8:
                break
        if alpha == restart_maxalpha or beta == restart_maxbeta or population[0].fitness <= low_bound:
            break
    return population

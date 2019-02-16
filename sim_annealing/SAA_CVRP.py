import math, numpy, random, copy

def _file_iter():
    input_file = 'B-n78-k10.vrp'  # 'A-n80-k10.vrp'
    with open(input_file) as f:
        for token in f.read().split():
            yield token


file_iter = _file_iter()

while True:
    line = file_iter.__next__()
    if line == 'TYPE':
        file_iter.__next__()
        if file_iter.__next__() != 'CVRP':
            raise ValueError('TYPE ERROR')
    elif line == 'DIMENSION':
        file_iter.__next__()
        city_num = int(file_iter.__next__())
    elif line == 'EDGE_WEIGHT_TYPE':
        file_iter.__next__()
        if file_iter.__next__() != 'EUC_2D':
            raise ValueError('EDGE_WEIGHT_TYPE ERROR')
    elif line == 'CAPACITY':
        file_iter.__next__()
        capacity = int(file_iter.__next__())
    elif line == 'NODE_COORD_SECTION':
        coordinates = [[0, 0] for _ in range(city_num)]
        for city_id in range(city_num):
            file_iter.__next__()
            coordinates[city_id][0] = int(file_iter.__next__())
            coordinates[city_id][1] = int(file_iter.__next__())
        distances = [[0 for i in range(city_num)] for y in range(city_num)]
        for i in range(city_num):
            for j in range(city_num):
                distances[i][j] = round(math.sqrt(math.pow(coordinates[i][0] - coordinates[j][0], 2) + math.pow(
                    coordinates[i][1] - coordinates[j][1], 2)))
    elif line == 'DEMAND_SECTION':
        demands = [None] * city_num
        for city_id in range(city_num):
            file_iter.__next__()
            demands[city_id] = int(file_iter.__next__())
        break

class customer:

    def __init__(self, num, x, y, demand):
        self.num = num
        self.x = x
        self.y = y
        self.demand = demand


customer_list=[]
for city_id in range(city_num):
    customer_list.append(customer(city_id, coordinates[city_id][0], coordinates[city_id][1], demands[city_id]))

def compute_cost(route):
    cost = [0 for i in range(len(route))]
    for i in range(len(route)):
        if len(route[i])!= 0:
            cost[i] += cost_matrix[0][route[i][0]]
            for j in range(len(route[i])-1):
                cost[i] += cost_matrix[route[i][j]][route[i][j+1]]
            cost[i] += cost_matrix[0][route[i][len(route[i])-1]]
    return cost

def compute_total_cost(route):
    cost = compute_cost(route)
    total_cost = sum(cost)
    return total_cost

def compute_load(route):
    load = [0 for i in range(len(route))]
    for i in range(len(route)):
        for j in range(len(route[i])):
            load[i] += customer_list[route[i][j]].demand
    return load


def initialize(customer_list):
    # simple initialization
    cost_matrix = [[0 for j in range(len(customer_list))] for i in range(len(customer_list))] #distance between each pair of customers

    for i in range(len(customer_list)):
        for j in range(i + 1, len(customer_list)):
            cost_matrix[i][j] = round(math.sqrt(
                pow((customer_list[i].x - customer_list[j].x), 2) + pow((customer_list[i].y - customer_list[j].y), 2)))
            cost_matrix[j][i] = cost_matrix[i][j]  # symmetrical matrix

    initial_load = [0]
    S = [[]]  # initial solution of route
    j = 0
    for i in range(1, len(customer_list)):
        if initial_load[j] + customer_list[i].demand <= capacity:
            initial_load[j] += customer_list[i].demand
            S[j].append(i)
        else:
            S.append([])
            initial_load.append(0)
            j = j + 1
            initial_load[j] += customer_list[i].demand
            S[j].append(i)
    initial_route = S
    return cost_matrix, initial_route, initial_load
    
cost_matrix, initial_route, initial_load = initialize(customer_list)

initial_cost = compute_cost(initial_route)
load = copy.deepcopy(initial_load)
cost = copy.deepcopy(initial_cost)
route = copy.deepcopy(initial_route)
initial_total_cost = compute_total_cost(initial_route)
total_cost = initial_total_cost
print('initial route: ',route)
print('initial load: ',load)
print('initial total cost: ',total_cost)

#a kind of transformation to generate a new solution of the route
def move(route):
    num = 5
    newroute = copy.deepcopy(route)
    for i in range(len(newroute)):
        newroute[i].insert(0, 0) #insert depot to the beginning of the route of each vehicle
    pairdistance = [(0,0)] #no0 element is the number of vi+1 customer, no1 element is the distance of pair customers
    #sort vi+1 customer number by distances of pairs from min to max
    for i in range(len(newroute)):
        for j in range(len(newroute[i])-1):
            d = cost_matrix[newroute[i][j]][newroute[i][j+1]]
            k = 0
            while k < len(pairdistance):
                if d < pairdistance[k][1]:
                    break
                k = k + 1
            pairdistance.insert(k,(newroute[i][j+1], d))
    protect_customer = []
    for k in range(num):
        protect_customer.append(pairdistance[k + 1][0]) #get 5 vi+1 customer numbers with the shortest distances of pairs，do not get no0 element since it is depot

    for i in range(len(newroute)):
        newroute[i].append(0)  # push depot to the last
    k = num
    while k > 0:
        i = random.randint(0, len(newroute) - 1)
        while len(newroute[i]) < 3:
            i = random.randint(0, len(newroute) - 1)
        j = random.randint(1, len(newroute[i]) - 2) #do not take no0 and the last element into account since they are depots
        if newroute[i][j] not in protect_customer:
            customer_id = newroute[i][j]
            del newroute[i][j]
            k = k - 1
        else:
            continue
        newload = compute_load(newroute)
        i = random.randint(0, len(newroute) - 1)
        while newload[i] + customer_list[customer_id].demand > capacity:  # do not violet the limit of capacity
            i = random.randint(0, len(newroute) - 1)
        deltamin = 99999  # record the minimum change of the cost after inserting the customers（actually must be a positive value）
        deltamin_pos = 0  # record the position of inserting the customers with the minimum change of the cost
        for j in range(1, len(newroute[i])):  # no0 and the last elements are depot, do not take them into account
            delta = cost_matrix[customer_id][newroute[i][j - 1]] + cost_matrix[customer_id][newroute[i][j]] - \
                    cost_matrix[newroute[i][j - 1]][newroute[i][j]]
            if delta < deltamin:
                deltamin = delta
                deltamin_pos = j
        newroute[i].insert(deltamin_pos, customer_id)
        newload[i] = newload[i] + customer_list[customer_id].demand

    for i in range(len(newroute)):  # delete elements of the depot
        del newroute[i][len(newroute[i]) - 1]
        del newroute[i][0]

    return newroute

#a kind of transformation to generate a new solution of the route
def replace_highest_average(route):
    num = 5
    newroute = copy.deepcopy(route)
    for i in range(len(newroute)):#insert elements of the depot
        newroute[i].insert(len(newroute[i]), 0)
        newroute[i].insert(0, 0)
    average_distance = [(0, 0)]#insert depot to initialize the list, the no0 element is the number of the customer，the no1 element is the average distances
    # sort customer numbers by average distances from max to min
    for i in range(len(newroute)):
        for j in range(1, len(newroute[i])-1):
            d = (cost_matrix[newroute[i][j]][newroute[i][j-1]] + cost_matrix[newroute[i][j]][newroute[i][j+1]]) / 2 #average distance
            k = 0
            while k < len(average_distance):
                if d > average_distance[k][1]:
                    break
                k = k + 1
            average_distance.insert(k,(newroute[i][j], d))
    customer_to_move = []
    for k in range(num):
        customer_to_move.append(average_distance[k][0])
    newload = compute_load(newroute)
    for customer_id in customer_to_move:
        for i in range(len(newroute)):
            if customer_id in newroute[i]: #find customer id in newroute. delete it and insert it again
                newroute[i].remove(customer_id)
                newload = compute_load(newroute)
                break
        i = random.randint(0, len(newroute) - 1)
        while newload[i] + customer_list[customer_id].demand > capacity:  #do not violet the limit of capacity
            i = random.randint(0, len(newroute) - 1)
        deltamin = 99999  #record the minimum change of the cost after inserting the customers（actually must be a positive value）
        deltamin_pos = 0  #record the position of inserting the customers with the minimum change of the cost
        for j in range(1, len(newroute[i])):  #no0 and the last elements are depot, do not take them into account
            delta = cost_matrix[customer_id][newroute[i][j-1]] + cost_matrix[customer_id][newroute[i][j]] - cost_matrix[newroute[i][j-1]][newroute[i][j]]
            if delta < deltamin:
                deltamin = delta
                deltamin_pos = j
        newroute[i].insert(deltamin_pos, customer_id)
        newload[i] = newload[i] + customer_list[customer_id].demand
    for i in range(len(newroute)):  #delete elements of the depot
        del newroute[i][len(newroute[i]) - 1]
        del newroute[i][0]
    return newroute

#generate a new solution of routes
def newsolution(route):
    newroute = copy.deepcopy(route)
    newroute = move(newroute)
    newroute = replace_highest_average(newroute)
    compute_total_cost(route)
    total_cost = compute_total_cost(route)
    newcost = compute_cost(newroute)
    new_total_cost = sum(newcost)
    delta = new_total_cost - total_cost #the change of the cost
    return newroute, delta

#parameters of SA algorithm
alpha = 0.99
beta = 1.05
M0 = 5
bestroute = copy.deepcopy(route)
best_total_cost = compute_total_cost(bestroute)
T = 5000 #initial temperature
Time = 0
MaxTime = 10000

#SA algorithm
while T > 0.001 and Time < MaxTime:
    M = M0
    while M >= 0:
        newroute, delta = newsolution(route)
        if delta < 0:
            route = copy.deepcopy(newroute)
            total_cost = compute_total_cost(route)
            if total_cost < best_total_cost:
                bestroute = copy.deepcopy(route)
                best_total_cost = total_cost
        elif math.exp(- delta / T) > random.random(): # randomly accept a worse solution
            route = copy.deepcopy(newroute)
            total_cost = compute_total_cost(route)
        M = M - 1
    Time = Time + M0
    T = alpha * T
    M0 = beta * M0

print('best route:',bestroute)
print('total cost of best route: ',best_total_cost)
print('cost saving: ', initial_total_cost - best_total_cost)

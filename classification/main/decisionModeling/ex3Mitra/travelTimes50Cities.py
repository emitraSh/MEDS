import csv
import gurobipy as gp
from gurobipy import GRB
gp.disposeDefaultEnv()

arcs = set()
cities = set()
distances = dict()

with open('travelTimes50Cities.csv', mode='r' , encoding='utf-8-sig') as file:
    reader = csv.DictReader(file, delimiter=';')

    for row in reader:
        origin = row['origin']
        destination = row['destination']
        distance_miles = float(row['distance_miles'])


        cities.add(origin)
        cities.add(destination)
        distances[(origin,destination)] = distance_miles

arcs.update(distances.keys())
cities = sorted(cities) #why?????????
cities = cities[:20]  # Keep only the first 20 cities
arcs = [(i, j) for i in cities for j in cities if i != j]
distances = {k: v for k, v in distances.items() if k[0] in cities and k[1] in cities}
startCity = 'New York'

m = gp.Model('travelTimes50Cities')


flow = {}
for a in arcs:
    flow[a] = m.addVar(vtype=GRB.BINARY, obj=distances[a] , name = f"from {a[0]} to {a[1]}")
m.ModelSense = 1


for city in cities:
    if city != startCity:
        constr_in = m.addConstr(gp.quicksum(flow[i, city] for i in cities if i!=city)== 1 , name=f'flow_in{city}')
        constr_out = m.addConstr(gp.quicksum(flow[city, j] for j in cities if j!= city)== 1 , name=f'flow_out{city}')

#why we cannot use DFJ ????????????
# set MTZ variables Miller Tucker Zemlin
 #Lambda :In Python, a lambda is a way to create a small, anonymous function (a function with no name) in one line.
 #ex: lambda x: x + 2
#   ________________   MTZ   _______________   :
mtz_vars = {}
cityCount = len(cities)
startCity = 'New York'

for city in cities:
    if city != startCity:
        mtz_vars[city] = m.addVar(vtype=GRB.CONTINUOUS, lb=2, ub=cityCount, name=f'u-{city}')
#why we are adding start city for mtz variables??????????
mtz_vars[startCity] = m.addVar(vtype=GRB.CONTINUOUS, lb=1 , ub=1, name=f'u-1')

#Making a new list containing non-origin cities,
non_initial_cities = list(filter(lambda selectedCities: selectedCities != startCity, cities))

for i in non_initial_cities:
    for j in non_initial_cities:
        if i != j and (i,j) in flow:
            #print(f'from {i} to {j}')
    #        flow_mtz= flow[(i,j)]
            constr_mtz = m.addLConstr(mtz_vars[j] - mtz_vars[i] +1 <= (cityCount +1)*(1- flow[i,j]), name=f'mtz_{i}_{j}')

m.optimize()

if m.status == GRB.Status.OPTIMAL:
    print(f'\nOptimal solution : {m.ObjVal:g}')
    print('SOLUTION:')

    for arc in arcs:
        if flow[arc].X >= 0.99:
            print(f'--{arc}-- distance: {distances[arc]} km')
else:
    print('No solution found')

m.dispose()







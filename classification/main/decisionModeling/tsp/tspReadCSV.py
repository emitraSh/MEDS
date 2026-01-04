import itertools
import gurobipy as gp
import csv
from gurobipy import GRB


arcs = set()
distances =  dict()
cities = set()

# Read CSV file, set encoding and delimiter
with open('../../../../Downloads/distanceMatrixCities6.csv', mode='r', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file, delimiter=';')

    for row in reader:
        origin = int(row['origin'])
        destination = int(row['destination'])
        distance = int(row['distance'])

        distances[(origin, destination)] = distance
        cities.add(origin)
        cities.add(destination)



# Extract arcs and cities
arcs.update(distances.keys())
cities = sorted(cities)


print("Cities:", cities)
for(i,j) in arcs:
    print(f"origin: {i} -- destination: {j}: distance {distances[(i,j)]} km")

model = gp.Model("TSP")


ds= {}
for d in distances:
    ds[d] = model.addVar(vtype=GRB.BINARY,name=f"Distance_{d[0]},{d[1]}")

networkDistance = gp.LinExpr()
for v in distances:
    networkDistance += ds[v] * distances[v]

model.setObjective(networkDistance, GRB.MINIMIZE)

"""for city in cities:
    flow_out= gp.LinExpr()
    flow_in= gp.LinExpr()
    for arc in arcs:
        if arc[1] == city:"""

def sublistsOfSize (lst, minsize, maxsize):
    sublists = []
    for r in range(minsize, maxsize+1):
        sublists.extend(itertools.combinations(lst,r))
    return [list(sublist) for sublist in sublists]

def addSTEConstraints(cities,movement_vars,model):
    subsets = list(sublistsOfSize(cities,2,len(cities)-1))

    for s in subsets:
        arcs_in_subsets = gp.LinExpr()
        for origin in s :
            for destination in s:
                if(origin,destination) in movement_vars:
                    arcs_in_subsets += movement_vars[(origin,destination)]

        model.addLConstr(arcs_in_subsets - len(s) + 1 <= 0, name=f"{origin},{destination}")




#
#  if (i ,j) in useLink:
# #         return useLink[(i, j)]
# #     else:
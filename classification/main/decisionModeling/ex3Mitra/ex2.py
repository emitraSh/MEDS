import csv
import gurobipy as gp
from gurobipy import GRB

arcs = set()
cities = set()
distances = dict()

with open('travelTimes50Cities.csv', mode='r', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file, delimiter=';')
    for row in reader:
        origin = row['origin']
        destination = row['destination']
        distance = float(row['distance_miles'])  # Assuming in miles

        cities.add(origin)
        cities.add(destination)
        distances[(origin, destination)] = distance

arcs = list(distances.keys())
cities = sorted(cities)
cityCount = len(cities)
startCity = 'New York'
cities = cities[:20]  # Keep only the first 20 cities
arcs = [(i, j) for i in cities for j in cities if i != j]
distances = {k: v for k, v in distances.items() if k[0] in cities and k[1] in cities}

# --- VRP Extension ---
crews = ['crew1', 'crew2', 'crew3']
K = crews

# --- Model ---
vrp = gp.Model("VRP_MTZ")
x = {
    (i, j, k): vrp.addVar(vtype=GRB.BINARY, obj=distances[(i, j)], name=f"x_{i}_{j}_{k}")
    for (i, j) in arcs if i != j for k in K
}
vrp.ModelSense = GRB.MINIMIZE

# --- Each city visited once ---
for c in cities:
    if c != startCity:
        vrp.addConstr(gp.quicksum(x[i, c, k] for i in cities if i != c for k in K if (i, c) in distances) == 1)
        vrp.addConstr(gp.quicksum(x[c, j, k] for j in cities if j != c for k in K if (c, j) in distances) == 1)

# --- Crew flow consistency ---
for k in K:
    for c in cities:
        if c != startCity:
            vrp.addConstr(gp.quicksum(x[i, c, k] for i in cities if i != c and (i, c) in distances) ==
                          gp.quicksum(x[c, j, k] for j in cities if j != c and (c, j) in distances))

# --- MTZ Variables for each crew ---
u = { (c, k): vrp.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=200, name=f"u_{c}_{k}")
      for c in cities if c != startCity for k in K }

for k in K:
    for i in cities:
        for j in cities:
            if i != j and i != startCity and j != startCity and (i, j) in distances:
                vrp.addConstr(u[j, k] - u[i, k] + 1 <= 200 * (1 - x[i, j, k]))

# --- Depot constraints ---
for k in K:
    vrp.addConstr(gp.quicksum(x[startCity, j, k] for j in cities if j != startCity and (startCity, j) in distances) == 1)
    vrp.addConstr(gp.quicksum(x[i, startCity, k] for i in cities if i != startCity and (i, startCity) in distances) == 1)

vrp.optimize()

import csv
import gurobipy as gp
from gurobipy import GRB

# --- Read CSV and Extract Data ---
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

# --- Model ---
m = gp.Model("TSP_MTZ")
flow = {
    (i, j): m.addVar(vtype=GRB.BINARY, obj=distances[(i, j)], name=f"x_{i}_{j}")
    for (i, j) in arcs if i != j
}
m.ModelSense = GRB.MINIMIZE

# --- Constraints: Flow In & Out ---
for c in cities:
    m.addConstr(gp.quicksum(flow[i, c] for i in cities if i != c and (i, c) in flow) == 1, f"in_{c}")
    m.addConstr(gp.quicksum(flow[c, j] for j in cities if j != c and (c, j) in flow) == 1, f"out_{c}")

# --- MTZ Variables & Constraints ---
u = {}
for c in cities:
    if c == startCity:
        u[c] = m.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=1, name=f"u_{c}")
    else:
        u[c] = m.addVar(vtype=GRB.CONTINUOUS, lb=2, ub=cityCount, name=f"u_{c}")

for i in cities:
    for j in cities:
        if i != j and (i, j) in flow and i != startCity and j != startCity:
            m.addConstr(u[j] - u[i] + 1 <= cityCount * (1 - flow[i, j]), f"mtz_{i}_{j}")

m.optimize()

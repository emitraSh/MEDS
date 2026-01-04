import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

flow = pd.read_csv("/Users/ngoclinhdao/Downloads/MS2 HW 4/flow_matrix.csv", delimiter =";")
distance = pd.read_csv("/Users/ngoclinhdao/Downloads/MS2 HW 4/distance_matrix.csv", delimiter = ";")

machines = sorted(set(flow['machineA']).union(flow['machineB']))
locations = sorted(set(distance['locationA']).union(distance['locationB']))

flow_matrix = pd.DataFrame(0, index=machines, columns=machines)
dist_matrix = pd.DataFrame(0, index=locations, columns=locations)

for _, row in flow.iterrows():
    i, j, val = row['machineA'], row['machineB'], int(row['flow'])
    flow_matrix.loc[i, j] = val
    flow_matrix.loc[j, i] = val

for _, row in distance.iterrows():
    k, l, val = row['locationA'], row['locationB'], int(row['distance'])
    dist_matrix.loc[k, l] = val
    dist_matrix.loc[l, k] = val

m = gp.Model("Factory Assigning Problem")

x = {(i, k): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")
    for i in machines for k in locations}
m.update()

obj = gp.QuadExpr()
for i in machines:
    for j in machines:
        for k in locations:
            for l in locations:
                obj += flow_matrix.loc[i, j] * dist_matrix.loc[k, l] * x[i, k] * x[j, l]


m.setObjective(obj, GRB.MINIMIZE)

for i in machines:
    expr = gp.LinExpr()
    for k in locations:
        expr += x[i, k]
    m.addConstr(expr == 1, name=f"machine_{i}")

for k in locations:
    expr = gp.LinExpr()
    for i in machines:
        expr += x[i, k]
    m.addConstr(expr == 1, name=f"location_{k}")

m.optimize()

if m.status == GRB.OPTIMAL:
    print("\nOptimal Machine-to-Location Assignments:")
    for i in machines:
        for k in locations:
            if x[i, k].X > 0.5:
                print(f"{i} is assigned to {k}")
    print(f"\nTotal transportation cost: {m.ObjVal:.2f}")
else:
    print("No optimal solution found.")

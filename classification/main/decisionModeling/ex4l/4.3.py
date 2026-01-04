import gurobipy as gp
from gurobipy import Model, GRB

model = Model("ProductMix_Quadratic")

r1 = model.addVar(vtype = GRB.CONTINUOUS, lb=0, name="r1")
r2 = model.addVar(vtype = GRB.CONTINUOUS, lb=0, name="r2")

profit = gp.QuadExpr()
profit += 200 * r1
profit += 200 * r2
profit += -100 * r1 * r1
profit += -100 * r2 * r2

model.setObjective(profit, GRB.MAXIMIZE)

model.addLConstr(r1 + r2 <= 2, "ProductionLimit")

model.optimize()

if model.status == GRB.OPTIMAL:
    print(f"Optimal r1: {r1.X:.4f} units/hour")
    print(f"Optimal r2: {r2.X:.4f} units/hour")
    print(f"Total Profit per Hour: ${model.ObjVal:.2f}")
else:
    print("No optimal solution found.")


from gurobipy import Model, GRB

model = Model("ProductMix_PriceElasticity")

x1 = model.addVar(lb=0, name="x1")
x2 = model.addVar(lb=0, name="x2")
x3 = model.addVar(lb=0, ub=20, name="x3") 


model.setObjective(
    10 * x1 + (1/100) * x1 * x1 +
    5 * x2 + (1/40) * x2 * x2 +
    5 * x3 + (1/50) * x3 * x3,
    GRB.MAXIMIZE
)

model.addLConstr(9 * x1 + 3 * x2 + 5 * x3 <= 500, name="milling")
model.addLConstr(5 * x1 + 4 * x2 <= 350, name="lathe")
model.addLConstr(3 * x1 + 2 * x3 <= 150, name="grinder")

model.optimize()

if model.status == GRB.OPTIMAL:
    print("\nOptimal production quantities:")
    print(f"P1 (x1): {x1.X:.2f} units")
    print(f"P2 (x2): {x2.X:.2f} units")
    print(f"P3 (x3): {x3.X:.2f} units")
    print(f"Maximum Profit: ${model.ObjVal:.2f}")
else:
    print("No optimal solution found.")

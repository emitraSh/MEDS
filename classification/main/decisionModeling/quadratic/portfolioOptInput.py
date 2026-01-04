import itertools
from statistics import covariance

import gurobipy as gp
import csv
from gurobipy import GRB

stocks = ["AAPL", "TSLA", "DIS", "AMD"]
expectedReturns = {
    "AAPL": 0.268385,
    "TSLA": 0.475549,
    "DIS": 0.114966,
    "AMD": 0.209862
}
standardDeviations = {
    "AAPL": 0.081699,
    "TSLA": 0.329160,
    "DIS": 0.068223,
    "AMD": 0.326817
}

Covariances = {

    ("AAPL", "TSLA",): 0.058321,
    ("AAPL", "DIS",): 0.031096,
    ("AAPL", "AMD",): 0.063529,

    ("TSLA", "DIS",): 0.047962,
    ("TSLA", "AMD",): 0.047868,

    ("DIS", "AMD",): 0.047868,
}

stockPrices = {
    "AAPL": 210,
    "TSLA": 318,
    "DIS": 110,
    "AMD": 97
}

minimalReturns = 0.2
"""
m= gp.Model("QuadraticPortfolio")

weight = {}
for w in stocks:
    weight[w] = m.addVar(vtype=GRB.CONTINUOUS,lb=0, ub=1,name=f"weight({w})")

optimization= gp.QuadExpr()
for (stockA, stockB) in Covariances:
    optimization +=  Covariances[(stockA, stockB)] * weight[stockA] * weight[stockB]

for sd in standardDeviations:
    optimization += (standardDeviations[sd]**2) * weight[sd]**2


m.setObjective(optimization, GRB.MINIMIZE)


sumWeight = m.addConstr((gp.quicksum(weight[w] for w in stocks) == 1), name=f"sumWeight{w}")


returnLimit = gp.LinExpr()
for s in stocks:
    returnLimit += weight[s] *expectedReturns[s]
for s in stocks:
    stockConst = m.addConstr(returnLimit >= minimalReturns, name=f"expectedReturn{s}")


m.optimize()
    
if m.status == GRB.OPTIMAL:
    print("------------------The Solution:---------------------")
    for w in stocks:
        print(f"Stock {w} - weight {weight[w].X:.4f}")
else:
    print("No solution found.")
"""

# Chat GPT version ####################



m = gp.Model("QuadraticPortfolio")

# Decision variables
weight = {w: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"weight({w})") for w in stocks}

# Create a symmetric covariance matrix
full_cov = {}
for (a, b), val in Covariances.items():
    full_cov[(a, b)] = val
    full_cov[(b, a)] = val
for s in stocks:
    full_cov[(s, s)] = standardDeviations[s] ** 2

# Objective: Minimize portfolio variance
objective = gp.QuadExpr()
for i in stocks:
    for j in stocks:
        objective += full_cov[(i, j)] * weight[i] * weight[j]
m.setObjective(objective, GRB.MINIMIZE)

# Constraint: weights sum to 1
m.addConstr(gp.quicksum(weight[w] for w in stocks) == 1, name="sumWeights")

# Constraint: expected return >= minimum
expected_return = gp.quicksum(expectedReturns[s] * weight[s] for s in stocks)
m.addConstr(expected_return >= minimalReturns, name="expectedReturn")

# Solve
m.optimize()

# Output
if m.status == GRB.OPTIMAL:
    print("------------------The Solution:---------------------")
    for w in stocks:
        print(f"Stock {w} - weight {weight[w].X:.4f}")
else:
    print("No solution found.")
    
#"""

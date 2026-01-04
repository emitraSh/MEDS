import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

mu_df = pd.read_csv("/Users/ngoclinhdao/Downloads/MS2 HW 4/100stocks_mean_returns.csv", delimiter = ',')
cov_df= pd.read_csv("/Users/ngoclinhdao/Downloads/MS2 HW 4/100stocks_unique_cov_matrix.csv", delimiter = ';')

stocks = mu_df['Stock'].tolist()

expectedReturns = dict(zip(mu_df['Stock'], mu_df['ExpectedReturn']))
n = len(stocks)

minReturn = 0.15

print(cov_df.columns.tolist()) 

#Creates a mapping from stock names to matrix indices
idx = {stock: i for i, stock in enumerate(stocks)}

cov_matrix = np.zeros((n, n))
for _, row in cov_df.iterrows():
    stockA = row['stockA']
    stockB = row['stockB']
    cov = row['covariance']
    
    i = idx[stockA]
    j = idx[stockB]
    cov_matrix[i, j] = cov_matrix[j, i] = cov

model = gp.Model("StockBroker")

weight ={}

for stock in stocks:
    weight[stock] = model.addVar(vtype = GRB.CONTINUOUS, lb=0, ub=1, name = f"weight_{stock}")
    model.update()

aggregateWeight = gp.LinExpr()
for stock in stocks:
    aggregateWeight += weight[stock]

model.addLConstr(aggregateWeight == 1.0, name = f"aggregateWeight")

expectedReturn = gp.LinExpr()
for stock in stocks:
    expectedReturn += expectedReturns[stock]*weight[stock]
model.addLConstr(expectedReturn >= minReturn, name = f"minimumReturn")


expectedRisk = gp.QuadExpr()
for i in range(n):
    for j in range(n):
        expectedRisk += cov_matrix[i, j] * weight[stocks[i]] * weight[stocks[j]]

model.setObjective(expectedRisk, GRB.MINIMIZE)

model.optimize()

if model.status != GRB.INFEASIBLE:
    print(f"Risk: {model.ObjVal:g}")
    print(f"Shares:")
    for stock in stocks:
        if (weight[stock].X*100 > 1.0):
            print(f"{stock} -- {round(100 * weight[stock].X, 2):} % ")
else:
    print("Not optimal")

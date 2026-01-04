import gurobipy as gp
import networkx as nx

from gurobipy import GRB

graph = nx.DiGraph()

links, capacity = gp.multidict({
    ("R1","A") : 75,
    ("R1","B") : 65,
    ("R2","A") : 40,
    ("R2","B") : 50,
    ("R2","C") : 60,
    ("R3","B") : 80,
    ("R3","C") : 70,
    ("A","D") : 60,
    ("A","E") : 45,
    ("B","D") : 70,
    ("B","E") : 55,
    ("B","F") : 45,
    ("C","E") : 70,
    ("C","F") : 90,
    ("D","T") : 120,
    ("E","T") : 190,
    ("F","T") : 130,
})

graph.add_edges_from(links)


mfm = gp.Model("Flow")


flow= {}

for l in links:
    flow[l] = mfm.addVar(vtype=GRB.INTEGER,lb=0, ub=capacity[l],name=f"flow_{l[0]},{l[1]}")

objFlowOut = 0
for j in graph.predecessors("T"):
    objFlowOut += flow[j,"T"]

mfm.setObjective(objFlowOut, GRB.MAXIMIZE)

for v in graph.nodes():
    if v not in ["R1", "R2", "R3","T"]:
        ingoingFlow = gp.quicksum(flow[i,v] for i in graph.predecessors(v))
        outgoingFlow = gp.quicksum(flow[v,j] for j in graph.successors(v))

        mfm.addLConstr(ingoingFlow - outgoingFlow == 0, name=f"constraint_{v}")


for e in links:
    mfm.addConstr(flow[e] <= capacity[e])

mfm.optimize()

if mfm.status == GRB.Status.OPTIMAL:
    print(f"\nThe optimal flow is {mfm.objVal:g}")
    print("------------------The Solution:---------------------")

    for l in links:
        if flow[l].X > 0.0:
            print(f"Link {l} - flow {flow[l].X} out of Capacity {capacity[l]}")
        else:
            print("no solution is found")


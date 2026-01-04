import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from decisionModeling.seervadaParkShortestInput import useLink, inGoingToDestination

graph = nx.DiGraph()

links, flyingTimes = gp.multidict({
    ("SEA","A"): 4.6,
    ("SEA","B"): 4.7,
    ("SEA","C"): 4.2,
    ("A","D"): 3.5,
    ("A","E"): 3.4,
    ("B","D"): 3.6,
    ("B","E"): 3.2,
    ("B","F"): 3.3,
    ("C","E"): 3.5,
    ("C","F"): 3.4,
    ("D","LHG"): 3.4,
    ("E","LHG"): 3.6,
    ("F","LHG"): 3.8,
})

graph.add_edges_from(links)
print("Flight Plan Input")
print(graph)
print("Nodes:", graph.nodes())
for l in links:
    print(f"link: {l}, flying time: {flyingTimes[l]}.")

#print(list(graph.successors("SEA")))

spm = gp.Model("Flight Plan - Shortest Time")

#adding binary variables for each link
useLink = {}
for l in links:
    useLink[l] = spm.addVar(vtype=GRB.BINARY, name = f"useLink_({l[0]},{l[1]})")

#adding objective
#LinExpr() is a class for building linear expressions manually — sums of variables times coefficients, specially in loops
objFlyingTime =gp.LinExpr()
for l in links:
    objFlyingTime += flyingTimes[l] * useLink[l]

spm.setObjective(objFlyingTime)
spm.ModelSense = 1 #Minimization

### Constraints
for node in graph.nodes():
    if node not in ["SEA","LHG"]:
        outgoing = gp.quicksum(useLink[node, j] for j in graph.successors(node))
        ingoing = gp.quicksum(useLink[i,node] for i in graph.predecessors(node))

        spm.addLConstr(outgoing - ingoing == 0, name= f"Fow Constraint{node}")

outgoingOrigin = gp.quicksum(useLink["SEA", m] for m in graph.successors("SEA"))
spm.addLConstr(outgoingOrigin ==1 , name= "Origin Constraint")

inGoingDestination = gp.quicksum(useLink[j, "LHG"] for j in graph.predecessors("LHG"))
spm.addLConstr(inGoingDestination == -1, name= "Destination Constraint")

spm.optimize()

## Print Output
if spm.status == GRB.OPTIMAL:
    print(f"\nTOTAL COSTS: {spm.ObjVal:g}")
    print("SOLUTION:")
    for l in links:
        if useLink[l].X > 0.99:
            print(f"Use link {l} -  Flying Time {flyingTimes[l]} .")

else:
    print("No optimal solution found.")
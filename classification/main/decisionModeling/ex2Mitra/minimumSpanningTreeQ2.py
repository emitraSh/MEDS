import itertools
import gurobipy as gp
import networkx as nx
from gurobipy import GRB

graph= nx.Graph()

def sublistsOfSize(lst, minsize, maxsize):
    sublists = []
    for r in range(minsize, maxsize+1):
        sublists.extend(itertools.combinations(lst,r))
    return [list(sublist) for sublist in sublists]

links, distances = gp.multidict({
    ("1","2"): 1.3,
    ("1","3"): 2.1,
    ("1","4"): 0.9,
    ("1","5"): 0.7,
    ("1","6"): 1.8,
    ("1","7"): 2.0,
    ("1","8"): 1.5,
    ("2","3"): 0.9,
    ("2","4"): 1.8,
    ("2","5"): 1.2,
    ("2","6"): 2.6,
    ("2","7"): 2.3,
    ("2","8"): 1.1,
    ("3","4"): 2.6,
    ("3","5"): 1.7,
    ("3","6"): 2.5,
    ("3","7"): 1.9,
    ("3","8"): 1.0,
    ("4","5"): 0.7,
    ("4","6"): 1.6,
    ("4","7"): 1.5,
    ("4","8"): 0.9,
    ("5","6"): 0.9,
    ("5","7"): 1.1,
    ("5","8"): 0.8,
    ("6","7"): 0.6,
    ("6","8"): 1.0,
    ("7","8"): 0.5,
})

graph.add_edges_from(links)

m = gp.Model("Minimum Spanning Tree of Tree")

useLink= {} # why we use set?
for l in links:
    useLink[l] = m.addVar(vtype=GRB.BINARY, name=f"link_{l[0]},{l[1]}")

networkDistance = gp.LinExpr()
for l in links:
    networkDistance += useLink[l] * distances[l]

m.setObjective(networkDistance)
m.ModelSense = 1


maxNodes = len(graph.nodes)-1
maxNodesConstr = gp.quicksum(useLink[l] for l in links)
m.addLConstr(maxNodes == maxNodesConstr, name="maxNodesConstr")

#For each subset S, we add up all links that actually exist in the potential network.
# Generate all subsets of nodes of size between 2 and (n - 1)
subsets = list(sublistsOfSize(graph.nodes,2,maxNodes))

# For each subset, apply subtour elimination constraint

for s in subsets:
    links_in_subset = gp.LinExpr()
    print(s)

    # Sum all links within the subset
    for i in s:
        for j in s:
            if (i, j) in useLink:
                links_in_subset += useLink[(i, j)]
    m.addLConstr(links_in_subset - len(s) + 1 <= 0 , name =f"s") #I couldn't write a name it became too long

print(len(subsets))


m.optimize()

## Print Output
if m.status == GRB.OPTIMAL:
    print(f"\nTOTAL COSTS: {m.ObjVal:g}")
    print("SOLUTION:")
    for l in links:
        if useLink[l].X > 0.99:
            print(f"Use link {l} -  Distances {distances[l]} .")

else:
    print("No optimal solution found.")

m.write("lll.lp")
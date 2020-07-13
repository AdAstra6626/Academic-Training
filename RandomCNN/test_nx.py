#%%
import networkx
from igraph import *
 
edges = [1,2,2,3,1,3]

g = graph()
g.add_vertices(3)
g.add_edges(edges)

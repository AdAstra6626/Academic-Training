from __future__ import division
import networkx as nx 
import networkx.algorithms.smallworld as sm 
import random


def gen_right_neighbors(nodes, node, K):
    rest_right = len(nodes) - node - 1
    result = []
    if rest_right >= K:
        for i in range(K):
            result.append(node + i + 1)
        return result
    else:
        temp = K - rest_right
        for i in range(rest_right):
            result.append(node + i + 1)
        for i in range(temp):
            result.append(i)
        return result
def gen_left_neighbors(nodes, node, K):
    rest_left = node
    result = []
    if rest_left >= K:
        for i in range(K):
            result.append(node - i - 1)
        return result
    else:
        temp = K - rest_left
        for i in range(rest_left):
            result.append(node - i - 1)
        for i in range(temp):
            result.append(len(nodes) - i - 1)
        return result

def get_neighbors(graph, node):
    return list(graph.neighbors(node))

def rewire(graph, node, i, j):
    graph.add_edge(node, i)
    graph.remove_edge(node, j)

def rand_select(neighbors):
    length = len(neighbors)
    rand = random.randint(0, length - 1)
    return neighbors[rand]

def rewire_node(graph, node, K, p):
    nodes = graph.nodes 
    right_neighbors = gen_right_neighbors(nodes, node, K)
    left_neighbors = gen_left_neighbors(nodes, node, K)
    for i in range(K):
        rand = random.random()
        if rand < p:
            neighbors = get_neighbors(graph, node)
            neighbors = list(set(neighbors) - set(right_neighbors) - set(left_neighbors))
            if len(neighbors) == 0:
                graph.add_edge(node, right_neighbors[i])
                continue
            j = rand_select(neighbors)
            rewire(graph, node, right_neighbors[i], j)
    for i in range(K):
        rand = random.random()
        if rand < p:
            neighbors = get_neighbors(graph, node)
            neighbors = list(set(neighbors) - set(right_neighbors) - set(left_neighbors))
            if len(neighbors) == 0:
                graph.add_edge(node, left_neighbors[i])
                continue
            j = rand_select(neighbors)
            rewire(graph, node, left_neighbors[i], j)

def rewire_graph(graph, K, p):
    for node in graph.nodes:
        rewire_node(graph, node, K, p)
    return graph 
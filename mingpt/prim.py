from collections import defaultdict
from itertools import combinations
from heapq import *
from tqdm import tqdm
import random

import numpy
from math import log

def func(x, a, b):
    y = a * log(x) + b
    return y

def prim_real(vertexs_tmp, node_pos, net_info, ratio, node_info, port_info, fixed_node_info,
    placedb):
    max_x = 0.0
    min_x = placedb.max_height * 1.5
    max_y = 0.0
    min_y = placedb.max_height * 1.5
    vertexs = list(vertexs_tmp)
    if len(vertexs)<=1:
        return 0
    adjacent_dict = {}
    for node in vertexs:
        adjacent_dict[node] = []
    for node1, node2 in list(combinations(vertexs, 2)):
        if node1 in node_pos:
            pin_x_1 = node_pos[node1][0] * ratio + node_info[node1]["x"] / 2 + net_info[node1]["x_offset"]
            pin_y_1 = node_pos[node1][1] * ratio + node_info[node1]["y"] / 2 + net_info[node1]["y_offset"]
        elif node1 in port_info:
            pin_x_1 = port_info[node1]['x']
            pin_y_1 = port_info[node1]['y']
        else:
            assert node1 in fixed_node_info
            pin_x_1 = fixed_node_info[node1]['raw_x'] + fixed_node_info[node1]["x"] / 2 + net_info[node1]["x_offset"]
            pin_y_1 = fixed_node_info[node1]['raw_y'] + fixed_node_info[node1]["y"] / 2 + net_info[node1]["y_offset"]
        if node2 in node_pos:
            pin_x_2 = node_pos[node2][0] * ratio + node_info[node2]["x"] / 2 + net_info[node2]["x_offset"]
            pin_y_2 = node_pos[node2][1] * ratio + node_info[node2]["y"] / 2 + net_info[node2]["y_offset"]
        elif node2 in port_info:
            pin_x_2 = port_info[node2]['x']
            pin_y_2 = port_info[node2]['y']
        else:
            assert node2 in fixed_node_info
            pin_x_2 = fixed_node_info[node2]['raw_x'] + fixed_node_info[node2]["x"] / 2 + net_info[node2]["x_offset"]
            pin_y_2 = fixed_node_info[node2]['raw_y'] + fixed_node_info[node2]["y"] / 2 + net_info[node2]["y_offset"]
        weight = abs(pin_x_1-pin_x_2) + \
                abs(pin_y_1-pin_y_2)
        max_x = max(pin_x_1, max_x)
        min_x = min(pin_x_1, min_x)
        max_y = max(pin_y_1, max_y)
        min_y = min(pin_y_1, min_y)
        max_x = max(pin_x_2, max_x)
        min_x = min(pin_x_2, min_x)
        max_y = max(pin_y_2, max_y)
        min_y = min(pin_y_2, min_y)
        adjacent_dict[node1].append((weight, node1, node2))
        adjacent_dict[node2].append((weight, node2, node1))
    '''
    {'A': [(7, 'A', 'B'), (5, 'A', 'D')], 
     'C': [(8, 'C', 'B'), (5, 'C', 'E')], 
     'B': [(7, 'B', 'A'), (8, 'B', 'C'), (9, 'B', 'D'), (7, 'B', 'E')], 
     'E': [(7, 'E', 'B'), (5, 'E', 'C'), (15, 'E', 'D'), (8, 'E', 'F'), (9, 'E', 'G')], 
     'D': [(5, 'D', 'A'), (9, 'D', 'B'), (15, 'D', 'E'), (6, 'D', 'F')], 
     'G': [(9, 'G', 'E'), (11, 'G', 'F')], 
     'F': [(6, 'F', 'D'), (8, 'F', 'E'), (11, 'F', 'G')]})
    '''
    start = vertexs[0]
    minu_tree = []
    visited = set()
    visited.add(start)
    adjacent_vertexs_edges = adjacent_dict[start]
    heapify(adjacent_vertexs_edges)
    cost = 0
    cnt = 0
    while cnt < len(vertexs)-1:
        weight, v1, v2 = heappop(adjacent_vertexs_edges)
        if v2 not in visited:
            visited.add(v2)
            minu_tree.append((weight, v1, v2))
            cost += weight
            cnt += 1
            for next_edge in adjacent_dict[v2]:
                if next_edge[2] not in visited:
                    heappush(adjacent_vertexs_edges, next_edge)
    return cost


def rand_generate_point(num_node):
    point_set = set()
    node_pos = {}
    min_x = 32
    min_y = 32
    max_x = 0
    max_y = 0
    for i in range(num_node):
        x = random.randint(0, 31)
        y = random.randint(0, 31)
        while (x, y) in point_set:
            x = random.randint(0, 31)
            y = random.randint(0, 31)
        point_set.add((x,y))
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    vertexs = list(range(num_node))
    point_set = list(point_set)
    for i, (x, y) in enumerate(point_set):
        node_pos[i] = (x, y)
    return vertexs, node_pos, min_x, min_y, max_x, max_y


def statistics():
    x = []
    y = []
    for i in range(349,350):
        sum_cost = 0
        for j in range(100):
            vertexs, node_pos, min_x, min_y, max_x, max_y = rand_generate_point(i)
            cost = prim(vertexs, node_pos)
            sum_cost += cost/((max_y-min_y)+(max_x-min_x))
        print("i = {}, avg_cost = {:.2f}".format(i, sum_cost/100.0))
        x.append(i)
        y.append(sum_cost/100.0)
    output= {}
    for a,b in zip(x,y):
        output[a] = b 
    print(output)

if __name__ == "__main__":
    statistics()


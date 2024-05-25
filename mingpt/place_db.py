import numpy as np
import os
from operator import itemgetter
from itertools import combinations
import pickle
import networkx as nx
# Macro dict (macro id -> name, x, y)

def read_node_file(fopen, benchmark):
    node_info = {}
    node_info_raw_id_name ={}
    port_info = {}
    fixed_node_info = {}
    node_cnt = 0
    node_range_list = {}
    if benchmark == "bigblue2" or benchmark == "bigblue4" or "ibm" in benchmark:
        node_range_list = pickle.load(open('node_list_{}_1024.pkl'.format(benchmark),'rb'))
    for line in fopen.readlines():
        if not line.startswith("\t") and not line.startswith(" "):
            continue
        line = line.strip().split()
        if "ibm" in benchmark:
            node_name = line[0]
            x = int(line[1])
            y = int(line[2])
            if node_name.startswith("p"):
                port_info[node_name] = {"x": x, "y": y}
                continue
            if len(node_range_list) > 0 and node_name not in node_range_list:
                continue
            node_info[node_name] = {"id": node_cnt, "x": x , "y": y, 'fixed': False }
            node_info_raw_id_name[node_cnt] = node_name
            node_cnt += 1
        else:
            if line[-1] != "terminal":
                continue
            node_name = line[0]
            if (benchmark == "bigblue2" or benchmark == "bigblue4") and node_name not in node_range_list:
                continue
            x = int(line[1])
            y = int(line[2])
            node_info[node_name] = {"id": node_cnt, "x": x , "y": y, 'fixed': False}
            if benchmark == "adaptec1" or benchmark == "bigblue1":
                if (x == 432 and y == 72) or (x == 72 and y == 432):
                    node_info[node_name]['fixed'] = True
                    continue
            elif benchmark == "adaptec2":
                if (x == 576 and y == 192) or (x == 192 and y == 576) or (x == 96 and y == 576) or (x == 576 and y == 96):
                    node_info[node_name]['fixed'] = True
                    continue
            node_info_raw_id_name[node_cnt] = node_name
            node_cnt += 1
    print("len node_info", len(node_info))
    return node_info, node_info_raw_id_name, port_info


def read_net_file(fopen, node_info, port_info):
    net_info = {}
    net_name = None
    net_cnt = 0
    for line in fopen.readlines():
        if not line.startswith("\t") and not line.startswith("  ") and \
            not line.startswith("NetDegree"):
            continue
        line = line.strip().split()
        if line[0] == "NetDegree":
            net_name = line[-1]
        else:
            node_name = line[0]
            if node_name in node_info or node_name in port_info:
                if not net_name in net_info:
                    net_info[net_name] = {}
                    net_info[net_name]["nodes"] = {}
                    net_info[net_name]["ports"] = {}
                if node_name in node_info:
                    x_offset = float(line[-2])
                    y_offset = float(line[-1])
                    net_info[net_name]["nodes"][node_name] = {}
                    net_info[net_name]["nodes"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
                else:
                    if len(line) >= 3:
                        x_offset = float(line[-2])
                        y_offset = float(line[-1])
                    else:
                        x_offset = 0.0
                        y_offset = 0.0
                    net_info[net_name]["ports"][node_name] = {}
                    net_info[net_name]["ports"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
    for net_name in list(net_info.keys()):
        if len(net_info[net_name]["nodes"]) <= 1:
            net_info.pop(net_name)
    for net_name in net_info:
        net_info[net_name]['id'] = net_cnt
        net_cnt += 1
    print("adjust net size = {}".format(len(net_info)))
    return net_info


def get_comp_hpwl_dict(node_info, net_info):
    # node_name
    comp_hpwl_dict = {}
    for net_name in net_info:
        max_idx = 0
        for node_name in net_info[net_name]["nodes"]:
            max_idx = max(max_idx, node_info[node_name]["id"])
        if not max_idx in comp_hpwl_dict:
            comp_hpwl_dict[max_idx] = []
        comp_hpwl_dict[max_idx].append(net_name)
    return comp_hpwl_dict


# node_to_net_set[node_name] = {'net_name_1', 'net_name_2', ..., 'net_name_n'}
def get_node_to_net_dict(node_info, net_info):
    node_to_net_dict = {}
    for node_name in node_info:
        node_to_net_dict[node_name] = set()
    for net_name in net_info:
        for node_name in net_info[net_name]["nodes"]:
            node_to_net_dict[node_name].add(net_name)
    return node_to_net_dict


def get_port_to_net_dict(port_info, net_info):
    port_to_net_dict = {}
    for port_name in port_info:
        port_to_net_dict[port_name] = set()
    for net_name in net_info:
        for port_name in net_info[net_name]["ports"]:
            port_to_net_dict[port_name].add(net_name)
    return port_to_net_dict


def read_pl_file(fopen, node_info, port_info):
    max_height = 0
    max_width = 0
    for line in fopen.readlines():
        line = line.strip()
        if not line.startswith('o') and not line.startswith('p'):
            continue
        line = line.strip().split()
        node_name = line[0]
        if not node_name in node_info and not node_name in port_info:
            continue
        place_x = int(line[1])
        place_y = int(line[2])
        if node_name in node_info:
            max_height = max(max_height, node_info[node_name]["x"] + place_x)
            max_width = max(max_width, node_info[node_name]["y"] + place_y)
        elif node_name in port_info:
            max_height = max(max_height, port_info[node_name]["x"] + place_x)
            max_width = max(max_width, port_info[node_name]["y"] + place_y)
        if node_name in node_info:
            node_info[node_name]["raw_x"] = place_x
            node_info[node_name]["raw_y"] = place_y
        else:
            port_info[node_name]["raw_x"] = place_x
            port_info[node_name]["raw_y"] = place_y
    return max(max_height, max_width), max(max_height, max_width)


def read_scl_file(fopen, benchmark):
    assert "ibm" in benchmark
    for line in fopen.readlines():
        if not "Numsites" in line:
            continue
        line = line.strip().split()
        max_height = int(line[-1])
        break
    return max_height, max_height


def get_node_id_to_name(node_info, node_to_net_dict):
    node_name_and_num = []
    for node_name in node_info:
        node_name_and_num.append((node_name, len(node_to_net_dict[node_name])))
    node_name_and_num = sorted(node_name_and_num, key=itemgetter(1), reverse = True)
    print("node_name_and_num", node_name_and_num)
    node_id_to_name = [node_name for node_name, _ in node_name_and_num]
    for i, node_name in enumerate(node_id_to_name):
        node_info[node_name]["id"] = i
    return node_id_to_name


def get_node_id_to_name_topology(node_info, node_to_net_dict, net_info, benchmark):
    node_id_to_name = []
    adjacency = {}
    for net_name in net_info:
        for node_name_1, node_name_2 in list(combinations(net_info[net_name]['nodes'],2)):
            if node_name_1 not in node_info or node_name_2 not in node_info:
                continue
            if node_name_1 not in adjacency:
                adjacency[node_name_1] = set()
            if node_name_2 not in adjacency:
                adjacency[node_name_2] = set()
            adjacency[node_name_1].add(node_name_2)
            adjacency[node_name_2].add(node_name_1)

    visited_node = set()

    node_net_num = {}
    print("node_info len", len(node_info))
    for node_name in node_info:
        node_net_num[node_name] = len(node_to_net_dict[node_name])
    
    node_net_num_fea = {}
    node_net_num_max = max(node_net_num.values())
    print("node_net_num_max", node_net_num_max)
    for node_name in node_info:
        node_net_num_fea[node_name] = node_net_num[node_name]/node_net_num_max
    
    node_area_fea = {}
    node_area_max_node = max(node_info, key = lambda x : node_info[x]['x'] * node_info[x]['y'])
    node_area_max = node_info[node_area_max_node]['x'] * node_info[node_area_max_node]['y']
    print("node_area_max = {}".format(node_area_max))
    for node_name in node_info:
        node_area_fea[node_name] = node_info[node_name]['x'] * node_info[node_name]['y'] / node_area_max
    
    if "V" in node_info:
        add_node = "V"
        visited_node.add(add_node)
        node_id_to_name.append((add_node, node_net_num[add_node]))
        node_net_num.pop(add_node)
    
    add_node = max(node_net_num, key = lambda v: node_net_num[v])
    visited_node.add(add_node)
    node_id_to_name.append((add_node, node_net_num[add_node]))
    node_net_num.pop(add_node)
    while len(node_id_to_name) < len(node_info):
        candidates = {}
        for node_name in visited_node:
            if node_name not in adjacency:
                continue
            for node_name_2 in adjacency[node_name]:
                if node_name_2 in visited_node:
                    continue
                if node_name_2 not in candidates:
                    candidates[node_name_2] = 0
                candidates[node_name_2] += 1
        # for remove all uncertain macros
        if True:
            for node_name in node_info:
                if node_name not in candidates and node_name not in visited_node:
                    candidates[node_name] = 0
        if len(candidates) > 0:
            if benchmark == "bigblue3":
                add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*100000 +\
                    node_info[v]['x']*node_info[v]['y'] * 1 + int(v[1:])*1e-6)
            else:
                add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*1000 +\
                    node_info[v]['x']*node_info[v]['y'] * 1 + int(v[1:])*1e-8)
        else:
            if benchmark == "bigblue3" or "ibm" in benchmark:
                add_node = max(node_net_num, key = lambda v: node_net_num[v]*100000 + node_info[v]['x']*node_info[v]['y']*1+ int(v[1:])*1e-8)
            else:
                add_node = max(node_net_num, key = lambda v: node_net_num[v]*1000 + node_info[v]['x']*node_info[v]['y']*1+ int(v[1:])*1e-8)

        visited_node.add(add_node)
        node_id_to_name.append((add_node, node_net_num[add_node]))
        node_net_num.pop(add_node)
    for i, (node_name, _) in enumerate(node_id_to_name):
        node_info[node_name]["id"] = i
    print("node_id_to_name")
    print(node_id_to_name)
    node_id_to_name_res = [x for x, _ in node_id_to_name]
    return node_id_to_name_res


def get_pin_cnt(net_info):
    pin_cnt = 0
    for net_name in net_info:
        pin_cnt += len(net_info[net_name]["nodes"])
    return pin_cnt


def get_total_area(node_info):
    area = 0
    for node_name in node_info:
        area += node_info[node_name]["x"] * node_info[node_name]["y"]
    return area

def divide_node(node_info):
    new_node_info = {}
    fixed_node_info = {}
    for node_name in node_info:
        if not node_info[node_name]['fixed']:
            new_node_info[node_name] = node_info[node_name]
        else:
            fixed_node_info[node_name] = node_info[node_name]
    return new_node_info, fixed_node_info


class PlaceDB():

    def __init__(self, benchmark = None, offset = 0, is_graph = False):
        if benchmark is None:
            self.benchmark = None
            self.node_info, self.node_info_raw_id_name, self.port_info = None, None, None
            self.node_cnt = None
            self.net_info = None
            self.net_cnt = None
            self.max_height, self.max_width = None, None
            self.port_to_net_dict = None
            self.node_to_net_dict = None
            self.node_id_to_name = None
            self.node_name_to_id = None
        else:
            self.benchmark = benchmark
            self.offset = 0
            
            assert os.path.exists(benchmark)
            node_file = open(os.path.join(benchmark, benchmark+".nodes"), "r")
            self.node_info, self.node_info_raw_id_name, \
                self.port_info = read_node_file(node_file, benchmark)
            pl_file = open(os.path.join(benchmark, benchmark+".pl"), "r")
            node_file.close()
            net_file = open(os.path.join(benchmark, benchmark+".nets"), "r")
            self.net_info = read_net_file(net_file, self.node_info, self.port_info)
            self.net_cnt = len(self.net_info)
            net_file.close()
            pl_file = open(os.path.join(benchmark, benchmark+".pl"), "r")
            if benchmark == "adaptec1" or benchmark == "bigblue1":
                read_pl_file(pl_file, self.node_info, self.port_info)
                self.max_height, self.max_width = int(10000 * 1), int(10000 * 1)
                self.offset = 459
            elif benchmark == "adaptec2":
                read_pl_file(pl_file, self.node_info, self.port_info)
                self.max_height, self.max_width = int(14000 * 1), int(14000 * 1)
                self.offset = 616
            else:
                self.max_height, self.max_width = read_pl_file(pl_file, self.node_info, self.port_info)
            pl_file.close()
            if not "ibm" in benchmark:
                self.port_to_net_dict = {}
            else:
                self.port_to_net_dict = get_port_to_net_dict(self.port_info, self.net_info)
                scl_file = open(os.path.join(benchmark, benchmark+".scl"), "r")
                self.max_height, self.max_width = read_scl_file(scl_file, benchmark)

            self.node_to_net_dict = get_node_to_net_dict(self.node_info, self.net_info)
            self.node_info, self.fixed_node_info = divide_node(self.node_info)
            self.node_id_to_name = get_node_id_to_name_topology(self.node_info, self.node_to_net_dict, self.net_info, self.benchmark)
            self.node_name_to_id =  dict((t, i) for i, t in enumerate(self.node_id_to_name))
            self.node_cnt = len(self.node_info)
            self.circuit_fea = self.get_circuit_fea(is_graph=False)
            self.adj, self.features, self.graph = self.get_graph_fea(offset)
    
    def debug_str(self):
        print("node_cnt = {}".format(len(self.node_info)))
        print("fixed_node_cnt = {}".format(len(self.fixed_node_info)))
        print("net_cnt = {}".format(len(self.net_info)))
        print("max_height = {}".format(self.max_height))
        print("max_width = {}".format(self.max_width))
        print("pin_cnt = {}".format(get_pin_cnt(self.net_info)))
        print("port_cnt = {}".format(len(self.port_info)))
        print("area ratio = {}".format(get_total_area(self.node_info)/(self.max_height*self.max_height)))

    def draw_hist(self):
        areas = []
        neighbors = []
        visited_neighbours =[]

        effective_net = set()
        for net_name in self.net_info:
            cnt = 0
            for node_name in self.net_info[net_name]['nodes']:
                if self.node_name_to_id[node_name] < 256:
                    cnt += 1
            if cnt >=2:
                effective_net.add(net_name)
        for i in range(min(256, len(self.node_info))):
            node_name = self.node_id_to_name[i]
            area = self.node_info[node_name]['x'] * self.node_info[node_name]['y']
            area = area / (self.max_height * self.max_height)
            areas.append(area)
            cnt = 0
            for net_name in self.node_to_net_dict[node_name]:
                if net_name in effective_net:
                    cnt += 1
            neighbor = cnt
            neighbors.append(neighbor)
        X = np.arange(min(256, len(self.node_info)))

        visited_net = set()
        for i in range(min(256, len(self.node_info))):
            node_name = self.node_id_to_name[i]
            cnt = 0
            for net_name in self.node_to_net_dict[node_name]:
                if net_name in effective_net:
                    if net_name in visited_net:
                        cnt += 1
                    else:
                        visited_net.add(net_name)
            visited_neighbour = cnt
            visited_neighbours.append(visited_neighbour)
    
    def get_graph_fea(self, offset = 0):
        graph = {}
        for i in range(min(256, self.node_cnt)):
            graph[i+offset] = set()
        for i, net_name in enumerate(self.net_info):
            node_list = []
            for node_name in self.net_info[net_name]['nodes']:
                if node_name in self.node_name_to_id and self.node_name_to_id[node_name] < 256:
                    node_list.append(self.node_name_to_id[node_name])
            for i, j in combinations(node_list, 2):
                graph[i+offset].add(j+offset)
                graph[j+offset].add(i+offset)
        features = np.zeros((min(256, self.node_cnt), 4))
        for i in range(min(256, self.node_cnt)):
            node_name = self.node_id_to_name[i]
            features[i][0] = self.node_info[node_name]['x'] / self.max_height
            features[i][1] = self.node_info[node_name]['y'] / self.max_height
            features[i][2] = self.node_info[node_name]['x'] * self.node_info[node_name]['y'] / (self.max_height ** 2)
            features[i][3] = i / 256.0
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        print("features shape", features.shape)
        print("adj shape", adj.shape)
        
        return adj, features, graph
    
    def get_circuit_fea(self, is_graph = True):
        if is_graph:
            graph_result = pickle.load(open("circuit_g_token-example.pkl", "rb"))
            z_emb_avg = graph_result[self.benchmark].detach().numpy()
            print("z_emb_avg shape", z_emb_avg.shape)
            circuit_fea = np.zeros((256*3,))
            circuit_fea[: z_emb_avg.shape[0]] = z_emb_avg
            return circuit_fea
        areas = []
        neighbors = []
        visited_neighbours =[]
        heights = []
        widths = []

        effective_net = set()
        for net_name in self.net_info:
            cnt = 0
            for node_name in self.net_info[net_name]['nodes']:
                if node_name not in self.node_info:
                    continue
                if self.node_name_to_id[node_name] < 256:
                    cnt += 1
            if cnt >=2:
                effective_net.add(net_name)
        for i in range(min(256, len(self.node_info))):
            node_name = self.node_id_to_name[i]
            area = self.node_info[node_name]['x'] * self.node_info[node_name]['y']
            area = area / (self.max_height * self.max_height)
            areas.append(area)
            width = self.node_info[node_name]['x'] / self.max_height
            widths.append(width)
            height = self.node_info[node_name]['y'] / self.max_height
            heights.append(height)
            cnt = 0
            for net_name in self.node_to_net_dict[node_name]:
                if net_name in effective_net:
                    cnt += 1
            neighbor = cnt
            neighbors.append(neighbor)
            
        visited_net = set()
        for i in range(min(256, len(self.node_info))):
            node_name = self.node_id_to_name[i]
            cnt = 0
            for net_name in self.node_to_net_dict[node_name]:
                if net_name in effective_net:
                    if net_name in visited_net:
                        cnt += 1
                    else:
                        visited_net.add(net_name)
            visited_neighbour = cnt
            visited_neighbours.append(visited_neighbour)
        
        if len(visited_neighbours) < 256:
            areas.extend([0]*(256-len(areas)))
            widths.extend([0]*(256-len(widths)))
            heights.extend([0]*(256-len(heights)))
            neighbors.extend([0]* (256-len(neighbors)))
            visited_neighbours.extend([0]* (256-len(visited_neighbours)))

        areas = np.array(areas)
        areas = areas / areas.max()
        widths = np.array(widths)
        width = widths / widths.max()
        heights = np.array(heights)
        heights = heights / heights.max()
        neighbors = np.array(neighbors)
        neighbors = neighbors / neighbors.max()
        visited_neighbours = np.array(visited_neighbours)
        visited_neighbours = visited_neighbours / visited_neighbours.max()
        circuit_fea = np.concatenate((areas, neighbors, visited_neighbours))
        print("circuit_fea shape", circuit_fea.shape)
        return circuit_fea


if __name__ == "__main__":
    placedb = PlaceDB("adaptec1")
    placedb2 = PlaceDB("adaptec1", is_graph = True)
    placedb.debug_str()



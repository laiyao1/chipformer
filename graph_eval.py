import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import networkx as nx
import numpy as np
import os
import time
import pickle
# from input_data import load_data
from preprocessing import *
import graph_args as args
import graph_model
from mingpt.place_db import PlaceDB
# Train on CPU (hide GPU) due to memory constraints


benchmark_list = ['adaptec1']
# benchmark_list = ['adaptec1', 'adaptec2', 'adaptec3', 'adaptec4',
# 'bigblue1', 'bigblue2', 'bigblue3', 'bigblue4', 
# 'ibm01', 'ibm02', 'ibm03', 'ibm04']

model_path = "save_graph_models/example_graph_model.pkl"
state_dict = torch.load(model_path)

result = {}

for benchmark in benchmark_list:
    place_db = PlaceDB(benchmark)
    features = place_db.features
    adj = place_db.adj
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_norm = preprocess_graph(adj)
    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                torch.FloatTensor(adj_norm[1]), 
                                torch.Size(adj_norm[2]))
    features = torch.tensor(features, dtype=torch.float32)
    
    model = getattr(graph_model, args.model)(adj_norm)
    model.load_state_dict(state_dict, strict = True)
    model.eval()

    z_emb = model.encode(features)

    print("z_emb", z_emb)
    print("z_emb shape", z_emb.shape)
    z_emb_avg = torch.mean(z_emb, axis=0)
    result[benchmark] = z_emb_avg

strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
pickle.dump(result, open("circuit_g_token-{}.pkl".format(strftime),'wb'))

import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import networkx as nx
import numpy as np
import os
import time

# from input_data import load_data
from preprocessing import *
import graph_args as args
import graph_model
from mingpt.place_db import PlaceDB
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

offset = 0
graph = {}
for i, benchmark in enumerate(["adaptec1"]):
    placedb = PlaceDB(benchmark, offset = offset)
    
    tmp_features, tmp_graph = placedb.features, placedb.graph
    offset += placedb.features.shape[0]
    print("tmp_graph", tmp_graph)
    graph = {**graph, **tmp_graph}
    if i==0:
        features = tmp_features
    else:
        features = np.concatenate((features, tmp_features), axis=0)

adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
print("adj shape", adj.shape)
print("features shape", features.shape)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)

num_nodes = adj.shape[0]

# Create Model
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)



adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                            torch.FloatTensor(adj_norm[1]), 
                            torch.Size(adj_norm[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                            torch.FloatTensor(adj_label[1]), 
                            torch.Size(adj_label[2]))
features = torch.tensor(features, dtype=torch.float32)

weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0)) 
weight_tensor[weight_mask] = pos_weight

# init model and optimizer
model = getattr(graph_model, args.model)(adj_norm)
optimizer = Adam(model.parameters(), lr=args.learning_rate)


def get_scores(edges_pos, edges_neg, adj_rec):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:

        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    preds_all_zero_one = np.where(preds_all < 0.65, 0, 1)
    
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    test_acc = (preds_all_zero_one == labels_all).sum() /   labels_all.shape[0]
    print("labels_all", labels_all)
    print("preds_all_zero_one", preds_all_zero_one)
    print("preds_all", preds_all)
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score, test_acc

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.65).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

# train model
for epoch in range(args.num_epoch):
    t = time.time()

    A_pred = model(features)
    optimizer.zero_grad()
    loss = log_lik = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
    if args.model == 'VGAE':
        kl_divergence = 0.5/ A_pred.size(0) * (1 + 2*model.logstd - \
            model.mean**2 - torch.exp(model.logstd)**2).sum(1).mean()
        loss -= kl_divergence

    loss.backward()
    optimizer.step()

    train_acc = get_acc(A_pred,adj_label)
    val_roc, val_ap, val_acc = get_scores(val_edges, val_edges_false, A_pred)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
          "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
          "val_ap=", "{:.5f}".format(val_ap),
          "val_acc=", "{:.5f}".format(val_acc),
          "time=", "{:.5f}".format(time.time() - t))


test_roc, test_ap, test_acc = get_scores(test_edges, test_edges_false, A_pred)
print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap),
      "test_acc=", "{:.5f}".format(test_acc))

print("saving model...")
model.eval()
strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

if not os.path.exists("save_graph_models"):
    os.mkdir("save_graph_models")
torch.save(model.state_dict(), "save_graph_models/{}-{:.4f}-{:.4f}.pkl".format(strftime, test_roc, test_ap))
from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
import torch.nn.functional as F
import torch.optim as optim
import assment_result
import os
from collections import Counter
from utils import load_data, accuracy, binary_accuracy, get_vttdata, get_afldata, get_degree_feature_list
from models import DGI

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--origin', action='store_true', default=True,
                    help='Keep the original implementation as the paper.')
parser.add_argument('--test_only', action="store_true", default=False,
                    help='Test on existing model')
parser.add_argument('--repeat', type=int, default=1,
                    help='number of experiments')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Show training process')
parser.add_argument('--split', type=str, default='random',
                    help='Data split method')
parser.add_argument('--rho', type=float, default=0.1,
                    help='Adj matrix corruption rate')
parser.add_argument('--corruption', type=str, default='node_shuffle',
                    help='Corruption method')

def get_data(timestep, node_num, edges_path):
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("prepared for loading data!")
    adj, features, graph = get_afldata(timestep, node_num, edges_path)
    print("Load have done!")
    idx_train, idx_val, idx_test = get_vttdata(node_num)

    if args.cuda:
        features = features
        adj = adj
        idx_train = idx_train
        idx_val = idx_val
        idx_test = idx_test
    return adj, features, idx_train, idx_val, idx_test, args, graph

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def train(num_epoch, time_step,last_embedding,patience=30, verbose=False):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_epoch = -1
    for epoch in range(num_epoch):
        t = time.time()
        optimizer.zero_grad()
        if time_step >= 1:
            getloss = True
            outputs, labels, smooth_loss = model(features, adj, last_embedding, getloss)
        else:
            outputs, labels, smooth_loss = model(features, adj)
        loss_train = F.binary_cross_entropy_with_logits(outputs,labels)
        alph = 0.01
        curr = 1 - alph
        loss_train = curr * loss_train - alph * smooth_loss
        acc_train = binary_accuracy(outputs, labels)
        loss_train.backward()
        optimizer.step()
        loss = loss_train.item()
        accuracy = acc_train.item()
        if verbose:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss),
                  'acc_train: {:.4f}'.format(accuracy),
                  'time: {:.4f}s'.format(time.time() - t))

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
        if epoch == best_epoch + patience:
            break

def test(verbose=False):
    with torch.set_grad_enabled(False):
        model.eval()
        last_embedding = 0
        getloss = False
        if isQloss:
            outputs, weight, bestQ_H = model(features, adj, last_embedding, getloss)
            outputs_numpy = bestQ_H.data.cpu().numpy()
        else:
            outputs, weight= model(features, adj, last_embedding, getloss) ##size is:torch.Size([2708, 512]),
            outputs_numpy = outputs.data.cpu().numpy()
        weight_numpy = weight.data.cpu().numpy()
    return outputs, weight, outputs_numpy


if __name__ == "__main__":

    time_weight_list=[0]
    embedding_list = [0]
    NMI_list = []
    accuracy_list = []
    ARI_list = []
    Q_list = []

    dataset = 'collegeMsg'
    node_num = 1899

    isone_hot = False
    islabel = True
    isevaexitnodes = False
    method = "DDGI_0.6_2"
    isQloss = False
    base_data_path = "data/"
    edges_base_path = "/edges"
    label_base_path = "/labels"
    edges_data_path = base_data_path + dataset + edges_base_path
    file_num = len(os.listdir(edges_data_path))
    if isone_hot:
        x_list, max_degree = get_degree_feature_list(edges_data_path,node_num)  #这个可以进行one-hot编码
    print("file_num:{}".format(file_num))
    for t in range(file_num):
        print("开始执行第{}个时间步".format(t+1))
        time_edges_path = edges_data_path + "\\edges_t" + str(t+1) + ".txt" # We can use this edge path
        adj, features, idx_train, idx_val, idx_test, args, graph = get_data(t, node_num, time_edges_path)
        print("Get data done!")
        if islabel:
            label_data_path = base_data_path + dataset + label_base_path + "\\label_" + str(t+1) +".txt"
            original_cluster = np.loadtxt(label_data_path, dtype=int)
        if isone_hot:
            print("the feature is generated by one-hot code!")
            features = x_list[t]
            features = sp.coo_matrix(features, shape=(node_num, max_degree))
            features = normalize(features)
            features = torch.FloatTensor(np.array(features.todense()))
        for i in range(args.repeat):
            # model
            model = DGI(num_feat=features.shape[1],
                        num_hid=args.hidden,
                        time_step=t,
                        graph = graph,
                        Qloss = isQloss,
                        time_weight=time_weight_list[-1],
                        dropout=args.dropout,
                        rho=args.rho,
                        corruption=args.corruption)
            print("----- %d / %d runs -----" % (i+1, args.repeat))
            t_total = time.time()
            if args.test_only:
                model = torch.load("model")
            else:
                train(args.epochs, t,embedding_list[-1], verbose=args.verbose)
                print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            outputs, weight, outputs_numpy = test(verbose=args.verbose)
            np.savetxt("embeddings/" + dataset +"/" + method +"/embedding_t" + str(t) + ".txt", outputs_numpy, fmt="%f")
            np.savetxt("embeddings/" + dataset + "/" + method + "/embedding_t" + str(t) + ".csv", outputs_numpy,
                       fmt="%f", delimiter=",")
            embedding_path = "embeddings/" + dataset +"/" + method +"/embedding_t" + str(t) + ".txt"
            embedding_path2 = "embeddings/" + dataset + "/" + method + "/embedding_t" + str(t) + ".csv"
            time_weight_list[-1] = weight
            embedding_list[-1] = outputs
            if islabel:
                if isevaexitnodes:
                    exitNode_list = sorted(list(graph.nodes()))
                    exit_labels = []
                    exit_embeddings= []
                    for i, el in enumerate(original_cluster):
                        if (i in exitNode_list):
                            exit_labels.append(el)
                    for j, en in enumerate(outputs_numpy):
                        if (j in exitNode_list):
                            exit_embeddings.append(en)
                    exit_embeddings = np.mat(exit_embeddings)
                    k = len(Counter(exit_labels))
                    print("节点数量为：{0}, embedding 数量为：{1}".format(len(exit_labels),len(exit_embeddings)))
                    NMI, F1score, ARI = assment_result.assement_result(exit_labels, exit_embeddings, k, t)
                    adj_path = "../adj//" + dataset + "//" + method + "/adj_t" + str(t) + ".csv"
                    Q = assment_result.assement_Q(adj_path, time_edges_path, node_num, embedding_path2, k, 1)
                    G = nx.read_adjlist(time_edges_path)
                    nx.draw(G)
                    print("NMI 值为：{}".format(NMI))
                    print("ARI 值为：{}".format(ARI))
                    print("Q 值为： {}".format(Q))
                    NMI_list.append(NMI)
                    ARI_list.append(ARI)
                    Q_list.append(Q)
                else:
                    print("同一个时间步评测所有节点")
                    k = len(Counter(original_cluster))
                    adj_path = "adj/" + dataset +"/" + method +"/adj_t" + str(t) + ".csv"
                    NMI, F1score, ARI = assment_result.assement_result(original_cluster, outputs_numpy, k, t)
                    Q = assment_result.assement_Q(adj_path, time_edges_path, node_num, embedding_path2, k, 1)
                    print("NMI 值为：{}".format(NMI))
                    print("ARI 值为：{}".format(ARI))
                    print("Q 值为： {}".format(Q))
                    NMI_list.append(NMI)
                    ARI_list.append(ARI)
                    Q_list.append(Q)

    if islabel:
        print("NMI 的平均值为：{0}, 值为：{1}".format(np.mean(NMI_list), NMI_list))
        print("ARI 的平均值为：{0}, 值为：{1}".format(np.mean(ARI_list), ARI_list))
        print("Q 的平均值为：{0}, 值为：{1}".format(np.mean(Q_list), Q_list))
    else:
        print("该数据集无真实划分，请继续进无标签评估实验！")
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.cluster import KMeans

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, origin=False):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.origin = origin

    def forward(self, x, adj):
        if self.origin:
            x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class DGI(nn.Module):
    def __init__(self, num_feat, num_hid, time_step, graph, Qloss, time_weight, dropout, rho=0.1, readout="average", corruption="node_shuffle"):
        super(DGI, self).__init__()
        self.time_time =time_step
        self.time_weight_weight = time_weight
        self.Qloss = Qloss
        self.graph = graph
        self.gc = GraphConvolution(num_feat, num_hid, time_step, time_weight)
        self.fc = nn.Linear(num_hid, num_hid, bias=False)
        self.dropout = dropout
        self.prelu = nn.PReLU()
        self.rho = rho
        self.bestQ = 0
        self.bestQ_H = 0
        self.readout = getattr(self, "_%s" % readout)
        self.corruption = getattr(self, "_%s" % corruption)

    def forward(self, X, A, last_embedding=0, getloss=False):
        x = F.dropout(X, self.dropout, training=self.training)
        HHH, weight = self.gc(x, A)
        H = self.prelu(HHH)
        if self.Qloss:
            k = 20
            node_num = 400
            B = self.modularity_generator(self.graph)
            print("B type is:{0}, shape：{1}，value is:{1}".format(type(B), B.shape, B))
            exist_H = self.get_exitembeddings(self.graph, H)
            print("exist_H type is:{0}, shape：{1}，value is:{1}".format(type(exist_H), exist_H.shape, exist_H))
            indicate_H = self.get_idcatematrix(node_num, k, H)
            Q_m = torch.mm(indicate_H.t(),torch.FloatTensor(B))
            Q_loss = torch.trace(torch.mm(Q_m,indicate_H))
            if self.bestQ < Q_loss:
                self.bestQ = Q_loss
                self.bestQ_H = H
            print("Q_loss: is:{0}, best Q is:{1}".format(Q_loss, self.bestQ))
        smooth_loss = 0
        if getloss:
            smooth_loss = torch.cosine_similarity(H,last_embedding).sum()

        if not self.training:
            if self.Qloss:
                return H, weight, self.bestQ_H
            else:
                return H, weight

        neg_X, neg_A = self.corruption(X, A)
        x = F.dropout(neg_X, self.dropout, training=self.training)
        neg_HHH, neg_weight = self.gc(x, neg_A)
        neg_H = self.prelu(neg_HHH)
        s = self.readout(H)
        x = self.fc(s)
        x = torch.mv(torch.cat((H, neg_H)), x)
        labels = torch.cat((torch.ones(X.size(0)), torch.zeros(neg_X.size(0))))
        if self.Qloss:
            return x, labels, smooth_loss, Q_loss, self.bestQ_H
        else:
            return x, labels, smooth_loss

    def _average(self, features):
        x = features.mean(0)
        return F.sigmoid(x)

    def _node_shuffle(self, X, A):
        perm = torch.randperm(X.size(0))
        neg_X = X[perm]
        return neg_X, A

    def euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()
        return dist.sum()

    def _adj_corrupt(self, X, A):
        rho = self.rho
        [n, m] = A.shape
        neg_A = A.clone()
        p = np.random.rand(n, m)
        d_A = np.zeros((n, m))
        d_A[p < rho] = 1
        neg_A = np.logical_xor(neg_A.to_dense().data.cpu().numpy(), d_A)
        idx = np.nonzero(neg_A)
        d_A = torch.sparse.FloatTensor(torch.LongTensor(np.array(idx)), torch.FloatTensor(np.ones(len(idx[0]))) , \
                                       torch.Size([n, m]))
        return X, d_A

    def modularity_generator(self,G):
        """
        Function to generate a modularity matrix.
        :param G: Graph object.
        :return laps: Modularity matrix.
        """
        print("Modularity calculation.\n")
        degrees = nx.degree(G)
        e_count = len(nx.edges(G))
        modu = np.array(
            [[float(degrees[node_1] * degrees[node_2]) / (2 * e_count) for node_1 in nx.nodes(G)] for node_2 in
             tqdm(nx.nodes(G))], dtype=np.float64)
        return modu
    def get_idcatematrix(self,node_num,k,embeddings):
        """
        :param node_num:
        :param k:
        :param embeddings:
        :return: 返回一个tensor张量矩阵
        """
        clf = KMeans(k)
        y_pred = clf.fit_predict(embeddings.detach().numpy())
        print("y_pred type is:{0}, shape is:{1}".format(type(y_pred), y_pred.shape))
        y_pred = torch.LongTensor(y_pred)
        ones = torch.sparse.torch.eye(k)
        y_one_hot = ones.index_select(0,y_pred)
        print("y_one_hot:{}".format(y_one_hot.size))
        return y_one_hot

    def get_exitembeddings(self, graph, embedding):
        exit_embeddings = []
        exitNode_list = sorted(list(graph.nodes()))
        for j, en in enumerate(embedding.detach().numpy()):
            if (j in exitNode_list):
                exit_embeddings.append(en)
        exit_embeddings = np.mat(exit_embeddings)
        return exit_embeddings

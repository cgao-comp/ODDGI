import math
import torch
import numpy as np
import utils as u
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, time_step, time_weight, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.time_step = time_step
        self.time_weight = time_weight
        self.out_features = out_features
        cell_args = u.Namespace({})
        cell_args.rows = in_features
        cell_args.cols = out_features
        self.evolve_weights = mat_GRU_cell(cell_args)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        if self.time_step>=1:
            # self.GCN_weights = self.evolve_weights(self.weight)
            self.GCN_weights = self.evolve_weights(self.time_weight)
            # print("xxxxxxxxxxxx当前时间步为：{},权重为：{}".format(time_step,self.GCN_weights))
            self.dgi_weight = self.GCN_weights
            self.dgi_weight = Parameter(self.dgi_weight)  ##需要将其参数化
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        if self.time_step >= 1000:                           ####如果要跑原始DGI，请把这个1改成1000
            # self.first_weight = self.evolve_weights(self.time_weight) #######跑聚类，这个注释掉
            # print("yyyyyyyyyy当前时间步为：{},权重为：{}".format(self.time_step, self.dgi_weight))
            self.first_weight = self.dgi_weight           #######跑聚类，这个打开
            support = torch.mm(input, self.first_weight)
            output = torch.spmm(adj, support)
        else:
            self.first_weight = self.weight
            support = torch.mm(input, self.first_weight)
            output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias, self.first_weight
        else:
            return output, self.first_weight
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class mat_GRU_cell(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                  args.cols,
                                  torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())

        self.choose_topk = TopK(feats=args.rows,
                                k=args.cols)

    def forward(self, prev_Q):
        prev_Q = prev_Q
        z_topk = prev_Q
        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)
        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)
        new_Q = (1 - update) * prev_Q + update * h_cap
        return new_Q


class mat_GRU_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.W)
        self.U = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.U)
        self.bias = Parameter(torch.zeros(rows, cols))

    def reset_param(self, t):
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        x = x
        hidden = hidden
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs, mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
                isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()
        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        return out.t()

import os
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import time
from tqdm import tqdm
import pandas as pd


def get_degree_feature_list(edges_list_path, node_num, init='one-hot'):
    x_list = []
    max_degree = 0
    adj_list = []
    degree_list = []
    ret_degree_list = []
    edges_dir_list = sorted(os.listdir(edges_list_path))
    for i in range(len(edges_dir_list)):
        edges_path = os.path.join(edges_list_path, edges_dir_list[i])
        adj_lilmatrix = get_adj_lilmatrix(edges_path,node_num)
        # node_num = len(adj)
        adj = sp.coo_matrix(adj_lilmatrix)
        adj_list.append(adj)
        degrees = adj.sum(axis=1).astype(np.int)
        max_degree = max(max_degree, degrees.max())
        degree_list.append(degrees)
        ret_degree_list.append(torch.FloatTensor(degrees).cuda() if torch.cuda.is_available() else degrees)
    for i, degrees in enumerate(degree_list):
        # other structural feature initialization techiniques can also be tried to improve performance
        if init == 'gaussian':
            fea_list = []
            for degree in degrees:
                fea_list.append(np.random.normal(degree, 0.0001, max_degree + 1))
            fea_arr = np.array(fea_list)
            fea_tensor = torch.FloatTensor(fea_arr)
            x_list.append(fea_tensor.cuda() if torch.cuda.is_available() else fea_tensor)
            return x_list, fea_arr.shape[1], ret_degree_list
        elif init == 'combine':
            fea_list = []
            for degree in degrees:
                fea_list.append(np.random.normal(degree, 0.0001, max_degree + 1))
            fea_arr = np.array(fea_list)
            ###################
            # here combine the adjacent matrix feature could improve strcutral role classification performance,
            # but if the graph is large, adj feature will be memory consuming
            fea_arr = np.hstack((fea_arr, adj_list[i].toarray()))
            ###################
            fea_tensor = torch.FloatTensor(fea_arr)
            x_list.append(fea_tensor.cuda() if torch.cuda.is_available() else fea_tensor)
            return x_list, fea_arr.shape[1], ret_degree_list
        elif init == 'one-hot':  # one-hot degree feature
            degrees = np.asarray(degrees,dtype=int).flatten()
            # print("degereeeee:{}".format(degrees))
            one_hot_feature = np.eye(max_degree + 1)[degrees]
            x_list.append(one_hot_feature.cuda() if torch.cuda.is_available() else one_hot_feature)


        else:
            raise AttributeError('Unsupported feature initialization type!')
    return x_list, max_degree + 1


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    # print("labels_onehot".format(labels_onehot))
    return labels_onehot

def check_and_make_path(to_make):
    if to_make == '':
        return
    if not os.path.exists(to_make):
        os.makedirs(to_make)


def load_data(dataset="cora", path="../data/", origin=False, split="random"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    raw_data = np.genfromtxt(os.path.join(path, dataset, "%s.content" % dataset),
                            dtype=np.str)
    raw_idxs, raw_features, raw_labels = raw_data[:, 0], raw_data[:, 1:-1], raw_data[:, -1]
    features = sp.csr_matrix(raw_features, dtype=np.float32)
    labels = encode_onehot(raw_labels)

    # build graph
    edges_unordered = np.genfromtxt(os.path.join(path, dataset, "%s.cites" % dataset),
                                    dtype=np.str)
    idx_map = {j: i for i, j in enumerate(raw_idxs)}
    idx_edges = []
    invalid = 0
    for u, v in edges_unordered:
        if u not in idx_map or v not in idx_map:
            invalid += 1
            continue
        idx_edges.append([idx_map[u], idx_map[v]])
    print("%d / %d invalid edges" % (invalid, len(edges_unordered)))
    edges = np.array(idx_edges, dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    if origin:
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    else:
        # this normalization method introduces gains
        adj = normalize(adj + sp.eye(adj.shape[0]))

    if split == "fix":
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
    elif split == "equal":
        idx_train = []
        for label in set(raw_labels):
            idxes = np.where(raw_labels == label)[0]
            idx_train += list(np.random.choice(idxes, 20, replace=False))
        rest = list(set(range(len(raw_labels))) - set(idx_train))
        idxes = np.random.choice(rest, 300 + 1000, replace=False)
        idx_val, idx_test = idxes[:300], idxes[300:]
    elif split == "random":
        all = range(len(raw_labels))
        # print("the type law_labels is {0}, and the value is {1} ".format(type(raw_labels), raw_labels))
        print("the type all is {0}, and the value is {1} ".format(type(all), all))
        idxes = np.random.choice(all, 140 + 300 + 1000)
        idx_train, idx_val, idx_test = idxes[:140], idxes[140:-1000], idxes[-1000:]
    else:
        raise ValueError("Unknown split type `%s`" % split)

    features = torch.FloatTensor(np.array(features.todense()))
    # print("np.where(labels)[1] 的值为：{0}, 类型为: {1}".format(np.where(labels)[1], type(np.where(labels)[1])))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test
    # return adj, features_adj, labels, idx_train, idx_val, idx_test

def get_vttdata(node_num):
    all=range(node_num)
    idxes = np.random.choice(all, 20 + 44 + 147)
    idx_train, idx_val, idx_test = idxes[:20], idxes[20:-147], idxes[-147:]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return idx_train, idx_val, idx_test


def get_afldata(timestep, node_num, edges_path):
    adj = get_adj_lilmatrix(edges_path, node_num)
    print("Got adj done!")
    features_adj = sp.coo_matrix(adj,dtype=float) ##如果还是one-hot就用这个
    print("Got features_adj done!")
    similairity_matirx, graph = getSimilariy(adj,node_num,edges_path)
    t1 = time.time()
    # similairity_matirx, graph = getSimilariy_modified(adj,node_num,edges_path)
    print("Got similairity_matirx done! Time cost is:{}".format(time.time()-t1))
    #######################################################################################
    ################################### 参数分析 ############################################ γ

    adj = adj + 0 * similairity_matirx # 0.2
    adj = sp.coo_matrix(adj)
    # print(features_adj)
    # features_adj = adj
    features_adj = normalize(features_adj)
    features_adj = torch.FloatTensor(np.array(features_adj.todense())) ##有时候需要注释掉，比如采用one-hot进行特征编码的时候
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # features = normalize(features)  ##本来是用来加载feature的，现在不用了，现在直接用adj当feature
    # features = torch.FloatTensor(np.array(features))   ##本来是用来加载feature的，现在不用了，现在直接用adj当feature
    # return adj, features, labels
    return adj, features_adj, graph

def getGraph(matrix,node_num,edges_path):
    """
    Function to convert a matrix to a networkx graph object.
    :param matrix: the matrix which to convert.
    :return graph: NetworkX grapg object.
    """
    # matrix = np.asarray(matrix,dtype=float)
    # matrix = sp.lil_matrix(matrix)
    G = nx.Graph()
    with open(edges_path, 'r') as fp:
        content_list = fp.readlines()
        # 0 means not ignore header
        for line in content_list[0:]:
            line_list = line.split(" ")
            from_id, to_id = line_list[0], line_list[1]
            # remove self-loop data
            if from_id == to_id:
                continue

            G.add_edge(int(from_id),int(to_id))

    # for i in range(node_num):
    #     for j in range(node_num):
    #         if matrix[i][j] == 1:
    #             G.add_edge(i,j)
    return G

def get_adj_lilmatrix(edge_path, node_num):
    A = sp.lil_matrix((node_num, node_num), dtype=int)
    # a = np.loadtxt("F:\postgradate_file\demo\Deep-Graph-Infomax(another)\data\enronMail\edges\edges_t1.txt",dtype=float)
    with open(edge_path, 'r') as fp:
        content_list = fp.readlines()
        # 0 means not ignore header
        for line in content_list[0:]:
            line_list = line.split(" ")
            from_id, to_id = line_list[0], line_list[1]
            # remove self-loop data
            if from_id == to_id:
                continue
            A[int(from_id), int(to_id)] = 1   ##这里要特别注意id是不是从0开始计数。贪方便的话，如果是1开始计数，可以废掉第0为，node_num 要加1
            A[int(to_id), int(from_id)] = 1

    # print("the type of A is:{}, the size of A is: {}".format(type(A), A.shape))
    # print(A)
    # B = sp.coo_matrix(A)
    # print("the type of B is:{}, the size of B is: {}".format(type(B), B.shape))
    return A

def getSimilariy(OneZeromatrix, node_num, edges_path):
    # similar_matrix = np.zeros((node_num,node_num),dtype=float)
    similar_matrix = sp.lil_matrix((node_num, node_num), dtype=float)
    graph = getGraph(OneZeromatrix, node_num,edges_path)
    print("Got graph done!")
    node_list = list(graph.nodes())
    # for i in range(node_num):
    #     if i not in list(graph.node()):
    #         graph.add_edge(i, i)
    for i, node in enumerate(node_list):
        neibor_i_list = list(graph.neighbors(node))
        first_neighbor = neibor_i_list
        for k, second_nighbor in enumerate(first_neighbor):
            second_list = list(graph.neighbors(second_nighbor))
            neibor_i_list = list(set(neibor_i_list).union(set(second_list)))
        neibor_i_num = len(first_neighbor) ## 节点i的邻居数量
        for j, node_j in enumerate(neibor_i_list):
            neibor_j_list = list(graph.neighbors(node_j))
            neibor_j_num = len(neibor_j_list)   ## 节点j的邻居数量
            commonNeighbor_list = [x for x in first_neighbor if x in neibor_j_list] ##公共邻居节点的列表。
            commonNeighbor_num = len(commonNeighbor_list)##公共邻居节点的数量集合。
            similar_matrix[node, node_j] = (2*commonNeighbor_num)/(neibor_j_num + neibor_i_num)
    return similar_matrix, graph


def getSimilariy_modified(OneZeromatrix,node_num,edges_path):
    # node_num = len(OneZeromatrix)
    # similar_matrix = np.zeros((node_num,node_num),dtype=float)
    similar_matrix = sp.lil_matrix((node_num,node_num),dtype=float)
    graph = getGraph(OneZeromatrix, node_num,edges_path)
    edges_list = list(graph.edges())
    node_list = list(graph.nodes())
    for i, node in enumerate(node_list):
        # print("计数第i个节点的dice相似度：{}".format(i))
        neibor_i_list = list(graph.neighbors(node))
        first_neighbor = neibor_i_list
        # print("the length of first norbor of i is {}".format(len(first_neighbor)))
        for k, second_nighbor in enumerate(first_neighbor):
            second_list = list(graph.neighbors(second_nighbor))
            neibor_i_list = list(set(neibor_i_list).union(set(second_list))) ##取一二阶邻居的并集
        # print("the length of norbor of i is {}".format(len(neibor_i_list)))
        neibor_i_num = len(first_neighbor)  ## 节点i的邻居数量
        for j, node_j in enumerate(neibor_i_list):
            neibor_j_list = list(graph.neighbors(node_j))
            neibor_j_num = len(neibor_j_list)  ## 节点j的邻居数量
            commonNeighbor_list = [x for x in first_neighbor if x in neibor_j_list]  ##公共邻居节点的列表。
            # commonNeighbor_list = list(set(first_neighbor).intersection(neibor_j_list))
            commonNeighbor_num = len(commonNeighbor_list)  ##公共邻居节点的数量集合。
            neibor_i_num_x = neibor_i_num
            if (i,j) in edges_list:
                commonNeighbor_num = commonNeighbor_num + 2
                neibor_j_num = neibor_j_num + 1
                neibor_i_num_x = neibor_i_num + 1
            # print("similiar_type:{0}, shape:{1}".format(type(similar_matrix),similar_matrix.shape))
            # print("i:{},j:{}".format(i,j))
            similar_matrix[node, node_j] = (2*commonNeighbor_num)/(neibor_j_num + neibor_i_num_x)
    return similar_matrix, graph




def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def accuracy(outputs, labels):
    preds = outputs.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def binary_accuracy(outputs, labels):
    preds = outputs.gt(0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)


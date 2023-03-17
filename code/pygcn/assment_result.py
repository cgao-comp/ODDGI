from sklearn.cluster import KMeans
import csv
import numpy as np
from sklearn import metrics
import warnings
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
import networkx as nx
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def getQ2(G, labels, matirx_path, n_clusters):
    A = np.loadtxt(matirx_path, dtype=int)
    n = len(G.nodes)
    e = len(G.edges)

    S = np.zeros((n, n_clusters))
    for index, value in enumerate(G.nodes):
        S[index][labels[index]] = 1

    B = np.zeros((n, n))
    for i, v_i in enumerate(G.nodes):
        for j, v_j in enumerate(G.nodes):
            B[i][j] = A[i][j] - G.degree(v_i) * G.degree(v_j) / (2 * e)

    Q = 1 / (2 * e) * np.trace(S.T @ B @ S)
    return Q


def getQ(G, labels, matirx_path):
    A = matirx_path
    node_num = len(G.nodes)
    edges_num = len(G.edges)
    sum = 0
    for i, d_i in enumerate(G.degree()):
        for j, d_j in enumerate(G.degree()):
            if labels[i] == labels[j]:
                sum += A[i][j] - d_i[1] * d_j[1] / (2 * edges_num)
    Q = sum / (2 * edges_num)
    return Q


def assement_Q(adjpath, input_path, node_n, embedding_path, k, flage):
    adj = np.zeros((node_n, node_n))
    f = open(input_path)
    csvfile = open(adjpath, 'w', newline = "")
    writer = csv.writer(csvfile)
    line = f.readline()
    while line:
        node1 = int(line.split()[0])
        node2 = int(line.split()[1])
        adj[node1][node2] = 1
        adj[node2][node1] = 1
        data = (node1, node2)
        writer.writerow([node1, node2])
        line = f.readline()
    csvfile.close()
    graph = graph_reader(adjpath)
    if flage == "openNE":
        a = np.loadtxt(embedding_path)
        x = a
        b = a[np.lexsort(a[:, ::-1].T)]
        x = b[..., 1:]
    else:
        x = pd.read_csv(embedding_path, header=None)
    clf = KMeans(k)
    y_pred = clf.fit_predict(x)
    Q = getQ(graph, y_pred, adj)
    return Q


def graph_reader(input_path):
    """
    Function to read a csv edge list and transform it to a networkx graph object.
    :param input_path: Path to the edge list csv.
    :return graph: NetworkX grapg object.
    """
    edges = pd.read_csv(input_path)
    graph = nx.from_edgelist(edges.values.tolist())
    return graph


def eval_classification(embeddings, labes, train_percent=0.8):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labes, train_size=train_percent,
                                                        test_size=1 - train_percent, random_state=666)

    clf = LinearSVC()
    clf.fit(X_train, y_train)
    res = clf.predict(X_test)

    accuracy = accuracy_score(y_test, res)
    macro = f1_score(y_test, res, average='macro')
    micro = f1_score(y_test, res, average='micro')
    return micro, macro, accuracy


def assement_result(labels, embeddings, k, t):
    print("聚类数为：{}".format(k))
    origin_cluster = labels
    a = 0
    sum = 0
    sumF1score = 0
    sumARI = 0
    sumAccuracy = 0
    while a < 10:
        clf = KMeans(k)
        y_pred = clf.fit_predict(embeddings)
        """"
        if a == 5 and t == 9:
            print(y_pred)
            workbook = xlwt.Workbook()
            worksheet = workbook.add_sheet('test')
            i = 0
            for ele in y_pred:
                worksheet.write(i, 0, str(ele))
                i = i + 1
            workbook.save('H:\\2020Summer\\动态社团挖掘\\论文撰写\\动态网络可视化\\results9.xls')
        """

        c = y_pred.T
        epriment_cluster = c

        NMI = metrics.normalized_mutual_info_score(origin_cluster, epriment_cluster)
        if a == 1:
            name = "time"+str(t)+".txt"
            np.savetxt(name, epriment_cluster,fmt="%d")
        F1_score = f1_score(origin_cluster, epriment_cluster, average='micro')
        ARI = adjusted_rand_score(origin_cluster, epriment_cluster)
        accuracy = accuracy_score(origin_cluster, epriment_cluster)
        sum = sum + NMI
        sumF1score = sumF1score + F1_score
        sumARI = sumARI + ARI
        sumAccuracy = accuracy + sumAccuracy
        a = a + 1
    average_NMI = sum / 10
    average_F1score = sumF1score / 10
    average_ARI = sumARI / 10
    average_Accuracy = sumAccuracy / 10
    print(type(embeddings))
    return average_NMI, average_F1score, average_ARI

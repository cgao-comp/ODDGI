import scipy.sparse as sp

edge_path = "F:\postgradate_file\demo\Deep-Graph-Infomax(another)\data\enronMail\edges\edges_t1.txt"
A = sp.lil_matrix((151,151),dtype=int)
with open(edge_path, 'r') as fp:
    content_list = fp.readlines()
    for line in content_list[0:]:
        line_list = line.split(" ")
        from_id, to_id = line_list[0], line_list[1]
        if from_id == to_id:
            continue
        print("from_id is:{0},to_id is:{1}".format(from_id,to_id))
        A[int(from_id), int(to_id)] = 1
        A[int(to_id), int(from_id)] = 1


print("the type of A is:{}, the size of A is: {}".format(type(A), A.shape))
print(A)
B = sp.coo_matrix(A)
print("the type of B is:{}, the size of B is: {}".format(type(B), B.shape))
print(B)


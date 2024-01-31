import csv
import pandas as pd
import numpy as np
import faiss

def getTrain_Test(fileName, m = 4, n_bits = 8, nlist = 256, w = 64, index_type = 'IVFPQ'):
    list_base, list_query, list_groundtruth = Get_Dataset(fileName)
    '''Get Train'''
    xb = np.array(list_base).astype('float32')
    xq = np.array(list_query).astype('float32')
    d = xb.shape[1]; nb = xb.shape[0]; nq = xq.shape[0]
    pq = faiss.IndexPQ(d, m, n_bits)
    pq.train(xb); pq.add(xb)

    code = pq.sa_encode(xb)
    code = pq.sa_decode(code)

    print("Train: code.shape: {}".format(code.shape))

    path = "data/" + fileName + "/train.csv"
    write2file(code, path)

    del code

    '''Get Test'''
    quantizer = faiss.IndexFlatL2(d)     
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, n_bits)
    index.train(xb)
    index.add(xb)
    index.nprobe = w     
    # search
    D, I = index.search(xq, 1000)
    print("Test: I.shape: {}".format(I.shape))
    I = list(I)
    print("Test: len(I): {}".format(len(I)))

    path = "data/" + fileName + "/test.csv"
    write2file(I, path)

    list_query_data = []
    for tmp_list in I:
        for tmp_Id in tmp_list:
             list_query_data.append(xb[tmp_Id])

    print("Test_vector: len(list_query_data): {}".format(len(list_query_data)))

    list_query_data = np.array(list_query_data).astype('float32')
    if index_type == 'PQ':
        code = pq.sa_encode(list_query_data)
        code = pq.sa_decode(code)
    else:
        code = index.sa_encode(list_query_data)
        code = index.sa_decode(code)
    print("Test_vector: code.shape: {}".format(code.shape))

    path = "data/" + fileName + "/test_vector.csv"
    write2file(code, path)

def write2file(data, path):
    list_data = list(data)
    print("list_data.len: {}".format(len(list_data)))
    with open(path, 'w', encoding='utf-8', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for data in list_data:
            writer.writerow(data)

    print("file:{}, success!".format(path))

def read_DataSet(path):
    data = pd.read_csv(path, header = None)
    total_num = data.shape[0]
    list = []
    for i in range(total_num):
        list.append(data.loc[i, 1:])
    return list

def Get_Dataset(path):
    path_base = path + "/base.csv"
    path_query = path + "/query.csv"
    path_groundtruth = path + "/groundtruth.csv"

    list_base = read_DataSet(path_base); list_query = read_DataSet(path_query)
    list_groundtruth = read_DataSet(path_groundtruth)

    return list_base, list_query, list_groundtruth

def test():
    path = 'GIST1M'
    getTrain_Test(path)

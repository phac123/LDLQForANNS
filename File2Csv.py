import numpy as np
import csv

'''
Convert .fvecs/.ivecs to .csv files.
'''

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def write2file(data, path):
    list_data = list(data)
    print("list_data.len: {}".format(len(list_data)))
    with open(path, 'w', encoding='utf-8', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for data in list_data:
            writer.writerow(data)

    print("file:{}, success!".format(path))

def Vec2Csv(path):
    # init path
    base_path = 'VecData/' + path + '/base.fvecs'
    query_path = 'VecData/' + path + '/query.fvecs'
    groundtruth_path = 'VecData/' + path + '/groundtruth.ivecs'

    # invert
    base = fvecs_read(base_path)
    query = fvecs_read(query_path)
    groundtruth_path = ivecs_read(groundtruth_path)
    base = base.tolist()
    query = query.tolist()
    groundtruth = groundtruth_path.tolist()

    # store path
    base_store_path = 'CsvData/' + path + '/base.csv'
    query_store_path = 'CsvData/' + path + '/query.csv'
    groundtruth_store_path = 'CsvData/' + path + '/groundtruth.csv'

    # write to .csv files
    write2file(base, base_store_path)
    write2file(query, query_store_path)
    write2file(groundtruth, groundtruth_store_path)

def test():
    path = "GIST1M"
    Vec2Csv(path)

# test()






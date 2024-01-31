import GetTrainAndTest
import pandas as pd
import numpy as np


def cal_intersection(x, y):
    sum = 0.0
    for tmp in x:
        if tmp in y:
            sum += 1

    return sum

def Get_Recall(path):
    pred_dataset = pd.read_csv('data/' + path + '/pred_nearest.csv', header = None)
    test_dataset = pd.read_csv('data/' + path + '/pred_nearest.csv', header = None)
    gt_dataset = pd.read_csv(path + "/" + "groundtruth.csv", header = None)

    test_dataset = np.array(test_dataset).astype("int32")
    pred_dataset = np.array(pred_dataset).astype("int32")
    gt_dataset = np.array(gt_dataset).astype("int32")

    print(f"""pred_dataset shape: {pred_dataset.shape},
              gt_dataset shape: {gt_dataset.shape}
    """)

    num_query = pred_dataset.shape[0]     # get the number of query vectors
    K = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    for k in K:
        recall = 0.0
        for i in range(num_query):
            tmp1 = pred_dataset[i, :]
            tmp2 = gt_dataset[i, 1: k + 1]
            num_same = cal_intersection(tmp1, tmp2)
            tmp_recall = (num_same * 1.0) / (k * 1.0)
            recall += tmp_recall
        recall = recall / num_query
        print("Recall@{}: {}%".format(k, recall * 100))

    print("*" * 100)

    for k in K:
        recall = 0.0
        for i in range(num_query):
            tmp1 = pred_dataset[i, : k]
            tmp2 = gt_dataset[i, 1: k + 1]
            num_same = cal_intersection(tmp1, tmp2)
            tmp_recall = (num_same * 1.0) / (k * 1.0)
            recall += tmp_recall
        recall = recall / num_query
        print("Recall@{}: {}%".format(k, recall * 100))

def test():
    path = 'GIST1M'
    Get_Recall(path)

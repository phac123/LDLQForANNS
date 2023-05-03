import csv

import GetTrainAndTest
import torch
import pandas as pd
import torch.nn as nn
import numpy as np


def Get_Pred_test(path):
    pred_dataset = pd.read_csv('data/' + path + '/pred.csv', header = None)   #(num_query * 1000, 128)
    test_dataset = pd.read_csv('data/' + path + '/test.csv', header = None)   #(num_query, 1000)

    num_query = int(test_dataset.shape[0])
    Myset_range = int(test_dataset.shape[1])
    D = pred_dataset.shape[1]
    f = True
    with open('data/' + path + '/pred_test.csv', 'w', encoding = 'utf-8', newline = '') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(num_query):
            if i % 100 == 0:
                print("Get_Pred_test: epoch_query: {}/{}.".format(i, num_query))
            for j in range(Myset_range):
                tmp_list = []
                tmp_Id = i * Myset_range + j
                tmp_list.append(test_dataset.loc[i, j])     
                for k in range(D):       
                    tmp_list.append(pred_dataset.loc[tmp_Id, k])
                if f == True:
                    print("len(tmp_list): {}".format(len(tmp_list)))
                    f = False
                writer.writerow(tmp_list)

def pred_nearest(path):
    query_dataset = pd.read_csv(path + '/query.csv', header = None)
    pred_test_dataset = pd.read_csv('data/' + path + '/pred_test.csv', header = None)

    num_query = query_dataset.shape[0]
    Myset_num = 1000

    list_all_pred_nearest = []
    for i in range(num_query):
        if i % 100 == 0:
            print("pred_nearest epoch_num: {}/{}".format(i, num_query))
        tmp_pred_nearest = []
        query_data = query_dataset.loc[i, 1: ]    
        query_data = list(query_data)             
        query_data = torch.tensor(query_data, dtype=float)
        for j in range(Myset_num):             
            tmp_Id = pred_test_dataset.loc[i * Myset_num + j, 0]     
            tmp_data = pred_test_dataset.loc[i * Myset_num + j, 1:]  
            tmp_data = list(tmp_data)
            tmp_data = torch.tensor(tmp_data, dtype=float)           
            pdist = nn.PairwiseDistance(p=2)
            dis = pdist(query_data, tmp_data)                        
            tmp_list = []; tmp_list.append(tmp_Id); tmp_list.append(dis)  
            tmp_pred_nearest.append(tmp_list)                    

        tmp_pred_nearest = sorted(tmp_pred_nearest, key = sort_by_dis) 
        list_all_pred_nearest.append([t[0] for t in tmp_pred_nearest])  

    GetTrainAndTest.write2file(list_all_pred_nearest, 'data/' + path + '/pred_nearest.csv')

def sort_by_dis(t):
    return t[1]

def test():
    # path = 'SIFT1M'
    # Get_Pred_test(path)
    # pred_nearest(path)
    pass

# test()


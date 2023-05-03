import csv
import pandas as pd
import torch
import NN
from GetTrainAndTest import *


def S2S_Predict(path):
    x_test = pd.read_csv('data/' + path + '/test_vector.csv', header=None).values
    print(f"""   test_vector size: {x_test.shape}
    """)

    # x_test = x_test.reshape(x_test.shape[0], NN.config['m'], -1)

    test_dataset = NN.MyDataset(x_test)
    test_loader = NN.DataLoader(test_dataset, batch_size=NN.config['batch_size'], shuffle=False, pin_memory=True)

    model = NN.MLP(d_model=x_test.shape[1], m = NN.config['m']).to(NN.device)
    model.load_state_dict(torch.load('./extra_model/model.ckpt'))

    preds = NN.predict(test_loader, model, NN.device)
    print("len(preds): {}".format(len(preds)))
    print("preds[0].size: {}".format(preds[0].shape))
    print(type(preds[0]))
    preds = np.array(preds)
    array_preds = np.array(preds).reshape(-1, x_test.shape[1])
    array_preds = array_preds.squeeze()
    # array_preds = array_preds.reshape(array_preds.shape[0], 1, -1)
    # array_preds = array_preds.squeeze()
    print("preds.shape: {}".format(array_preds.shape))
    preds = list(array_preds)
    write2file(preds, 'data/' + path + '/pred.csv')

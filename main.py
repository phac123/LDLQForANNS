import GetTrainAndTest
import File2Csv
import O2E
import Train
import NN
path = 'GIST'
GetTrainAndTest.getTrain_Test(path, 8, 8, 1024, 32, "IVFPQ") 

path = "GIST1M"
dataset_type = 'IVFPQ'
model_type = 'GAN'
config_demo = {
    'seed': 123,
    'select_all': True,
    'valid_ratio': 0.2,
    'n_epoch': 10,
    'batch_size': 64,
    'lr': 1e-4,
    "model_type": "GAN",
    'save_path': './models/model.ckpt',  # Your model will be here.
    'm': 8,                              # the number of subspaces
    "n_critic": 1,
    "in_dim": 128,    
    "workspace_dir": NN.workspace_dir, 
    'dataset_type': 'GIST1M'
}

config = NN.Config(123, True, 0.2, 10, 64, 1e-4, 'GAN', './models/model.ckpt', 8, 1, 128, NN.workspace_dir, 'GIST1M')
trainer = NN.TrainerGAN(config, 128)
trainer.train()

import Predict
Predict.S2S_Predict(path)

import Search
Search.Get_Pred_test(path)
Search.pred_nearest(path)

import GetRecallAndPrecision
GetRecallAndPrecision.Get_Recall(path)

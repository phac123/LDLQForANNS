'''
train: 'EmbeddingData/path/PQ/IVFPQ/embedding.csv'(N, D), 'CsvData/path/base.csv'(N, D);
'''

import NN
import pandas as pd

def train(path, datasety_type, model_type, config):
    config_demo = {
        'seed': 123,  # Your seed number, you can pick your lucky number. :)
        'select_all': True,  # Whether to use all features.
        'valid_ratio': 0.2,  # validation_size = train_size * valid_ration
        'n_epochs': 10,  # Number of epochs.
        'batch_size': 32,
        'learning_rate': 2e-5,
        'early_stop': 400,  # If model has not improved for this many consecutive epochs, stop training.
        'save_path': './models/SIFT10K/PQ/model.ckpt',  # Your model will be here.
        'm': 8  # the number of subspaces
    }
    config = NN.Config(123, True, 0.2, 5000, 512, 2e-3, 400, 'models/' + path + '/' + datasety_type + '/model.ckpt', 8)
    NN.same_seed(config.seed)

    train_data_x = pd.read_csv('EmbeddingData/' + path + '/' + datasety_type + '/embedding.csv', header = None).values
    train_data_y = pd.read_csv('CsvData/' + path + '/base.csv', header = None).values

    train_data_x, valid_data_x = NN.train_valid_split(train_data_x, config.valid_ratio, config.seed)
    train_data_y, valid_data_y = NN.train_valid_split(train_data_y, config.valid_ratio, config.seed)

    # Print out the data size
    print(f""" train_data_x size: {train_data_x.shape}
    valid_data_x size: {valid_data_x.shape}
    train_data_y size: {train_data_y.shape}
    valid_data_y size: {valid_data_y.shape}
    """)

    # Select features
    x_train, x_valid, y_train, y_valid = NN.select_feat(train_data_x, valid_data_x, train_data_y, valid_data_y,
                                                        config.select_all)

    # splite
    x_train = x_train.reshape(x_train.shape[0], config.m, -1)
    x_valid = x_valid.reshape(x_valid.shape[0], config.m, -1)
    y_train = y_train.reshape(y_train.shape[0], config.m, -1)
    y_valid = y_valid.reshape(y_valid.shape[0], config.m, -1)

    # Pring traindata size
    print(f"""x_train size: {x_train.shape}
    x_valid size: {x_valid.shape}
    y_train size: {y_train.shape}
    y_valid size: {y_valid.shape}
    """)

    # Converto to Mydataset
    train_dataset, valid_dataset = NN.MyDataset(x_train, y_train), \
                                NN.MyDataset(x_valid, y_valid)

    # Pytorch data loader loads pytorch dataset into batches
    train_loader = NN.DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True, pin_memory = True)
    valid_loader = NN.DataLoader(valid_dataset, batch_size = config.batch_size, shuffle = True, pin_memory = True)

    '''Start Training !!!'''
    if model_type == 'MLP':
        model = NN.MLP(d_model = x_train.shape[2], m = config.m).to(NN.device)
    elif model_type == 'TransformerEncoder':
        model = NN.TransformerEncoder_bn(d_model=x_train.shape[2], dropout=0.1, m=config.m).to(NN.device)

    config.set_path('models/' + path + '/' + datasety_type + "/" + model_type + '/model.ckpt')


    # model = NN.SelfAttentin_One_Dense(4, x_train.shape[2], y_train.shape[2])
    # model = NN.TransformerEncoder(d_model = x_train.shape[2], dropout = 0.1)
    # model = NN.TransformerEncoder_bn(d_model = x_train.shape[2], dropout = 0.1, m = config.m)
    # model = NN.MLP(d_model=x_train.shape[2])
    # model = NN.SelfAttention_Multi_Dense(x_train.shape[2], m = config.m).to(NN.device)
    # config.set_path('models/' + path + '/' + datasety_type + "/TransformerEncoder" + '/model.ckpt')
    NN.trainer(train_loader, valid_loader, model, config, NN.device)

def test():
    path = 'SIFT'
    train(path, 'PQ')

# test()

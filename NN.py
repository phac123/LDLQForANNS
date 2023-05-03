''' Import Packages '''
import math
import numpy as np
import pandas as pd
import os
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

'''Some Utility Functions'''
def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set_tmp, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(data_set), np.array(valid_set)

def predict(test_loader, model, device):
    model.eval()     # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu().numpy())
    return preds

'''Dataset'''
class MyDataset(Dataset):
    def __init__(self, x, y = None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

'''Neural Network Model'''
class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob = 0.1, attention_probs_dropout_prob=0.1):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention"
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim = -1)(attention_scores)
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)

        return hidden_states

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=32, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=2048, nhead=16
        )
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, mels):
        """
        :param mels:
            mels: (batch size, m, N/m); (batch size, m, d_model)
        :return:
            out: (batch size, m, N/m); (batch size, m, d_model)
        """
        # (batch size, n, d_models) -> (n, batch size, d_models)
        out = mels.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        out = self.pred_layer(out)

        return out

'''Batch Normalization'''
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y

class TransformerEncoder_bn(nn.Module):
    def __init__(self, d_model=32, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=2048, nhead=16
        )
        self.pred_layer1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.bn = BatchNorm(d_model, num_dims = 2)
        self.pred_layer2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, mels):
        """
        :param mels:
            mels: (batch size, m, N/m); (batch size, m, d_model)
        :return:
            out: (batch size, m, N/m); (batch size, m, d_model)
        """
        # (batch size, n, d_models) -> (n, batch size, d_models)
        out = mels.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = self.pred_layer1(out)
        out = self.bn(out)
        out = self.pred_layer2(out)

        return out

'''Batch Normalization'''
class TransformerEncoder_bn(nn.Module):
    def __init__(self, d_model=32, dropout = 0.1, m = 4):
        super().__init__()
        self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=2048, nhead=16
        )
        self.Linear1 = nn.Linear(d_model, d_model)
        self.bn = nn.BatchNorm1d(d_model * m)
        self.pred_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def transpose_for_bn_input(self, x):
        m = x.size()[1]; d = x.size()[2]
        new_x_shape = (x.size()[0], m * d)
        x = x.contiguous().view(*new_x_shape)
        return x, m, d

    def transpose_for_bn_output(self, x, m, d):
        new_x_shape = (x.size()[0], m, d)
        x = x.view(*new_x_shape)
        return x

    def forward(self, mels):
        """
        :param mels:
            mels: (batch size, m, N/m); (batch size, m, d_model)
        :return:
            out: (batch size, m, N/m); (batch size, m, d_model)
        """
        # (batch size, n, d_models) -> (n, batch size, d_models)
        out = mels.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        out = self.Linear1(out)
        bn_input, m, d = self.transpose_for_bn_input(out)
        bn_output = self.bn(bn_input)
        out = self.transpose_for_bn_output(bn_output, m, d)
        out = self.pred_layer(out)

        return out

class MLP(nn.Module):
    def __init__(self, d_model, m):
        super().__init__()
        self.d_model = d_model
        self.m = m
        self.SubLinearList = nn.ModuleList()

        for i in range(m):
            self.SubLinearList.append(self.MLP(int(d_model/m)))

        self.AllLinear = self.MLP(d_model)

    def MLP(self, in_dim):
        return nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, in_dim * 16),
            nn.ReLU(),
            self.cycle_bn(in_dim * 16, in_dim * 8),
            self.cycle_bn(in_dim * 8, in_dim * 4),
            self.cycle_bn(in_dim * 4, in_dim * 2),
            self.cycle_bn(in_dim * 2, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, in_dim)
        )

    def cycle_bn(self, in_dim, out_dim):
        return nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        ''' (batch_size, D) -> (batch_size, m, D/m) -> (m, batch_size, D/m)
         -> (m, batch_size, D/m), train -> (m, batch_size, D/m).'''
        bs = x.shape[0]; d = int(x.shape[1] / self.m)
        x = x.view(bs, self.m, d)     
        x = x.permute(1, 0, 2)        
        x_list = []
        for i in range(self.m):
            tmp = self.SubLinearList[i](x[i])  
            x_list.append(tmp)
        x = torch.tensor([item.cpu().detach().numpy() for item in x_list]).cuda() 


        x = x.permute(1, 0, 2)    
        x = x.contiguous().view(bs, 1, -1)    
        x = x.squeeze()
        out = self.AllLinear(x)

        return out

class MLP1(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.pred_layer = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, d_model * 16),
            nn.ReLU(),
            self.cycle_bn(d_model * 16, d_model * 8),
            self.cycle_bn(d_model * 8, d_model * 4),
            self.cycle_bn(d_model * 4, d_model * 2),
            self.cycle_bn(d_model * 2, d_model),
            self.cycle_bn(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, d_model)
        )

    def cycle_bn(self, in_dim, out_dim):
        return nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.pred_layer(x)
        return out

# Generator
# setting for weight init function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    """
    Input shape: (batch, in_dim)
    Output shape: (batch, out_dim) // out_dim = in_dim
    """
    def __init__(self, in_dim):
        super().__init__()

        # input: (batch, in_dim)
        self.l1 = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, in_dim * 16),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            self.d_bn_relu(in_dim * 16, in_dim * 8),    
            self.d_bn_relu(in_dim * 8, in_dim * 4),      
            self.d_bn_relu(in_dim * 4, in_dim * 2),     
            self.d_bn_relu(in_dim * 2, in_dim),          
        )
        self.l3 = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
        )
        self.apply(weights_init)

    def d_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        y = self.l3(y)

        return y

'''Feature Selection'''
def select_feat(train_data_x, valid_data_x, train_data_y, valid_data_y, select_all = True):
    '''Select usefuel features to delete the Id of xxx_y'''
    x_train = train_data_x; x_valid = valid_data_x
    y_train = train_data_y[:, 1: ]; y_valid = train_data_y[:, 1:]

    return x_train, x_valid, y_train, y_valid

'''Training Loop'''
def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr = config['learning_rate'], momentum = 0.9)

    writer = SummaryWriter()   # Writer of tensorboard

    # if not os.path.isdif('./models'):
    #     os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position = 0, leave = True)

        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)     # Move your data to device
            model = model.to(device)
            # print(x.shape); print(y.shape)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1} / {n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()     # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss: .4f}, Valid loss: {mean_valid_loss: .4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])     # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not imporving, so we halt the training session.')
            return

'''Configurations'''
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    'seed': 123,     # Your seed number, you can pick your lucky number. :)
    'select_all': True,     # Whether to use all features.
    'valid_ratio': 0.2,     # validation_size = train_size * valid_ration
    'n_epochs': 10,     # Number of epochs.
    'batch_size': 32,
    'learning_rate': 2e-5,
    'early_stop': 400,     # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt',     # Your model will be here.
    'm': 8     # the number of subspaces
}

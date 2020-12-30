
from dataset import *
from model import *
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import time

import matplotlib.pyplot as plt

df = pd.read_csv(r'data/BTCUSDT.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)

T = 10
batch_size = 256
encoder_hidden = 128
decoder_hidden = 128
learning_rate = 0.001
epoch = 7000
weight_decay = 0
device = 'cuda'

dataset = TimeSeriesDataset(df,[],'Close',T,1)
train_loader, test_loader = dataset.get_loaders(batch_size)

sample = list(train_loader)[0]
print(sample.X[0])
print(sample.y_prev[0])
print(sample.y_target[0])
print(sample.X.shape)
print(sample.y_prev.shape)
print(sample.y_target.shape)
input_size = dataset.input_size

model = DSTP_rnn(input_size,T,encoder_hidden,decoder_hidden,learning_rate,weight_decay)


def evaluate(model : DSTP_rnn,data_loader : DataLoader,epoch = -1):
    
    batch_size = data_loader.batch_size
    y_pred = []
    y_true = []
    with torch.no_grad():
        for iter, batch in enumerate(data_loader, 1):
            input_weighted, input_encoded = model.Encoder(batch.X.float().to(device),batch.y_prev.float().to(device)) #cuda
            y_pred_price = model.Decoder(input_encoded, batch.y_prev.float().to(device))#cuda
            
            y_true = y_true + batch.y_target.detach().numpy()[:,0].tolist()
            y_pred = y_pred + y_pred_price.cpu().detach().numpy()[:,0].tolist()
            
    if epoch == -1:
        figname = 'plots/prediction.png'
    else:
        figname = 'plots/%s.png'%(str(epoch))
    plt.figure(figsize=(12, 6))
    plt.plot(y_true,label='true')
    plt.plot(y_pred,label='pred')
    plt.legend()
    
    plt.savefig(figname, format='png', bbox_inches='tight', transparent=True)
    plt.close()
   
    
def train(epoch : int,train_loader: DataLoader,test_loader:DataLoader):
    '''
    ref_idx = index of time series [0,1,2,3,4,5....,N]
    batch_size = 128
    T = 10
    input_size = 30
    for each batch
    
    indices = [0,...,127] # batchsize
    x shape = [128,10 - 1, 30]
    y_prev shape = [128,10 - 1]
    y_prev shape = [128,1]
    
    
    
    
    
    '''
        
    train_num = len(train_loader)
    print(f"[Data Info] number of training instances: {train_num}")
    
    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        model.train()
        
        # for iter, batch in tqdm(enumerate(train_loader, 1), desc="--training batch", total=len(train_loader)):
        for iter, batch in enumerate(train_loader, 1):
            model.encoder_optimizer.zero_grad()
            model.decoder_optimizer.zero_grad()
            
            input_weighted, input_encoded = model.Encoder(batch.X.float().to(device),batch.y_prev.float().to(device)) #cuda
            y_pred_price = model.Decoder(input_encoded, batch.y_prev.float().to(device))#cuda

            loss = model.criterion_price(y_pred_price, batch.y_target.float().to(device))
            epoch_loss += loss.item()
            loss.backward()
       
            model.encoder_optimizer.step()
            model.decoder_optimizer.step()
            model.zero_grad()
            
        end_time = time.time()
        
        if i % 10 == 0:
            print("Epoch %d: %.5f, Time is %.2fs\n" % (i, epoch_loss, end_time - start_time), flush=True)
        if i % 1000 == 0 and i!=0 :
              torch.save(model.state_dict(), 'dstprnn_model_{}.pkl'.format(epoch))
              
        if i % 50 == 0 and i != 0:
            evaluate(model,test_loader,epoch = i)

        if i % 100 == 0 and i != 0:
            for param_group in model.encoder_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.95
            for param_group in model.decoder_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.95
                

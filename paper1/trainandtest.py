import argparse
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from  torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from data_loader import load_data
from genvec import genseries
from getemb import geninitemb
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score,roc_curve,plot_roc_curve,auc
   

    
class VarAutoEncoder(nn.Module):
    def __init__(self,pre_emb,input_dim):
        super(VarAutoEncoder,self).__init__()        
        self.entity_emb = nn.Embedding.from_pretrained(pre_emb,freeze=False)
        self.encoder = nn.Sequential(  
            

            nn.Linear(input_dim,128),
            nn.Dropout(0.3),
            nn.Sigmoid(),
            nn.Linear(128,32),
            nn.Sigmoid()
        )   
        
        self.decoder = nn.Sequential(
            nn.Linear(16,128),
            nn.Sigmoid(),
            nn.Linear(128,input_dim),
            nn.Sigmoid()
        )
    

    def forward(self, x):
        batchsz = x.size(0)      
        temp= self.entity_emb(x)       

        x= temp.flatten(1)       

        h_mu_sigma = self.encoder(x)  
        mu, sigma = h_mu_sigma.chunk(2, dim = 1)  

        h = mu + sigma * torch.rand_like(sigma) 
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma,2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) -1) / batchsz        

        x_hat = self.decoder(h)
        return x_hat, kld, x 

def get_recon_err(model,X):
    re=model(X)
    temp=(re[0] - re[2]) ** 2   
    myerr=torch.mean(temp, dim=1).detach().numpy()
    return myerr





def compute_kl_loss( p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')    

    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

def compute_kl_loss( p, q, pad_mask=None):    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')   
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

def plot_test(pre_test_num, test_num, err_test,title, ax):    
    ax.plot(pre_test_num, label='reconstructed')
    ax.plot(test_num, label='true')
   
    ax.set_title(f'{title} (loss: {np.around(err_test, 7)})')
    ax.legend()


def tr_and_te(args):    
    atts=args.atts
    show_loss = False
    data_info = load_data(args)  
    [sers,pca]=genseries(data_info,args.genusernum,args)
    [emb,voc]=geninitemb(args.vecsize,sers,args.dicsize,args.sersize,pca)  
    
    gennum=args.genusernum  
    device = "cpu"
    num_epochs =args.n_epochs 
    atts=args.atts
    df = pd.DataFrame(emb)
    dfwei = pd.DataFrame(voc).values
    x_nom=df.iloc[0:gennum,:]
    x_nov=df.iloc[gennum:gennum+atts,:]
    x_train, x_nom_test = train_test_split(x_nom, train_size = 0.70, random_state = 1)
    x_test = np.concatenate([x_nom_test,x_nov],axis = 0)
    y_test = np.concatenate([np.zeros(len(x_nom_test)),np.ones(len(x_nov))])
    x_train = np.array(x_train)
    x_test = np.array(x_test)     
    x_train,x_test = torch.LongTensor(x_train),torch.LongTensor(x_test)    
    
    train_set = TensorDataset(x_train)
    train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True)  
    
    input_dim = args.vecsize*x_train.shape[1]    
    pre_emb=torch.FloatTensor(dfwei)    
    
    model = VarAutoEncoder(pre_emb,input_dim)   
    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    loss_func = nn.MSELoss()
    all_loss = []
  
    for epoch in range(num_epochs):
        total_loss=0
        l1_reg=0
        for step, (x,) in enumerate(train_loader):
        
            x_recon, kld, rawx= model(x)  
    
            x_recon2, kld2, rawx2= model(x) 
            kl_loss = compute_kl_loss(x_recon,x_recon2,pad_mask=None)
            
            loss = 0.5*(loss_func(x_recon, rawx)+loss_func(x_recon2,rawx))+args.weight*kl_loss   
    
            if kld is not None:
                
                loss = loss+args.weight*kld
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        
        total_loss /= len(train_set)
        all_loss.append(total_loss)   
        
        print("epoch:",epoch,'total_loss',total_loss,'kl_loss',kl_loss.item())
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_epochs),all_loss)
    plt.show()
     
    recon_err_train = get_recon_err(model,x_train)
    recon_err_test = get_recon_err(model,x_test)
    recon_err = np.concatenate([recon_err_train, recon_err_test])
    labels = np.concatenate([np.zeros(len(recon_err_train)), y_test])
    index = np.arange(0, len(labels))
     
    threshold = np.linspace(0, 5, 100)
    acc_list = []
    f1_list = [] 
    
    for t in threshold:
        y_pred = (recon_err_test >t).astype(np.int32)
        acc_list.append(accuracy_score(y_pred, y_test))
        f1_list.append(f1_score(y_test, y_pred, average='binary',pos_label=1))
        
    i = np.argmax(f1_list)
    t = threshold[i]
    score = f1_list[i]
    print('threshold: %.3f,  f1 score: %.3f' % (t, score))
    t=t
    
    
    y_pred = (recon_err_test >t).astype(np.int32)
    actual=y_test
    predicted=y_pred
    
    p = precision_score(actual, predicted, average='binary',pos_label=1)
    p2 = precision_score(actual, predicted, average='macro')
    p3 = precision_score(actual, predicted, average='weighted')    
     
    r = recall_score(actual, predicted, average='binary',pos_label=1)
    r2 = recall_score(actual, predicted, average='macro')
    r3 = recall_score(actual, predicted, average='weighted')
     
    f1score = f1_score(actual, predicted, average='binary',pos_label=1)
    f1score2 = f1_score(actual, predicted, average='macro')
    f1score3 = f1_score(actual, predicted, average='weighted')
    fpr,tpr,tempthre=roc_curve(actual,recon_err_test, pos_label=1,sample_weight=None,drop_intermediate=True)
    
     

    return r,p,f1score,fpr











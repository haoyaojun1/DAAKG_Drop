# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:54:37 2024

@author: hyj
"""

from sklearn.decomposition import PCA
import numpy as np
def genseries(data_info, genuser,args):
    alldata=data_info[0]   
    userdict=data_info[10]   
    allcl=data_info[5]     
    
    y_test = np.concatenate([np.zeros(genuser),np.ones(len(allcl)-genuser)])   
    
    userlen=np.zeros((len(userdict),1))  
    for i in range(len(userdict)):
        if i not in userdict:
            userlen[i,0]=0
        else:
            userlen[i,0]=len(userdict[i])
    
    #对用户评分数进行过滤
    nu_dict={}
    cur=0
    list_ytest=[]
    record_dict={}        
    for i in range(len(userdict)):
        if userlen[i,0]>=1:   
            record_dict[cur]=i
            nu_dict[cur]=userdict[i]
            cur=cur+1
            list_ytest.append(y_test[i].astype(int))
            
  
    gennum=0
    for i in list_ytest:
        if i==0:
            gennum=gennum+1
            
    

    userh_dict={}
    userr_dict={}
    usert_dict={}
    h=args.n_hop        
    for i in range(len(record_dict)):
        realuser=record_dict[i]
        userh_dict[i]=[]
        userr_dict[i]=[]
        usert_dict[i]=[]
        for j in range(h):
            temph=allcl[realuser][j][0]
            tempt=allcl[realuser][j][2]
            tempr=allcl[realuser][j][1]
            
            for k in temph:                   
                userh_dict[i].append(k)
            for l in tempt:                    
                usert_dict[i].append(l)
            for m in tempr:
                userr_dict[i].append(m)        
    
    
    
    harr=np.zeros((len(userh_dict),h*args.n_memory),int)     
    tarr=np.zeros((len(usert_dict),h*args.n_memory),int)    
    
    for i in range(len(userh_dict)):
        harr[i]=np.array(userh_dict[i]) 
    for i in range(len(usert_dict)):
        tarr[i]=np.array(usert_dict[i])
        
    pair_dict=dict()
    for i in range(gennum):   
        for j in range(harr.shape[1]):
            item1=harr[i,j]
            item2=tarr[i,j]
            if (item1,item2) not in pair_dict:
                pair_dict[(item1,item2)]=[]
            if i not in pair_dict[(item1,item2)]:
                pair_dict[(item1,item2)].append(i)
                

    b=harr-harr  
    sortht=harr-harr
    hindex=harr-harr
    tindex=harr-harr
    cid=harr-harr
    did=harr-harr
    
    for i in range(harr.shape[0]):                               
        for j in range(harr.shape[1]):
            if (harr[i,j],tarr[i,j]) not in pair_dict:
                b[i,j]=0
            else:                
                b[i,j]=len(pair_dict[(harr[i,j],tarr[i,j])])  #b每个项目的流行度
                
    
    
    #查找项目对的索引值
    
    # for i in range(harr.shape[0]):
    #     for j in range(harr.shape[1]):
    #         cid[i,j]=pair_id.index((harr[i,j],tarr[i,j]))
 
    for i in range(harr.shape[0]):
                                
                                     
     
        temp=np.sort(b[i,:])  
        temp2=np.argsort(b[i,:])  
        sortht[i]=temp
        
        
    gpair_dict=dict()
    for i in range(gennum):   
        for j in range(harr.shape[1]):
            item1=harr[i,j]
            item2=tarr[i,j]
            if (item1,item2) not in gpair_dict:
                gpair_dict[(item1,item2)]=[]
            if i not in gpair_dict[(item1,item2)]:
                gpair_dict[(item1,item2)].append(i)
          

                
    pair_id=[]
    for pair in pair_dict:
        pair_id.append(pair)



    ur={}
    gcnt=0
    for i in range(len(pair_id)):
        realuser=pair_dict[pair_id[i]]
        for j in realuser:
            # if j<6035:
                if j not in ur:
                    ur[j]=[]
                if i not in ur[j]:
                    ur[j].append(i)
                    gcnt=gcnt+1
                
                   
    m_rate=np.zeros((len(record_dict),len(pair_id)),int)  

    for i in ur:
        allid=ur[i]
        for j in allid:
            temp=pair_id[j]
            if (temp in pair_dict) and (temp not in gpair_dict):
                m_rate[i,j]=1  #len(pair_dict[pair_id[j]])+10
            else:
                m_rate[i,j]=5
 
    pca = PCA(n_components= 5)
    x_new1 = pca.fit_transform(m_rate)        
  
   
    return sortht,x_new1
        

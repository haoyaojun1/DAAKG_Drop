# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:18:33 2024

@author: hyj
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
import pandas as pd





def generate_w2v_feats(df,vsize):


    sentences = df.values.tolist()
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]   
    w2v_model = Word2Vec(sentences, window=12, min_count=3, sg=0, epochs=50, negative=5, hs=1, vector_size=vsize,  seed=15)
    w2v_model.wv.save_word2vec_format('pop.txt',binary=False)
    for i in range(len(sentences)):
        sentences[i] = [w2v_model.wv[x] for x in sentences[i] if x in w2v_model.wv.key_to_index]    
    word_vector_dict={}
    for word in w2v_model.wv.index_to_key:
        word_vector_dict[word]=list(w2v_model.wv[word])     
 

    emb_size=128
    emb_matrix = []
    for seq in sentences:
        if len(seq) > 0:
            emb_matrix.append(np.sum(seq, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix = np.array(emb_matrix)
    return emb_matrix,word_vector_dict



def geninitemb(vecsize,sers,dictnum,sersize,pca):

    df =pd.DataFrame(sers)    
    emb_avg,wdic=generate_w2v_feats(df.iloc[:,0:64],vecsize)
    voc=np.zeros((dictnum,vecsize))    
    for i in range(dictnum):
        if str(i) in wdic:
            voc[i,:]=wdic[str(i)]    
    
    emb=df.iloc[:,0:sersize].values   
    arr_pca =pca    
    scaler = MinMaxScaler()
    arr_pca= scaler.fit_transform(arr_pca)
    emb_joint = np.concatenate((emb,arr_pca),axis=1)
    
    return emb,voc


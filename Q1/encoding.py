# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def getOneHotEncoding(data,mediandict,colend,colstart=0):
    cols=data.columns.values
    distinct_vals=[]
    for col in cols:
        unq=data[col].unique()
        unq.sort()
        distinct_vals.append(unq)
    
    datacopy=data.copy()
    for i,col in enumerate(cols):
        if i in range(colstart,colend):
            if col in mediandict.keys():
                datacopy[col]= datacopy[col].map(lambda x:np.array([mediandict[col]>=x,mediandict[col]<x]).astype('int'))
            else:
                datacopy[col]= datacopy[col].map(lambda x:(distinct_vals[i]==x).astype('int'))
        
    onehot=[]
    for i,row in datacopy.iterrows():
        onehot.append(np.hstack(row.ravel()))
        
    onehot=pd.DataFrame(onehot) ## train one hot encoded - part a) answer
    return onehot
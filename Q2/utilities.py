# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def getHotOneToDec(hotone):
    dec= np.array(hotone)
    max_index = np.argmax(dec, axis=1)
    max_index=np.array(max_index)
    return max_index

def findacc(data,gold):
    data=getHotOneToDec(data)
    gold=getHotOneToDec(gold)
    data=(gold==data)
    data=pd.DataFrame(data)
    data['sum']=data.apply(np.prod, axis=1)
    return sum(data['sum'])/len(data['sum'])

def confusionMatrix(classes,gold,pred):
    cm=np.zeros([len(classes),len(classes)])
    for index,i in enumerate(gold):
        if i==pred[index]:
            cm[i-1][i-1]=cm[i-1][i-1]+1
        else:
            cm[i-1][pred[index]-1]=cm[i-1][pred[index]-1]+1
    return cm
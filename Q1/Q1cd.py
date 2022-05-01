# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from encoding import getOneHotEncoding
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import json

train = pd.read_csv("./bank_dataset/bank_dataset/bank_train.csv",delimiter=";")
val = pd.read_csv("./bank_dataset/bank_dataset/bank_val.csv",delimiter=";")
test = pd.read_csv("./bank_dataset/bank_dataset/bank_test.csv",delimiter=";")

contu={}
cols= train.columns.values
for i in cols:
    contu[i]=0

contu['age']=1
contu['balance']=1
contu['campaign']=1
contu['duration']=1
contu['pdays']=1
contu['previous']=1

mediandict={}
for i in contu.keys():
    if contu.get(i)==1:
       mediandict[i]=train[i].median()

def getEncoding(data,mediandict):
    dataonehot=getOneHotEncoding(data,mediandict,16)
    dataonehotcols=list(dataonehot.columns.values)
    dataonehotstr=[]
    for i in dataonehotcols:
        dataonehotstr.append(str(i))
    
    dataonehot.columns=dataonehotstr
    
    dataonehot['87']=dataonehot['87'].map(lambda x:int(x=='yes'))
    return dataonehot

features=len(train.columns.values)-1

param_grid={'n_estimators':[50,150,250,350,450],
            'max_features':[0.1,0.3,0.5,0.7,0.9],
            'min_samples_split':[2,4,6,8,10]}

trainonehot=getEncoding(train,mediandict)
valonehot=getEncoding(val,mediandict)
testonehot=getEncoding(test,mediandict)

x=trainonehot.iloc[:,:87]
y=trainonehot.iloc[:,-1]

xv=valonehot.iloc[:,:87]
yv=valonehot.iloc[:,-1]

xt=testonehot.iloc[:,:87]
yt=testonehot.iloc[:,-1]
#
#for est in range(nestimators):
#    for maxf in range(nmaxfeatures):
#        for nmin in range(nminsampsplit):
#            model=RandomForestClassifier(n_estimators=50+(100*est),max_features=0.1+(0.2*maxf),min_samples_split=2+(2*nmin),bootstrap=True,oob_score=True)
#            models[est,maxf,nmin]=model
#            score = cross_val_score(model, trainonehot.iloc[:,:87], trainonehot.iloc[:,-1], scoring='accuracy', n_jobs=-1, error_score='raise')

baseestimator = RandomForestClassifier(random_state=0)

sh = GridSearchCV(baseestimator, param_grid,verbose=2.5).fit(x, y)

np.save('./sh',sh)

op=sh.best_params_

np.save('./op',op)

op=np.load('./op.npy',allow_pickle=False)

optimalmodel=RandomForestClassifier(n_estimators=350,max_features=0.3,min_samples_split=10,bootstrap=True,oob_score=True)

#optimalmodel=RandomForestClassifier(n_estimators=op.get('n_estimators'),max_features=op.get('max_features'),min_samples_split=op.get('min_samples_split'),bootstrap=True,oob_score=True)

optimalmodel.fit(x,y)

accuracy_train=optimalmodel.score(x,y) # 94.16 %
accuracy_oob=optimalmodel.oob_score_ # 89.84 %
accuracy_val=optimalmodel.score(xv,yv) # 89.89 %
accuracy_test=optimalmodel.score(xt,yt) # 89.07 %

# accuracies are near about same as compared to the part b

################### d ##########################

min_samples_splits_val=[]
min_samples_splits_test=[]
for i in range(5):
    varmodel=RandomForestClassifier(n_estimators=op.get('n_estimators'),max_features=op.get('max_features'),min_samples_split=2+(2*i),bootstrap=True,oob_score=True)
    varmodel.fit(x,y)
    min_samples_splits_val.append(varmodel.score(xv,yv))
    min_samples_splits_test.append(varmodel.score(xt,yt))
    

max_features_val=[]
max_features_test=[]
for i in range(5):
    varmodel=RandomForestClassifier(n_estimators=op.get('n_estimators'),max_features=0.1+(0.2*i),min_samples_split=op.get('min_samples_split'),bootstrap=True,oob_score=True)
    varmodel.fit(x,y)
    max_features_val.append(varmodel.score(xv,yv))
    max_features_test.append(varmodel.score(xt,yt)) #takes more time to run


n_estimators_val=[]
n_estimators_test=[]
for i in range(5):
    varmodel=RandomForestClassifier(n_estimators=50+(100*i),max_features=op.get('max_features'),min_samples_split=op.get('min_samples_split'),bootstrap=True,oob_score=True)
    varmodel.fit(x,y)
    n_estimators_val.append(varmodel.score(xv,yv))
    n_estimators_test.append(varmodel.score(xt,yt))


saved_var_modelaccs={}
saved_var_modelaccs['min_samples_splits_val']=min_samples_splits_val
saved_var_modelaccs['min_samples_splits_test']=min_samples_splits_test
saved_var_modelaccs['max_features_val']=max_features_val
saved_var_modelaccs['max_features_test']=max_features_test
saved_var_modelaccs['n_estimators_val']=n_estimators_val
saved_var_modelaccs['n_estimators_test']=n_estimators_test

with open('saved_var_modelaccs.json','w') as f:
    json.dump(saved_var_modelaccs,f)

plt.plot(2+np.array(range(5))*2, min_samples_splits_val, label = "Validation")
plt.plot(2+np.array(range(5))*2, min_samples_splits_test, label = "Test")
plt.ylabel('Accuracy')
plt.xlabel('min_samples_splits')
plt.legend()
plt.show()


plt.plot(0.1+np.array(range(5))*0.2, max_features_val, label = "Validation")
plt.plot(0.1+np.array(range(5))*0.2, max_features_test, label = "Test")
plt.ylabel('Accuracy')
plt.xlabel('max_features')
plt.legend()
plt.show()


plt.plot(50+np.array(range(5))*100, n_estimators_val, label = "Validation")
plt.plot(50+np.array(range(5))*100, n_estimators_test, label = "Test")
plt.ylabel('Accuracy')
plt.xlabel('n_estimators')
plt.legend()
plt.show()

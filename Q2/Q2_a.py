# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parta import getOneHotEncoding
from neuralnet import neuralNetwork
from utilities import confusionMatrix,findacc,getHotOneToDec

############### a ##################
def importData():
    train=pd.read_csv('poker-hand-training-true.data',header=None)
    onehottrain=np.array(getOneHotEncoding(train,11,0))
    test=pd.read_csv('poker-hand-testing.data',header=None)
    onehottest=np.array(getOneHotEncoding(test,11,0))
    return train,onehottrain,test,onehottest

train,onehottrain,test,onehottest=importData()

############### c ###################
def trainnn(nn,x,y,iterations=10000,lr=0.1,stoppage=0.000001,return_stat=False,adaptive=False):
    err=[]
    iters=[]
    accs=[]
    for itr in range(iterations):
        nn.trainNN(x,y,lr,adaptive)
        if itr%100 ==0:
            out=nn.feedForward(x)
            e=nn.mbsgd(y,out)*100
            err.append(e)
            a=findacc(out,y)
            accs.append(a)
            iters.append(itr)
            print("Loss: ",e,'% | ','Acc :',a)
            if len(err)>1 and abs(err[-2]-err[-1])<0.000001:
                if return_stat:
                    return iters,accs
                break
    if return_stat:
        return iters,accs

def getAccuracy(nn,gold,data):
    out=nn.feedForward(data)
    acc=findacc(out,gold)
    return acc

# from set {5,10,15,20,25}, as a single layerd NN
nn=neuralNetwork(input_size=85,output_size=10,layers=[25],batch_size=100,layer_act='sigmoid',op_act='sigmoid')

iters,accs_train=trainnn(nn,onehottrain[:,:85],onehottrain[:,85:],1000,0.1,0.000001,True)

acc_train=getAccuracy(nn,onehottrain[:,85:],onehottrain[:,:85])

acc_test=getAccuracy(nn,onehottest[:,85:],onehottest[:,:85])

plt.plot(iters, accs_train, label = "Train Accuracies")
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.legend()
plt.show()

# best is 25 unit single layered NN with acc "92.27%"
# stoppage criterian mbsgd loss less than 0.00001
# time taken to train the newtwork 3.5min for approx 2000 iters
# test accuracy is "91.15%"

output=getHotOneToDec(nn.feedForward(onehottrain[:,:85]))
cm_train=confusionMatrix(list(range(10)),np.array(train)[:,-1],output)

output=getHotOneToDec(nn.feedForward(onehottest[:,:85]))
cm_test=confusionMatrix(list(range(10)),np.array(test)[:,-1],output)

def layersVsAcc(layer_act='sigmoid',op_act='sigmoid',adaptive=False):
    nlayers=[]
    nlayer_accs=[]
    for i in [5,10,15,20,25]:
        print('Single layer with',i,'units')
        nn=neuralNetwork(input_size=85,output_size=10,layers=[i],batch_size=100,layer_act=layer_act,op_act=op_act)
        iters,accs_train=trainnn(nn,onehottrain[:,:85],onehottrain[:,85:],1000,0.1,0.000001,True,adaptive)
    
        nlayers.append(i)
        nlayer_accs.append(max(accs_train))
    return nlayers,nlayer_accs
    

nlay,nlay_acc=layersVsAcc()

plt.plot(nlay, nlay_acc, label = "Train Accuracies vs Units in Layer")
plt.ylabel('Accuracy')
plt.xlabel('No of units in Single layer')
plt.legend()
plt.show()

################ d ######################
nn=neuralNetwork(input_size=85,output_size=10,layers=[25],batch_size=100,layer_act='sigmoid',op_act='sigmoid')

iters,accs_train=trainnn(nn,onehottrain[:,:85],onehottrain[:,85:],1000,0.1,0.000001,True,True)

## train adaptive 92.32 %
acc_train_adapt=getAccuracy(nn,onehottrain[:,85:],onehottrain[:,:85])
## test adaptive 91.96 %
acc_test_adapt=getAccuracy(nn,onehottest[:,85:],onehottest[:,:85])
# yes, adaptive learning is faster and seems the curve to the decent is much smoother
nlay_ada,nlay_acc_ada=layersVsAcc(True)

plt.plot(nlay, nlay_acc, label = "Train Accuracies vs Units in Layer Adaptive")
plt.ylabel('Accuracy')
plt.xlabel('No of units in Single layer')
plt.legend()
plt.show()

output=getHotOneToDec(nn.feedForward(onehottrain[:,:85]))
cm_train_adaptive=confusionMatrix(list(range(10)),np.array(train)[:,-1],output)

output=getHotOneToDec(nn.feedForward(onehottest[:,:85]))
cm_test_adaptive=confusionMatrix(list(range(10)),np.array(test)[:,-1],output)

############# e ###################
nn=neuralNetwork(input_size=85,output_size=10,layers=[100,100],batch_size=100,layer_act='relu',op_act='sigmoid')

iters,accs_train=trainnn(nn,onehottrain[:,:85],onehottrain[:,85:],1000,0.1,0.00000001,True,True)

acc_train_relu=getAccuracy(nn,onehottrain[:,85:],onehottrain[:,:85]) # 92.22 % adaptive # 92.33%

acc_test_relu=getAccuracy(nn,onehottest[:,85:],onehottest[:,:85]) # 89.22 % adaptive # 91.85%

plt.plot(iters, accs_train, label = "Train Accuracies")
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.legend()
plt.show()

output=getHotOneToDec(nn.feedForward(onehottrain[:,:85]))
cm_train_relu=confusionMatrix(list(range(10)),np.array(train)[:,-1],output)

output=getHotOneToDec(nn.feedForward(onehottest[:,:85]))
cm_test_relu=confusionMatrix(list(range(10)),np.array(test)[:,-1],output)

############ f ##################
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(100,100),batch_size=100,learning_rate='adaptive',solver='sgd', max_iter=300,verbose=True)
clf=clf.fit(onehottrain[:,:85],onehottrain[:,85:])

clf_train_acc=clf.score(onehottrain[:,:85],onehottrain[:,85:]) # 96.04 % accuracy

clf_test_acc=clf.score(onehottest[:,:85],onehottest[:,85:]) # 93.78 % acc

#comparison with part e, accuracy by MLP is more 


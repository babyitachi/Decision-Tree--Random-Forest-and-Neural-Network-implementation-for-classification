# -*- coding: utf-8 -*-
import sys
import os.path
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parta import getOneHotEncoding
from neuralnet import neuralNetwork
from utilities import confusionMatrix,findacc,getHotOneToDec
from sklearn.neural_network import MLPClassifier

class Args:
    path_of_train_data=""
    path_of_test_data=""
    part_num=""
    def __init__(self, train, test, part):
        self.path_of_train_data = train
        self.path_of_test_data= test
        self.part_num=part
    
#################### Console Arguments ##################
def read_cli():
    a=sys.argv[1]
    b=sys.argv[2]
    c=sys.argv[3]
    args= Args(a,b,c)

    return args

############## functions ################
def importData(args):
    train=pd.read_csv(args.path_of_train_data,header=None)
    onehottrain=np.array(getOneHotEncoding(train,11,0))
    test=pd.read_csv(args.path_of_test_data,header=None)
    onehottest=np.array(getOneHotEncoding(test,11,0))
    return train,onehottrain,test,onehottest

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

def layersVsAcc(onehottrain,layer_act='sigmoid',op_act='sigmoid',adaptive=False):
    nlayers=[]
    nlayer_accs=[]
    for i in [5,10,15,20,25]:
        print('Single layer with',i,'units')
        nn=neuralNetwork(input_size=85,output_size=10,layers=[i],batch_size=100,layer_act=layer_act,op_act=op_act)
        iters,accs_train=trainnn(nn,onehottrain[:,:85],onehottrain[:,85:],800,0.1,0.000001,True,adaptive)
    
        nlayers.append(i)
        nlayer_accs.append(max(accs_train))
    return nlayers,nlayer_accs

def checkData(): 
    if os.path.exists('./train.npy') and os.path.exists('./onehottrain.npy') and os.path.exists('./test.npy') and os.path.exists('./onehottest.npy'): 
        return True 
    else: 
        print('Please run part a first.') 
        return False

############## parts ####################

def part_a(args):
    train,onehottrain,test,onehottest=importData(args)
    np.save('./train.npy',train)
    np.save('./onehottrain.npy',onehottrain)
    np.save('./test.npy',test)
    np.save('./onehottest.npy',onehottest)

def part_b():
    print("Created a Neural Network")

def part_c():
    if not checkData():
        return
    train=np.load('./train.npy')
    onehottrain=np.load('./onehottrain.npy')
    test=np.load('./test.npy')
    onehottest=np.load('./onehottest.npy')

    # from set {5,10,15,20,25}, as a single layerd NN
    nn=neuralNetwork(input_size=85,output_size=10,layers=[25],batch_size=100,layer_act='sigmoid',op_act='sigmoid')

    iters,accs_train=trainnn(nn,onehottrain[:,:85],onehottrain[:,85:],800,0.1,0.000001,True)
    acc_train=getAccuracy(nn,onehottrain[:,85:],onehottrain[:,:85])
    print('Train Accuracy:',acc_train)

    acc_test=getAccuracy(nn,onehottest[:,85:],onehottest[:,:85])
    print('Test Accuracy:',acc_test)

    plt.plot(iters, accs_train, label = "Train Accuracies")
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()

    output=getHotOneToDec(nn.feedForward(onehottrain[:,:85]))
    cm_train=confusionMatrix(list(range(10)),np.array(train)[:,-1],output)
    print('Train Confusion matrix:','\n',cm_train.astype('int'))

    output=getHotOneToDec(nn.feedForward(onehottest[:,:85]))
    cm_test=confusionMatrix(list(range(10)),np.array(test)[:,-1],output)
    print('Test Confusion matrix:','\n',cm_test.astype('int'))

    nlay,nlay_acc=layersVsAcc(onehottrain)

    plt.plot(nlay, nlay_acc, label = "Train Accuracies vs Units in Layer")
    plt.ylabel('Accuracy')
    plt.xlabel('No of units in Single layer')
    plt.legend()
    plt.show()

def part_d():
    if not checkData():
        return
    train=np.load('./train.npy')
    onehottrain=np.load('./onehottrain.npy')
    test=np.load('./test.npy')
    onehottest=np.load('./onehottest.npy')

    nn=neuralNetwork(input_size=85,output_size=10,layers=[25],batch_size=100,layer_act='sigmoid',op_act='sigmoid')

    iters,accs_train=trainnn(nn,onehottrain[:,:85],onehottrain[:,85:],800,0.1,0.000001,True,True)

    plt.plot(iters, accs_train, label = "Train Accuracies Adaptive")
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()

    ## train adaptive 92.32 %
    acc_train_adapt=getAccuracy(nn,onehottrain[:,85:],onehottrain[:,:85])
    print('Train Accuracy Adaptive:',acc_train_adapt)
    ## test adaptive 91.96 %
    acc_test_adapt=getAccuracy(nn,onehottest[:,85:],onehottest[:,:85])
    print('Test Accuracy Adaptive:',acc_test_adapt)

    output=getHotOneToDec(nn.feedForward(onehottrain[:,:85]))
    cm_train_adaptive=confusionMatrix(list(range(10)),np.array(train)[:,-1],output)
    print('Train Adaptive Confusion matrix:','\n',cm_train_adaptive.astype('int'))

    output=getHotOneToDec(nn.feedForward(onehottest[:,:85]))
    cm_test_adaptive=confusionMatrix(list(range(10)),np.array(test)[:,-1],output)
    print('Test Adaptive Confusion matrix:','\n',cm_test_adaptive.astype('int'))

    nlay_ada,nlay_acc_ada=layersVsAcc(onehottrain,'sigmoid','sigmoid',True)

    plt.plot(nlay_ada, nlay_acc_ada, label = "Train Accuracies vs Units in Layer Adaptive")
    plt.ylabel('Accuracy')
    plt.xlabel('No of units in Single layer')
    plt.legend()
    plt.show()

def part_e():
    if not checkData():
        return
    train=np.load('./train.npy')
    onehottrain=np.load('./onehottrain.npy')
    test=np.load('./test.npy')
    onehottest=np.load('./onehottest.npy')

    nn=neuralNetwork(input_size=85,output_size=10,layers=[100,100],batch_size=100,layer_act='relu',op_act='sigmoid')

    iters,accs_train=trainnn(nn,onehottrain[:,:85],onehottrain[:,85:],800,0.1,0.00000001,True,True)

    acc_train_relu=getAccuracy(nn,onehottrain[:,85:],onehottrain[:,:85]) # 92.22 % adaptive # 92.33%
    print('Train Accuracy Relu:',acc_train_relu)

    acc_test_relu=getAccuracy(nn,onehottest[:,85:],onehottest[:,:85]) # 89.22 % adaptive # 91.85%
    print('Test Accuracy Relu:',acc_test_relu)

    plt.plot(iters, accs_train, label = "Train Accuracies Relu")
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()

    output=getHotOneToDec(nn.feedForward(onehottrain[:,:85]))
    cm_train_relu=confusionMatrix(list(range(10)),np.array(train)[:,-1],output)
    print('Train Relu Confusion matrix:','\n',cm_train_relu.astype('int'))

    output=getHotOneToDec(nn.feedForward(onehottest[:,:85]))
    cm_test_relu=confusionMatrix(list(range(10)),np.array(test)[:,-1],output)
    print('Test Relu Confusion matrix:','\n',cm_test_relu.astype('int'))

    nlay_ada,nlay_acc_ada=layersVsAcc(onehottrain,'relu','sigmoid',True)

    plt.plot(nlay_ada, nlay_acc_ada, label = "Train Accuracies vs Units in Layer Relu adaptive")
    plt.ylabel('Accuracy')
    plt.xlabel('No of units in Single layer')
    plt.legend()
    plt.show()

def part_f():
    if not checkData():
        return
    onehottrain=np.load('./onehottrain.npy')
    onehottest=np.load('./onehottest.npy')

    clf = MLPClassifier(hidden_layer_sizes=(100,100),batch_size=100,learning_rate='adaptive',solver='sgd', max_iter=300,verbose=False)
    clf=clf.fit(onehottrain[:,:85],onehottrain[:,85:])

    clf_train_acc=clf.score(onehottrain[:,:85],onehottrain[:,85:]) # 96.04 % accuracy
    print('Train Accuracy MLP:',clf_train_acc)

    clf_test_acc=clf.score(onehottest[:,:85],onehottest[:,85:]) # 93.78 % acc
    print('Test Accuracy MLP:',clf_test_acc)

def part_g():
    if not checkData():
        return
    train=np.load('./train.npy')
    test=np.load('./test.npy')
    onehottest=np.load('./onehottest.npy')
    
    numlist = [1,10,11,12,13]
    perms=set(itertools.permutations(numlist))
    perms=list(perms)
    
    for i in range(1,5):
        for j in perms:
            train.loc[len(train.index)]=[i,j[0],i,j[1],i,j[2],i,j[3],i,j[4],9]
        
    onehottrain=np.array(getOneHotEncoding(train,11,0))
    
    nn=neuralNetwork(input_size=85,output_size=10,layers=[25],batch_size=100,layer_act='sigmoid',op_act='sigmoid')
    
    iters,accs_train=trainnn(nn,onehottrain[:,:85],onehottrain[:,85:],800,0.1,0.000001,True)
    
    acc_train_g=getAccuracy(nn,onehottrain[:,85:],onehottrain[:,:85]) # 92.561%
    print('Train Accuracy:',acc_train_g)
    
    acc_test_g=getAccuracy(nn,onehottest[:,85:],onehottest[:,:85]) # 92.20%
    print('Test Accuracy:',acc_test_g)
    
    output=getHotOneToDec(nn.feedForward(onehottrain[:,:85]))
    cm_train_g=confusionMatrix(list(range(10)),np.array(train)[:,-1],output)
    print('Train Confusion matrix:','\n',cm_train_g.astype('int'))
    
    output=getHotOneToDec(nn.feedForward(onehottest[:,:85]))
    cm_test_g=confusionMatrix(list(range(10)),np.array(test)[:,-1],output)
    print('Test Confusion matrix:','\n',cm_test_g.astype('int'))


############# main func #####################
if __name__ == '__main__':
    args = read_cli()
    if args.part_num=='a':
        part_a(args)
    elif args.part_num=='b':
        part_b()
    elif args.part_num=='c':
        part_c()
    elif args.part_num=='d':
        part_d()
    elif args.part_num=='e':
        part_e()
    elif args.part_num=='f':
        part_f()
    elif args.part_num=='g':
        part_g()
    else:
        print('Invald part selection.')
    
    
# -*- coding: utf-8 -*-
import sys
import os.path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from encoding import getOneHotEncoding
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class Args:
    path_of_train_data=""
    path_of_test_data=""
    path_of_val_data=""
    part_num=""
    def __init__(self, train, test,val, part):
        self.path_of_train_data = train
        self.path_of_test_data= test
        self.path_of_val_data= val
        self.part_num=part
    
#################### Console Arguments ##################
def read_cli():
    a=sys.argv[1]
    b=sys.argv[2]
    c=sys.argv[3]
    d=sys.argv[4]
    args= Args(a,b,c,d)

    return args

############### functions #############
def getData(args):
    train = pd.read_csv(args.path_of_train_data,delimiter=";")
    test = pd.read_csv(args.path_of_test_data,delimiter=";")
    val = pd.read_csv(args.path_of_val_data,delimiter=";")
    return train,test,val

def computeMI(Hy,y,col):
    sub_dict={}
    for index,row in y.iterrows():
        if row[0] not in sub_dict.keys():
            sub_dict[row[0]]={row[1]:row[2]}
        else:
            sub_dict[row[0]][row[1]]=row[2]
    hrelvals=[]
    total=0
    for vals in sub_dict.keys():
        hyx=0
        v=sub_dict[vals]
        xx=0
        yy=0
        if 'yes' in v:
            xx=v['yes']
        
        if 'no' in v:
            yy=v['no']
        ss=xx+yy
        xx=xx/ss
        yy=yy/ss
        if xx==0 or yy==0:
            hyx=hyx+0
        else:
            hyx=hyx-(xx*np.log2(xx)+yy*np.log2(yy))
        total=total+ss
        hrelvals.append(hyx*ss)
    
    f = Hy - (sum(hrelvals)/total)
    return f,sub_dict

def findTotal(sub_dict):
    tot=0
    for vals in sub_dict.keys():
        v=sub_dict[vals]
        xx=0
        yy=0
        if 'yes' in v:
            xx=v['yes']
        
        if 'no' in v:
            yy=v['no']
        ss=xx+yy
        tot=tot+ss
    return tot

def dataSplit(data,column,sub_dict,isContu=0):
    split_data={}
    if isContu==0:
        for i in sub_dict.keys():
            split_data[i]=data.loc[data[column]==i]
            split_data[i].drop(column,1)
    else:
        sumi=[]
        for i in sub_dict.keys():
            sumi.append(i)
        sumi=np.median(sumi)
        split_data['>='+str(sumi)]=data.loc[data[column]>=sumi]
        split_data['>='+str(sumi)].drop(column,1)
        
        split_data[sumi]=data.loc[data[column]<sumi]
        split_data[sumi].drop(column,1)
    return split_data
    

def MutualInfo(classVal,Y_pos,Y_neg,op_col,contu):
    Pyyes=Y_pos/(Y_pos+Y_neg)
    Pyno=Y_neg/(Y_pos+Y_neg)
    Hy=Pyyes*np.log2(Pyyes)+Pyno*np.log2(Pyno)
    Hy=-1*Hy
    probs=[]
    dicts={}
    i=0
    c=0
    if len(classVal.columns.values)==2:
        cl=classVal.columns.values[0]
        l=classVal.copy()
        l['newcolumn']=1
        y=l.groupby([cl,'y'],axis=0,as_index=False).count()
        prb,dit=computeMI(Hy,y,cl)
        probs.append(prb)
        dicts[i]=dit
        c=0
        l=l[[cl,'y']]
    else:
        for col in classVal:
            if col != op_col:
                y=classVal.groupby([col,'y'],axis=0,as_index=False).count()
                prb,dit=computeMI(Hy,y,col)
                probs.append(prb)
                dicts[i]=dit
            i=i+1
    
        maxval=max(probs)
        c=probs.index(maxval)
        cl=classVal.columns.values[c]
    split_data=dataSplit(classVal,cl,dicts[c],contu[cl])
    return c,split_data

class treeNode():
    def __init__(self,name,nodes,value,pos,neg,iscontu=0):
        self.name=name
        self.cnode=nodes
        self.value= value
        self.iscontu=iscontu
        self.pos=pos
        self.neg=neg

def createtree(data,colname,value,nodes,pdata,ndata,contu,tnodes=-1):
    lcontu=0
    if colname=='':
        lcontu=0
    else:
        lcontu=contu[colname]
        
    if pdata == 0:
        return treeNode('no',[],'no',0,ndata,lcontu)
    if ndata == 0:
        return treeNode('yes',[],'yes',pdata,0,lcontu)
    
    root = treeNode("",[],"",pdata,ndata,0)
    
    if tnodes>0 or tnodes<=-1:
        if len(data.columns.values)>1:
            index,split_data=MutualInfo(data,pdata,ndata,'y',contu)
            name=data.columns.values[index]
            val=[]
            node=[]
            for i in split_data:
                d=split_data.get(i)
                d=d.drop(data.columns.values[index],1)
                node.append(createtree(d,data.columns.values[index],i,[],len(d.loc[d['y']=='yes']),len(d.loc[d['y']=='no']),contu,tnodes-1))
                val.append(i)
            root=treeNode(name,node,val,pdata,ndata,contu[name])
        else:
            if pdata>ndata:
                root.name='yes'
                root.value='yes'
                root.pos=pdata
                root.neg=ndata
            else:
                root.name='no'
                root.value='no'
                root.pos=pdata
                root.neg=ndata
    else:
        if pdata>ndata:
            root.name='yes'
            root.value='yes'
            root.pos=pdata
            root.neg=ndata
        else:
            root.name='no'
            root.value='no'
            root.pos=pdata
            root.neg=ndata
        
    return root

def sizeoftree(tree):
    if tree is None:
        return 0
    else: 
        k=1
        if len(tree.cnode)>0:
            for i in tree.cnode:
                p=sizeoftree(i)
                if p is not None:
                    k=k+ p
            return k
        else:
            return 1

def maxDepth(tree):
    if tree is None:
        return 0
    else :
        depth=[]
        if len(tree.cnode)>0:
            for i in tree.cnode:
                p=maxDepth(i)
                if p is not None:
                    depth.append(p)
            return max(depth)+1
        else:
            return 1

def predict(data,tree):
    while len(tree.cnode)>0:
        d=data[tree.name]
        if tree.iscontu==1:
            p=0
            if d>=tree.value[1]:
                p=0
            else:
                p=1
            tree=tree.cnode[p]
        else:
            if d not in tree.value:
                return 'no'
            p=tree.value.index(d) 
            tree=tree.cnode[p]
    if tree.pos>tree.neg:
        return 'yes'
    else:
        return 'no'

def getaccuracy(data,tree):
    ans=[]
    for i in range(len(data)):
        ans.append(predict(data.iloc[i],tree))
    y=[]
    a=[]
    for i in range(len(data)):
       if data['y'][i]=='no':
           y.append(0)
       else:
           y.append(1)
       if ans[i]=='no':
           a.append(0)
       else:
           a.append(1)
       
    y=np.array(y)
    a=np.array(a)
    
    dataacc= sum((y==a).astype('int'))/len(y)
    return dataacc

def pruneTree(tree):
    if len(tree.cnode):
        pe=abs(tree.pos-tree.neg)
        if pe!=0 and pe!=1:
            pe=pe*np.log(pe)
        else:
            pe=0
        if len(tree.cnode)>0 and len(tree.cnode[0].cnode)==0:
            cl=0
            for i in tree.cnode:
                k=abs(i.pos-i.neg)
                if k!=0 and k!=1:
                    k=k*np.log(k)
                else:
                    k=0
                cl=cl+k
            if pe<=cl:
                tree.cnode=[]
        
    for i,node in enumerate(tree.cnode):
        pruneTree(tree.cnode[i])
    return tree

def getPlotdata(train,val,test,lend,contu,pruning=False):
    accs=[]
    for i in range(1,lend+1):
        print(i)
        ss=[]
        tree=createtree(train,"","",[],len(train.loc[train['y']=='yes']),len(train.loc[train['y']=='no']),contu,i)
        if pruning:
            tree=pruneTree(tree)
        ss.append(sizeoftree(tree))
        ss.append(getaccuracy(train,tree))
        ss.append(getaccuracy(val,tree))
        ss.append(getaccuracy(test,tree))
        accs.append(ss)
    return accs

def getContu(train):
    colDict={}
    for col in train:
        colDict[col]=train[col].drop_duplicates()

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
    return colDict,contu

def getEncoding(data,mediandict):
    dataonehot=getOneHotEncoding(data,mediandict,16)
    dataonehotcols=list(dataonehot.columns.values)
    dataonehotstr=[]
    for i in dataonehotcols:
        dataonehotstr.append(str(i))
    
    dataonehot.columns=dataonehotstr
    
    dataonehot['87']=dataonehot['87'].map(lambda x:int(x=='yes'))
    return dataonehot

arr=[]

def traverse(root):
#    arr.append(root)
   print(root.name,root.value,root.iscontu)
   if len(root.cnode)!=0:
       for i in root.cnode:
           arr.append(i)
   if len(arr)>0:
       traverse(arr.pop())


############# parts ##############
def part_a(train,test,val):
    colDict_multi,contu_multi=getContu(train)
    
    tree=createtree(train,"","",[],len(train.loc[train['y']=='yes']),len(train.loc[train['y']=='no']),contu_multi)

    print('Multi-way split')
    trainacc=getaccuracy(train,tree) # 98.71 %
    print('\t','Train Accuracy: ',trainacc)
    valacc=getaccuracy(val,tree) # 87.41 %
    print('\t','Validation Accuracy: ',valacc)
    testacc=getaccuracy(test,tree) # 86.37 %
    print('\t','Test Accuracy: ',testacc)

    
    mediandict={}
    for i in contu_multi.keys():
        if contu_multi.get(i)==1:
            mediandict[i]=train[i].median()
    
    trainonehot=getOneHotEncoding(train,mediandict,16)
    valonehot=getOneHotEncoding(val,mediandict,16)
    testonehot=getOneHotEncoding(test,mediandict,16)

    trainonehot.rename(columns = {87:'y'}, inplace = True)

    onehotcols=list(trainonehot.columns.values)
    onehotcolsstr=[]
    for i in onehotcols:
        onehotcolsstr.append(str(i))

    contu_one={}
    for i in onehotcolsstr:
        contu_one[i]=0
        
    trainonehot.columns=onehotcolsstr
    valonehot.columns=onehotcolsstr
    testonehot.columns=onehotcolsstr
    del onehotcols

    onehottree=createtree(trainonehot,"","",[],len(trainonehot.loc[train['y']=='yes']),len(trainonehot.loc[train['y']=='no']),contu_one,1)

    print('One-hot encoding')
    trainacc=getaccuracy(trainonehot,onehottree) # 89.64 %
    print('\t','Train Accuracy: ',trainacc)
    valacc=getaccuracy(valonehot,onehottree) # 89.16 %
    print('\t','Validation Accuracy: ',valacc)
    testacc=getaccuracy(testonehot,onehottree) # 88.80 %
    print('\t','Test Accuracy: ',testacc)

    
    pldata=np.array(getPlotdata(train,val,test,len(colDict_multi)-1,contu_multi))
    plt.plot(pldata[:,0], pldata[:,1], label = "Train")
    plt.plot(pldata[:,0], pldata[:,2], label = "Validation")
    plt.plot(pldata[:,0], pldata[:,3], label = "Test")
    plt.xlabel('Number of Nodes in tree')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def part_b(train,test,val):

    colDict,contu=getContu(train)
    
    tree=createtree(train,"","",[],len(train.loc[train['y']=='yes']),len(train.loc[train['y']=='no']),contu)

    pruned_tree=pruneTree(tree)

    print('Pruned tree')
    trainacc=getaccuracy(train,pruned_tree) # 92.60 %
    print('\t','Train Accuracy: ',trainacc)
    valacc=getaccuracy(val,pruned_tree) # 88.10 %
    print('\t','Validation Accuracy: ',valacc)
    testacc=getaccuracy(test,pruned_tree) # 87.48 %
    print('\t','Test Accuracy: ',testacc)

    pldata=np.array(getPlotdata(train,val,test,len(colDict)-1,contu,True))
    plt.plot(pldata[:,0], pldata[:,1], label = "Train")
    plt.plot(pldata[:,0], pldata[:,2], label = "Validation")
    plt.plot(pldata[:,0], pldata[:,3], label = "Test")
    plt.xlabel('Number of Nodes in tree')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def part_c(train,test,val):
    
    colDict,contu=getContu(train)
    mediandict={}
    for i in contu.keys():
        if contu.get(i)==1:
            mediandict[i]=train[i].median()

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

    baseestimator = RandomForestClassifier(random_state=0)

    sh = GridSearchCV(baseestimator, param_grid,verbose=2.5).fit(x, y)
    op=sh.best_params_

    np.save('./op.npy',op)
    
    optimalmodel=RandomForestClassifier(n_estimators=op.get('n_estimators'),max_features=op.get('max_features'),min_samples_split=op.get('min_samples_split'),bootstrap=True,oob_score=True)
    #optimalmodel=RandomForestClassifier(n_estimators=350,max_features=0.3,min_samples_split=10,bootstrap=True,oob_score=True)

    optimalmodel.fit(x,y)

    accuracy_train=optimalmodel.score(x,y) # 94.16 %
    print('Train Accuracy: ',accuracy_train)
    accuracy_oob=optimalmodel.oob_score_ # 89.84 %
    print('Out-of-bag Accuracy: ',accuracy_oob)
    accuracy_val=optimalmodel.score(xv,yv) # 89.89 %
    print('Validation Accuracy: ',accuracy_val)
    accuracy_test=optimalmodel.score(xt,yt) # 89.07 %
    print('Test Accuracy: ',accuracy_test)
    


def part_d(op):
    colDict,contu=getContu(train)
    mediandict={}
    for i in contu.keys():
        if contu.get(i)==1:
            mediandict[i]=train[i].median()

    trainonehot=getEncoding(train,mediandict)
    valonehot=getEncoding(val,mediandict)
    testonehot=getEncoding(test,mediandict)

    x=trainonehot.iloc[:,:87]
    y=trainonehot.iloc[:,-1]

    xv=valonehot.iloc[:,:87]
    yv=valonehot.iloc[:,-1]

    xt=testonehot.iloc[:,:87]
    yt=testonehot.iloc[:,-1]

    min_samples_splits_val=[]
    min_samples_splits_test=[]
    print('min_samples_splits_test started')
    for i in range(5):
        varmodel=RandomForestClassifier(n_estimators=op.get('n_estimators'),max_features=op.get('max_features'),min_samples_split=2+(2*i),bootstrap=True,oob_score=True)
        varmodel.fit(x,y)
        min_samples_splits_val.append(varmodel.score(xv,yv))
        min_samples_splits_test.append(varmodel.score(xt,yt))
    print('min_samples_splits_test done')
        

    max_features_val=[]
    max_features_test=[]
    print('max_features_test started')
    for i in range(5):
        varmodel=RandomForestClassifier(n_estimators=op.get('n_estimators'),max_features=0.1+(0.2*i),min_samples_split=op.get('min_samples_split'),bootstrap=True,oob_score=True)
        varmodel.fit(x,y)
        max_features_val.append(varmodel.score(xv,yv))
        max_features_test.append(varmodel.score(xt,yt)) #takes more time to run
    print('max_features_test done')


    n_estimators_val=[]
    n_estimators_test=[]
    print('n_estimators_test started')
    for i in range(5):
        varmodel=RandomForestClassifier(n_estimators=50+(100*i),max_features=op.get('max_features'),min_samples_split=op.get('min_samples_split'),bootstrap=True,oob_score=True)
        varmodel.fit(x,y)
        n_estimators_val.append(varmodel.score(xv,yv))
        n_estimators_test.append(varmodel.score(xt,yt))
    print('n_estimators_test done')

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


############# main func #####################
if __name__ == '__main__':

    args = read_cli()
    train,test,val=getData(args)

    if args.part_num=='a':
        part_a(train,test,val)

    elif args.part_num=='b':
        part_b(train,test,val)

    elif args.part_num=='c':
        part_c(train,test,val)

    elif args.part_num=='d':
        if os.path.exists('./op.npy'):
            op=np.load('./op.npy',allow_pickle=True)
            op=list(op.flat)[0]
            part_d(op)
        else:
            print('Please run part c, before running part d')

    else:
        print('Invald part selection.')
    
    
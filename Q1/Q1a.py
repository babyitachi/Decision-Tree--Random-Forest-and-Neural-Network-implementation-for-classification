# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
  
train = pd.read_csv("./bank_dataset/bank_dataset/bank_train.csv",delimiter=";")

colDict={}

for col in train:
    colDict[col]=train[col].drop_duplicates()
    
# mutual information
# H(Y,X) = H(Y) - H(Y|X)
# H(Y|X=x) = Sum p(X=x)H(Y|X=x)
# H(Y) = Sum P * log P    
    
y_yes=train.loc[train['y']=='yes'].drop('y', 1)
y_no=train.loc[train['y']=='no'].drop('y', 1)

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
#        sumi=0
#        total=0
        sumi=[]
        for i in sub_dict.keys():
            sumi.append(i)
#        total=findTotal(sub_dict)
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

def createtree(data,colname,value,nodes,pdata,ndata,tnodes=-1):
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
                node.append(createtree(d,data.columns.values[index],i,[],len(d.loc[d['y']=='yes']),len(d.loc[d['y']=='no']),tnodes-1))
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

#indexoffeature=MutualInfo(train,len(y_yes),len(y_no),'y',contu)

tree=createtree(train,"","",[],len(train.loc[train['y']=='yes']),len(train.loc[train['y']=='no']))

#arr=[]
#
#def traverse(root):
##    arr.append(root)
#    print(root.name,root.value,root.iscontu)
#    if len(root.cnode)!=0:
#        for i in root.cnode:
#            arr.append(i)
#    if len(arr)>0:
#        traverse(arr.pop())
#
#traverse(tree)

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

number_nodes=sizeoftree(tree)
height_tree=maxDepth(tree)

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

val = pd.read_csv("./bank_dataset/bank_dataset/bank_val.csv",delimiter=";")
test = pd.read_csv("./bank_dataset/bank_dataset/bank_test.csv",delimiter=";")

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

trainacc=getaccuracy(train,tree) # 98.71 %
valacc=getaccuracy(val,tree) # 87.41 %
testacc=getaccuracy(test,tree) # 86.37 %


def getPlotdata(train,val,test,lend,pruning=False):
    accs=[]
    for i in range(1,lend+1):
        print(i)
        ss=[]
        tree=createtree(train,"","",[],len(train.loc[train['y']=='yes']),len(train.loc[train['y']=='no']),i)
        if pruning:
            tree=pruneTree(tree)
        ss.append(sizeoftree(tree))
        ss.append(getaccuracy(train,tree))
        ss.append(getaccuracy(val,tree))
        ss.append(getaccuracy(test,tree))
        accs.append(ss)
    return accs

pldata=np.array(getPlotdata(train,val,test,len(colDict)-1))

np.save('./pldata.npy',pldata)

pldata=np.load('./pldata.npy')

plt.plot(pldata[:,0], pldata[:,1], label = "Train")
plt.plot(pldata[:,0], pldata[:,2], label = "Validation")
plt.plot(pldata[:,0], pldata[:,3], label = "Test")
plt.xlabel('Number of Nodes in tree')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#######################3 for one hot encoding####################
from encoding import getOneHotEncoding

mediandict={}
for i in contu.keys():
    if contu.get(i)==1:
       mediandict[i]=train[i].median()

train = pd.read_csv("./bank_dataset/bank_dataset/bank_train.csv",delimiter=";")
val = pd.read_csv("./bank_dataset/bank_dataset/bank_val.csv",delimiter=";")
test = pd.read_csv("./bank_dataset/bank_dataset/bank_test.csv",delimiter=";")

trainonehot=getOneHotEncoding(train,mediandict,16)
valonehot=getOneHotEncoding(val,mediandict,16)
testonehot=getOneHotEncoding(test,mediandict,16)

trainonehot.rename(columns = {87:'y'}, inplace = True)

onehotcols=list(trainonehot.columns.values)
onehotcolsstr=[]
for i in onehotcols:
    onehotcolsstr.append(str(i))

contu={}
for i in onehotcolsstr:
    contu[i]=0
    
trainonehot.columns=onehotcolsstr
valonehot.columns=onehotcolsstr
testonehot.columns=onehotcolsstr
del onehotcols

y_yes=trainonehot.loc[trainonehot['y']=='yes'].drop('y', 1)
y_no=trainonehot.loc[trainonehot['y']=='no'].drop('y', 1)

onehottree=createtree(trainonehot,"","",[],len(trainonehot.loc[train['y']=='yes']),len(trainonehot.loc[train['y']=='no']),2)

onehottrainacc=getaccuracy(trainonehot,onehottree) # 89.64%
onehotvalacc=getaccuracy(valonehot,onehottree) # 89.16 %
onehottestacc=getaccuracy(testonehot,onehottree) # 88.80 % 
#one hot encoded tree seems less overfit on training data

pldata=np.array(getPlotdata(trainonehot,valonehot,testonehot,len(contu)-1))

plt.plot(pldata[:,0], pldata[:,1], label = "Train")
plt.plot(pldata[:,0], pldata[:,2], label = "Validation")
plt.plot(pldata[:,0], pldata[:,3], label = "Test")
plt.xlabel('Number of Nodes in tree')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#################### b - pruning ########################
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

pruned_tree=pruneTree(tree)
    
trainacc_pruned=getaccuracy(train,pruned_tree) # 92.60%
valacc_pruned=getaccuracy(val,pruned_tree) # 88.10%
testacc_pruned=getaccuracy(test,pruned_tree) # 87.48 %

pruned_pldata=np.array(getPlotdata(train,val,test,len(colDict)-1,True))

plt.plot(pruned_pldata[:,0], pruned_pldata[:,1], label = "Train")
plt.plot(pruned_pldata[:,0], pruned_pldata[:,2], label = "Validation")
plt.plot(pruned_pldata[:,0], pruned_pldata[:,3], label = "Test")
plt.xlabel('Number of Nodes in Pruned tree')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


plt.plot(pldata[:,0], pldata[:,1], label = "Train")
plt.plot(pruned_pldata[:,0], pruned_pldata[:,1], label = "Pruned Train")
plt.plot(pldata[:,0], pldata[:,2], label = "Validation")
plt.plot(pruned_pldata[:,0], pruned_pldata[:,2], label = "Pruned Validation")
plt.xlabel('Number of Nodes in tree')
plt.ylabel('Accuracy')
plt.legend()
plt.show()




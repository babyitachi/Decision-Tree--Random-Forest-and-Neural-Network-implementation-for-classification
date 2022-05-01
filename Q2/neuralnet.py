# -*- coding: utf-8 -*-
import numpy as np

class neuralNetwork():
    def __init__(self,input_size,output_size,layers,batch_size,layer_act,op_act):
        self.input_size=input_size
        self.output_size=output_size
        self.layers=layers
        self.batch_size=batch_size
        self.layer_act=layer_act
        self.op_act=op_act
        
        self.weights=[]
        a=self.input_size
        for l in range(len(self.layers)+1):
            if l==len(self.layers):
                b=self.output_size
            else:
                b=self.layers[l]
            self.weights.append(np.random.uniform(-1, 1, (a,b)))
            if l!=len(self.layers):
                a=self.layers[l]
            
    def activaton(self, func, z, derivative=False):
        if func=='sigmoid':
            if derivative==True: 
                return z*(1-z)
            return 1/(1+np.exp(-z))
        if func=='relu':
            if derivative==True: 
                z[z<=0] = 0
                z[z>0] = 1
                return z
            return np.maximum(0, z)
        return 1
    
    def mbsgd(self,y,out,derivative=False):
        if derivative:
            return (y-out)
        return np.mean(np.sum(np.square(y-out),axis=0))/(2*len(y))
        
    
    def feedForward(self,x):
        self.layerop=[] 
        a=x
        for l in range(len(self.layers)+1):
            b=self.weights[l]
            nw=np.dot(a,b)
            if l!=len(self.layers):
                nw=self.activaton(self.layer_act,nw)
                self.layerop.append(nw)
                a=nw
            else:
                nw=self.activaton(self.op_act,nw)
                return nw
    
    def backPropogate(self,x,y,out,alpha):
        error_op = self.mbsgd(y,out,True)
        del_op = error_op*self.activaton(self.op_act,out,True)
        
        a=del_op
        for l in range(len(self.layers),-1,-1):
            if l==0:
                b=x
            else:
                b=self.layerop[l-1]
            error_l=np.dot(a,self.weights[l].T)
            del_l=error_l*self.activaton(self.layer_act,b,True)
            self.weights[l]=self.weights[l]+alpha*np.dot(b.T,a)
            a=del_l
        
    def trainNN(self,X,Y,alpha,adaptive=False):
        for i in range(int(len(Y)/self.batch_size)):
            x=X[self.batch_size*i:self.batch_size+self.batch_size*i]
            y=Y[self.batch_size*i:self.batch_size+self.batch_size*i]
            op=self.feedForward(x)
            ualpha=alpha
            if adaptive:
                if i!=0:
                    ualpha=alpha/np.sqrt(i)
            self.backPropogate(x,y,op,ualpha)
        return op
        
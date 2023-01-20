# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 00:23:34 2022

@author: webot
"""
import torch
from torch import nn
def tes(*args,k=1):
    for i in args:
        (n,m)=i.shape
        x1=eval("k")
        #print(n,m,k,x1)

tes(torch.rand(3,5),k=666)
#%%

#%% 计算度矩阵

def degAT(AT):
    #print(AT)
    D=torch.sum(AT,dim=0)
    D=torch.pow(D,-0.5)
    #print(D)
    D12=torch.diag(D)
    #print(D12)   
    return D12
# 计算相关矩阵的邻接矩阵
def fromXtoA(x):
    
    if x.shape[1]>1:
        X=torch.corrcoef(x)
        
        a = X.min()
        b = X.max()
        
        X_transformed = torch.where(X < a + (b - a) / 2, torch.tensor(0.), torch.tensor(1.))
        X_transformed = torch.tril(X_transformed, -1)
        for i in range(X_transformed.shape[0]):
            X_transformed[i,i]=1
    else:
        X_transformed=torch.eye(x.shape[0])
            
    return degAT(X_transformed)


class Model(nn.Module):
    def __init__(self,*args,K,L1,L2,out):
        super(Model,self).__init__()
        """
         输入参数是N个网络的特征数据
         X1=m*n m个节点 n个特征
         X2= k*u k个节点 u 个特征
         ....
         
         Xn
         
         数学原理模型
         W1*X1*W_1+b1 X1=n*m-> W1=n*n W_1=m*K b1=n*1 ->out=n*K
         W2*X2*W_2+b2 X1=nn*mm->W2=nn*nn W_2=mm*K b2= nn*1 ---> out=nn*K
         .......
         Wn*Xn*W_n+bn
         
         X1*X2->X12  
         X1*X3->X13
         
    
        """
        idx=0
        for Xi in args:
            idx+=1
            (n,m)=Xi.shape
            exec(f"self.W{idx}=nn.Parameter(torch.rand(m,K))")
            exec(f"self.b{idx}=nn.Parameter(torch.rand(n,1))")
            
        
        self.relu=nn.LeakyReLU()
        self.tanh=nn.Tanh()
        self.L1=nn.Linear(K,out_features=K)
        
        self.ler1=nn.Linear(in_features=K,out_features=L1)
        self.ler2=nn.Linear(in_features=idx*L1, out_features=L2)
        self.out=nn.Linear(in_features=L2,out_features=out)
   
       
    def forward(self,*args):
        idx=0
        x_i=torch.empty(0)
        for Xi in args:
            idx+=1
            Wi=fromXtoA(Xi)
            xi=eval(f"Wi.mm(Xi).mm(self.W{idx})+self.b{idx}")
            xi=self.relu(xi)
            xi=self.L1(xi)
            
            xi=self.relu(xi)
            
            xi=self.ler1(xi)
            xi=self.relu(xi.sum(dim=0))
            #print(x_i.shape,xi.shape)
            x_i=torch.cat((x_i,xi),dim=0)
            # if idx>=3:
               # print(x_i)
            
        #print(x_i)    
        xl2=self.ler2(self.relu(x_i))
        xl2=self.tanh(xl2)
        out=self.out(xl2)
        return out
        
    
        
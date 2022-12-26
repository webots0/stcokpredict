# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 04:52:04 2022

@author: webot
"""

print('----------')


# 新闻数据，随时间变化的一个网络结构

# 股票数据,随时间变化的一个单序列数据

'''
网络： X 特征数据 H 图结构  

股票： Y 特征数据

X-H来发现训练发现与Y特征的相关性,以及预测未来的Y

很合理
'''
import torch
import numpy as np

from torch import nn
# 5个节点
node=5
# 12 条边
eg=12

# 5个节点，每个节点三个特征，
XT=torch.Tensor(np.random.rand(node,3))

# 5个节点的节点邻接矩阵
AT=torch.Tensor(np.random.randint(2,size=(node,node)))
for i in range(5):
    AT[i][i]=1

# 计算度矩阵
def degAT(AT):
    #print(AT)
    D=torch.sum(AT,dim=0)
    D=torch.pow(D,-0.5)
    #print(D)
    D12=torch.diag(D)
    #print(D12)   
    return D12

# 12条边，每条边4个特征
ET=torch.Tensor(np.random.rand(12,4))

# 前6个数据点
YT=torch.Tensor(np.random.rand(6,1))

# 预测最后3个数据点
yt=torch.Tensor(np.random.rand(3,1))

# 12*4 

def gcn(AT,XT,ET,outSize):
    D12=degAT(AT)
    W1=torch.Tensor(np.random.rand(XT.shape[1],outSize))
    W2=torch.Tensor(np.random.rand(ET.shape[1],outSize))
    b1=torch.Tensor(np.random.rand(AT.shape[1],1))
    AX=D12.mm(AT).mm(D12).mm(XT).mm(W1)
    WX=torch.sum(ET.mm(W2))
    y=AX+WX+b1
    return y


H1=gcn(AT,XT,ET,7)
    
def corssNN(XT,ET,YT):
    # XT 5*3   W3       YT 6*1
    # 5*3 3*6  6*1 5*1
    W3=torch.Tensor(np.random.rand(XT.shape[1],YT.shape[0]))
    
    xyT=XT.mm(W3).mm(YT)
    # ET 12*4 W4        YT 6*1
    # 12*4 4*6 6*1  12*1
    W4=torch.Tensor(np.random.rand(ET.shape[1],YT.shape[0]))
    eyT=torch.sum(ET.mm(W4).mm(YT))
    
    b2=torch.Tensor(np.random.rand(XT.shape[0],1))
    
    y=xyT+eyT+b2
    return y
    
    
H2=corssNN(XT,ET,YT)
print(H1.shape)
print(H2.shape)
    

def mutH1H2(H1,H2,outSize):
    W5=torch.Tensor(np.random.rand(H1.shape[1],H2.shape[0]))
    W6=torch.Tensor(np.random.rand(H2.shape[1],outSize))
    H12=H1.mm(W5).mm(H2).mm(W6)
    return H12

H12=mutH1H2(H1,H2,12)

Linear=nn.Linear(12,3)
out=Linear(H12)
print(H12.shape,out.shape)




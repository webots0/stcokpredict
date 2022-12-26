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

def gcn(AT,XT,ET,W1,W2,b1):
    D12=degAT(AT)
    # W1=torch.Tensor(np.random.rand(XT.shape[1],outSize1))
    # W2=torch.Tensor(np.random.rand(ET.shape[1],outSize1))
    # b1=torch.Tensor(np.random.rand(AT.shape[1],1))
    
    AX=D12.mm(AT).mm(D12).mm(XT).mm(W1)
    WX=torch.sum(ET.mm(W2))
    y=AX+WX+b1
    return y


#H1=gcn(AT,XT,ET,7)
    
def corssNN(XT,ET,YT,W3,W4,b2):
    # XT 5*3   W3       YT 6*1
    # 5*3 3*6  6*1 5*1
    # W3=torch.Tensor(np.random.rand(XT.shape[1],YT.shape[0]))
    # W4=torch.Tensor(np.random.rand(ET.shape[1],YT.shape[0]))
    # b2=torch.Tensor(np.random.rand(XT.shape[0],1))
    
    
    xyT=XT.mm(W3).mm(YT)
    # ET 12*4 W4        YT 6*1
    # 12*4 4*6 6*1  12*1
    
    eyT=torch.sum(ET.mm(W4).mm(YT))
    
   
    
    y=xyT+eyT+b2
    return y
    
    
#H2=corssNN(XT,ET,YT)


def mutH1H2(H1,H2,W5,W6):
    # W5=torch.Tensor(np.random.rand(H1.shape[1],H2.shape[0]))
    # W6=torch.Tensor(np.random.rand(H2.shape[1],outSize2))
    
    H12=H1.mm(W5).mm(H2).mm(W6)
    return H12

#H12=mutH1H2(H1,H2)



#%%

class Quadratic(nn.Module):
    def init(self, a, b, c):
        super(Quadratic, self).init()
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))
        self.c = nn.Parameter(torch.tensor(c))
    
    def forward(self, x):
        
        return self.a * x**2 + self.b * x + self.c

#%% 
class Model(nn.Module):
    def init(self,AT,XT,ET,YT,outSize1,outSize2):
        """
        W1=torch.Tensor(np.random.rand(XT.shape[1],outSize1))
        W2=torch.Tensor(np.random.rand(ET.shape[1],outSize1))
        b1=torch.Tensor(np.random.rand(AT.shape[1],1))
        W3=torch.Tensor(np.random.rand(XT.shape[1],YT.shape[0]))
        W4=torch.Tensor(np.random.rand(ET.shape[1],YT.shape[0]))
        b2=torch.Tensor(np.random.rand(XT.shape[0],1))
        W5=torch.Tensor(np.random.rand(H1.shape[1],H2.shape[0]))
        W6=torch.Tensor(np.random.rand(H2.shape[1],outSize2))
    
        """
        W1=torch.Tensor(np.random.rand(XT.shape[1],outSize1))
        W2=torch.Tensor(np.random.rand(ET.shape[1],outSize1))
        W3=torch.Tensor(np.random.rand(XT.shape[1],YT.shape[0]))
        W4=torch.Tensor(np.random.rand(ET.shape[1],YT.shape[0]))
        b1=torch.Tensor(np.random.rand(AT.shape[1],1))
        b2=torch.Tensor(np.random.rand(XT.shape[0],1))
        self.W1=nn.Parameter(W1)
        self.W2=nn.Parameter(W1)
        self.W3=nn.Parameter(W3)
        self.W4=nn.Parameter(W4)
        self.b1=nn.Parameter(b1)
        self.b2=nn.Parameter(b2)
        
        H1= gcn(AT,XT,ET,W1,W2,b1)
        H2= corssNN(XT,ET,YT,W3,W4,b2)
        
        W5=torch.Tensor(np.random.rand(H1.shape[1],H2.shape[0]))
        W6=torch.Tensor(np.random.rand(H2.shape[1],outSize2))
        self.W5=nn.Parameter(W5)
        self.W6=nn.Parameter(W6)
        
        self.relu=nn.relu()
        self.sigmoid=nn.sigmoid()
        
    def forward(self,AT,XT,ET,YT):
        
        H1=gcn(AT,XT,ET,self.W1,self.W2,self.b1)
        H1=self.relu(H1)
        H2=corssNN(XT,ET,YT,self.W3,self.W4,self.b2)
        H2=self.relu(H2)
        H12=mutH1H2(H1,H2,self.W5,self.W6)
        H12=self.sigmoid(H12)
        
        return H12
        
    
    
    
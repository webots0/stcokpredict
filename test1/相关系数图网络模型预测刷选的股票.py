# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 00:09:50 2022

@author: webot
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 02:37:26 2022

@author: webot
"""


import torch
import numpy as np
from torch import nn

# 将相关矩阵转为节点的邻接矩阵

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
            
    return X_transformed

X=torch.rand(3,2)
a=fromXtoA(X)

#print(a)
#%% 新闻数据，随时间变化的一个网络结构

# 股票数据,随时间变化的一个单序列数据

'''
网络： X 特征数据 H 图结构  

股票： Y 特征数据

X-H来发现训练发现与Y特征的相关性,以及预测未来的Y

很合理
'''

# 5个节点
node=5
# 12 条边
eg=12
T=list(range(0,500))
T=[x/30 for x in T]
idx=0
allXT=[]
allAT=[]
allET=[]
allYT=[]
allyt=[]
for t in T:
    if idx>7:
        kxt=torch.Tensor(np.array([1,2,3])*t)
        # 5个节点，每个节点三个特征，
        XT=torch.Tensor(np.random.rand(node,3))*kxt
        allXT.append(XT)
        
        # 5个节点的节点邻接矩阵
        if idx==8:
            AT=torch.Tensor(np.random.randint(2,size=(node,node)))
            for i in range(5):
                AT[i][i]=1
            
        allAT.append(AT)
        
        ket=torch.Tensor(np.array([4,3,2,1])*t)
        # 12条边，每条边4个特征
        ET=torch.Tensor(np.random.rand(12,4))*ket
        allET.append(ET)
        
        # 前6个数据点
        dt=np.array(T[idx-7:idx-1])
        dt=np.sin(dt)**2
        YT=torch.Tensor(dt).view(-1,1).t()
        allYT.append(YT)
        
        # 预测最后3个数据点
        dt=np.array(T[idx-1:idx])
        dt=np.sin(dt)**2
        yt=torch.Tensor(dt)
        allyt.append(yt)
        
    idx+=1

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



class Gint:
    def __init__(self,XT,YT,ET,H1out1,H1out2,H2out1,H2out2):
        # XT 5*3
        linear1=nn.Linear(in_features=XT.shape[1],out_features=1)
        linear2=nn.Linear(in_features=YT.shape[1],out_features=1)
        self.XYT=torch.cat((linear1(XT),linear2(YT)),dim=0)
        linear3=nn.Linear(in_features=ET.shape[1],out_features=1)
        linear4=nn.Linear(in_features=YT.shape[1],out_features=1)
        self.EYT=torch.cat((linear3(ET),linear4(YT)),dim=0)
        
        self.H1out1=H1out1
        self.H1out2=H1out2
        self.H2out1=H2out1
        self.H2out2=H2out2
        
    def graf(self,matrix):
        
        matrix = torch.where(matrix > 0, torch.tensor(1.), torch.tensor(0.)) # 将矩阵里面的大于0的变成1，小于等于0的变成0

        for i in range(matrix.size(0)):
            matrix[i, i] = 1. # 将矩阵对角线赋值为1
        matrix=degAT(matrix)# 求度矩阵
        return matrix
    
    def getWb(self,AT,XT,ET):
        # AT 5*5 XT=5*3 W1 3*H1out1
        W1=torch.rand(XT.shape[1],self.H1out1)
        b1=torch.rand(XT.shape[0],1)
        
        # W2=12*12   ET=12*4 W3 4*H1out2
        W2=torch.randn(ET.shape[0],ET.shape[0])
        #W2=self.graf(W2)
        W3=torch.rand(ET.shape[1],self.H1out2)
        b2=torch.rand(ET.shape[0],1)
        
        # W4 6*6 XYT=(5+1)*1 W5 1*1 b3 6*1
        W4=torch.randn(self.XYT.shape[0],self.XYT.shape[0])
        W5=torch.rand(self.XYT.shape[1],self.H2out1)
        #W4=self.graf(W4)
        b3=torch.rand(self.XYT.shape[0],1)
        
        # W6=13*13   EYT=(12+1)*1 W7=1*1 b4=13*1
        W6=torch.randn(self.EYT.shape[0],self.EYT.shape[0])
        W7=torch.rand(self.EYT.shape[1],self.H2out2)
        #W6=self.graf(W6)
        b4=torch.rand(self.EYT.shape[0],1)
        
        
        
        return (W1,W2,W3,W4,W5,W6,W7,b1,b2,b3,b4)
clsA=Gint(XT,YT,ET,1,1,1,1)

(W1,W2,W3,W4,W5,W6,W7,b1,b2,b3,b4)=clsA.getWb(AT,XT,ET)



def graf0(matrix):
        
    matrix = torch.where(matrix > 0, torch.tensor(1.), torch.tensor(0.)) # 将矩阵里面的大于0的变成1，小于等于0的变成0

    for i in range(matrix.size(0)):
        matrix[i, i] = 1. # 将矩阵对角线赋值为1
    matrix=degAT(matrix)# 求度矩阵
    return matrix
def grafW(matrix):
    matrix = torch.where(matrix > 0, torch.tensor(1.), torch.tensor(0.)) # 将矩阵里面的大于0的变成1，小于等于0的变成0

    for i in range(matrix.size(0)):
        matrix[i, i] = 1. # 将矩阵对角线赋值为1
    return matrix

aa=grafW(W6)



class Model(nn.Module):
    def __init__(self,W1,W3,W5,W7,b1,b2,b3,b4):
        super(Model,self).__init__()
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
       
        self.W1=nn.Parameter(W1)
        #self.W2=nn.Parameter(W2)
        self.W3=nn.Parameter(W3)
        #self.W4=nn.Parameter(W4)
        self.W5=nn.Parameter(W5)
        #self.W6=nn.Parameter(W6)
        self.W7=nn.Parameter(W7)
        self.b1=nn.Parameter(b1)
        self.b2=nn.Parameter(b2)
        self.b3=nn.Parameter(b3) 
        self.b4=nn.Parameter(b4)
      
      
        
        self.relu=nn.LeakyReLU()
        self.tanh=nn.Tanh()
        self.L1=nn.Linear(in_features=XT.shape[1],out_features=1)
        self.L2=nn.Linear(in_features=YT.shape[1],out_features=1)
        self.L3=nn.Linear(in_features=ET.shape[1],out_features=1)
        self.L4=nn.Linear(in_features=YT.shape[1],out_features=1)
        self.ler1=nn.Linear(in_features=36,out_features=18)
        self.ler2=nn.Linear(in_features=18, out_features=1)
    def XEY(self,XT,YT,ET):
        
        
        XYT=torch.cat((self.L1(XT),self.L2(YT)),dim=0)

        EYT=torch.cat((self.L3(ET),self.L4(YT)),dim=0)
        return (XYT,EYT)
    def H1T(self,AT,XT,ET):
        H1=graf0(AT).mm(XT).mm(self.W1)+self.b1
        #W2=graf0(W2)
        W2=fromXtoA(ET)
        H2=graf0(W2).mm(ET).mm(self.W3)+self.b2
        return (H1,H2)
    
    def H2T(self,XT,YT,ET):
        (XYT,EYT)=self.XEY(XT,YT,ET)
        W4=fromXtoA(XYT)
        W6=fromXtoA(EYT)
        H1=graf0(W4).mm(XYT).mm(self.W5)+self.b3
        H2=graf0(W6).mm(EYT).mm(self.W7)+self.b4
        return (H1,H2)

    def ford(self,AT,XT,ET,YT):
        
        (H1,E1)=self.H1T(AT,XT,ET)
        (H2,E2)=self.H2T(XT,YT,ET)
        
        return (H1,H2,E1,E2)
    def forward(self,AT,XT,ET,YT,outS):
      
        (H1,H2,E1,E2)=self.ford(AT,XT,ET,YT)
        H12=torch.cat((H1.view(-1),H2.view(-1)),dim=0)
        E12=torch.cat((E1.view(-1),E2.view(-1)),dim=0)
        HE12=torch.cat((H12.view(-1),E12.view(-1)),dim=0)
        
        
        out=self.ler1(self.tanh(HE12))
        out=self.ler2(out)
        
        return out
    
md=Model(W1, W3, W5, W7, b1, b2, b3, b4)

a=md(AT,XT,ET,YT,1)
print(a)    
 
loss_fn = nn.MSELoss()
#loss_fu= nn.L1Loss()
optimizer = torch.optim.SGD(md.parameters(), lr=0.01)


yt=yt.view(-1)
loss=0
w110=[]
w22=[]
w33=[]
idf=0
for epoch in range(300):
# 计算预测值
    idx=0
    loss=[]
    for i in allAT:
        AT=i
        XT=allXT[idx]
        ET=allET[idx]
        YT=allYT[idx]
        yt=allyt[idx].view(-1)
        
        # 清空梯度    
        
        optimizer.zero_grad()
        
        y_pred = md(AT,XT,ET,YT,1)
        # 计算损失
        #print(y_pred,yt)
        loss0 =loss_fn(y_pred, yt)
        
        loss.append(loss0)
       
       
        idx+=1
        
        # 计算梯度
        
        loss0.backward()
        lr = 0.01;
        for name, param in md.named_parameters():
            if name == 'W1':
                param.data*=1
        optimizer.step()
        # if epoch>50:
            
        #     for param_group in optimizer.param_groups:
                
        #         param_group['lr'] = 0.01
                
        # 更新参数
        if idx==250:
            break
            
        #     tW=md.state_dict()
        #     w00=tW['W1']
            
        
    loss1=sum(loss)/len(loss)
    #print(loss1)
    if (epoch+1)%10==0:
        # md0=md;
        # tW=md0.state_dict()
        #w1=tW['W1']
        #print('W1====',w1)
        #w110.append(md.state_dict())
        # print(tW['W1'])
        # w2=tW['W2']
        # w3=tW['W3']
        
        # w22.append(w2)
        # w33.append(w3)
        # 创建权重数据目录
        # import os
        # if not os.path.exists('权重数据'):
        #     os.makedirs('权重数据')
        
        # 保存权重值
        #torch.save(model.state_dict(), '权重数据/model_weights.pt')
        torch.save(md.state_dict(), 'WeightData/model_weights_{}.pt'.format(epoch+1))
       
        print(f'----{epoch+1}-----',loss1.tolist())
        
#%%
testW=md.state_dict()
# w2=grafW(testW['W1'])
# w4=grafW(testW['W3'])
# w6=grafW(testW['W5'])
# print(w4)

import matplotlib.pyplot as plt
y0=np.sin(T)**2
plt.plot(y0[0:500],color='blue')
yt=y0[0:6]
y00=yt.tolist()
yt=torch.Tensor(yt).view(-1,1).t()
idx=0
for i in allAT:
    
    histy=yt[:,1:]
    AT=i
    XT=allXT[idx]
    ET=allET[idx]
    if idx<250:
        yt=allYT[idx]
    y_pred = md(AT,XT,ET,yt,1).view(-1,1)
    y00.append(y_pred[0].tolist()[0])
    yt=torch.cat((histy,y_pred),dim=-1)
    idx+=1
    #break
    
plt.plot(y00[0:500],color='red')

#%% 线性模型预测序列
import torch
import torch.nn as nn

# 定义线性层模型
class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 60)
        self.linear2=nn.Linear(60,output_size)
    
    def forward(self, x):
        x=self.linear1(x)
        x=self.linear2(x)
        return x

# 初始化模型
model = LinearModel(6, 1)

# 定义输入数据
inputs = torch.randn(1, 6)

# 进行预测
outputs = model(YT)

print(outputs)


loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(200):
    
    idx=0
    loss=[]
    for i in allAT:
        AT=i
        XT=allXT[idx]
        ET=allET[idx]
        YT=allYT[idx]
        yt=allyt[idx].view(-1)
        y_pred = model(YT)
        # 计算损失
        #print(y_pred,yt)
        loss0 =loss_fn(y_pred, yt)
        
        loss.append(loss0)
       
       
        idx+=1
        if idx==250:
            print('----------')
            break
    # 清空梯度    
    loss1=sum(loss)/len(loss)
    optimizer.zero_grad()
    # 计算梯度
    loss1.backward()
    
    # 更新参数
    optimizer.step()
   
   # print(loss1)
    if epoch%10==0:
        print('-----2----',loss1.tolist())
        if loss1<0.0001:
            break
        
    
#%%    
import matplotlib.pyplot as plt
y0=np.sin(T)**2
plt.plot(y0[0:500],color='blue')
yt=y0[0:6]
y00=yt.tolist()
yt=torch.Tensor(yt)
idx=0
for i in allAT:
    histy=yt[1:]
    AT=i
    XT=allXT[idx]
    ET=allET[idx]
    #yt=allYT[idx]
    y_pred = model(yt).view(-1,1)
    y00.append(y_pred[0].tolist()[0])
    yt=torch.cat((histy,y_pred[0]),dim=0)
    idx+=1
    #break
        
plt.plot(y00[0:500],color='green')








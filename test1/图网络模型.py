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
T=list(range(0,50))
T=[x/12 for x in T]
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
        YT=torch.Tensor(dt).view(-1,1)
        allYT.append(YT)
        
        # 预测最后3个数据点
        dt=np.array(T[idx-1:idx])
        dt=np.sin(dt)**2
        yt=torch.Tensor(dt)
        allyt.append(yt)
        
    idx+=1



#%% 计算度矩阵

def degAT(AT):
    #print(AT)
    D=torch.sum(AT,dim=0)
    D=torch.pow(D,-0.5)
    #print(D)
    D12=torch.diag(D)
    #print(D12)   
    return D12


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
    
    eyT=torch.mean(ET.mm(W4).mm(YT))
    
   
    
    y=xyT+eyT+b2
    return y
    
    
#H2=corssNN(XT,ET,YT)


def mutH1H2(H1,H2,W5,W6,b3):
    # W5=torch.Tensor(np.random.rand(H1.shape[1],H2.shape[0]))
    # W6=torch.Tensor(np.random.rand(H2.shape[1],outSize2))
    
    #H12=torch.cat((H1.mm(W5),H2.mm(W6)),dim=0)
    H12=H1.mm(W5)+H2.mm(W6)+b3
    return H12

#H12=mutH1H2(H1,H2)




#%%
outSize1=1
outSize2=2
W1=torch.Tensor(np.random.rand(XT.shape[1],outSize1))
W2=torch.Tensor(np.random.rand(ET.shape[1],outSize1))
W3=torch.Tensor(np.random.rand(XT.shape[1],YT.shape[0]))
W4=torch.Tensor(np.random.rand(ET.shape[1],YT.shape[0]))
b1=torch.Tensor(np.random.rand(AT.shape[1],1))
b2=torch.Tensor(np.random.rand(XT.shape[0],1))
H1= gcn(AT,XT,ET,W1,W2,b1)
H2= corssNN(XT,ET,YT,W3,W4,b2)
W5=torch.Tensor(np.random.rand(H1.shape[1],outSize2))
W6=torch.Tensor(np.random.rand(H2.shape[1],outSize2))
b3=torch.Tensor(np.random.rand(AT.shape[0],1))
class Model(nn.Module):
    def __init__(self,W1,W2,W3,W4,W5,W6,b1,b2,b3):
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
        self.W2=nn.Parameter(W2)
        self.W3=nn.Parameter(W3)
        self.W4=nn.Parameter(W4)
        self.b1=nn.Parameter(b1)
        self.b2=nn.Parameter(b2)
        self.b3=nn.Parameter(b3) 
        
      
        self.W5=nn.Parameter(W5)
        self.W6=nn.Parameter(W6)
        
        self.relu=nn.LeakyReLU()
        self.tanh=nn.Tanh()
        self.linear=nn.Linear(in_features=outSize2, out_features=1)
        
    def forward(self,AT,XT,ET,YT):
        
        H1=gcn(AT,XT,ET,self.W1,self.W2,self.b1)
        H1=self.tanh(H1)
        H2=corssNN(XT,ET,YT,self.W3,self.W4,self.b2)
        H2=self.tanh(H2)
        H12=mutH1H2(H1,H2,self.W5,self.W6,self.b3)
        H12=self.linear(H12)
        H12=torch.sum(H12,dim=0)
        
        return H12

md=Model(W1,W2,W3,W4,W5,W6,b1,b2,b3)        
loss_fn = nn.MSELoss()
#loss_fu= nn.L1Loss()
optimizer = torch.optim.SGD(md.parameters(), lr=0.02)


scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
yt=yt.view(-1)
loss=0
for epoch in range(30):
# 计算预测值
    idx=0
    loss=[]
    for i in allAT:
        AT=i
        XT=allXT[idx]
        ET=allET[idx]
        YT=allYT[idx]
        yt=allyt[idx].view(-1)
        y_pred = md(AT,XT,ET,YT)
        # 计算损失
        #print(y_pred,yt)
        loss0 =loss_fn(y_pred, yt)
        
        loss.append(loss0)
       
       
        idx+=1
    # 清空梯度    
    loss1=sum(loss)/len(loss)
    optimizer.zero_grad()
    # 计算梯度
    loss1.backward()
    
    # 更新参数
    optimizer.step()
   
    #print(loss1)
    if epoch%10==0:
        print('----1-----',loss1.tolist())
        
#%%
import matplotlib.pyplot as plt
y0=np.sin(T)**2
plt.plot(y0[0:50],color='blue')
yt=y0[0:6]
y00=yt.tolist()
yt0=torch.Tensor(yt).view(-1,1)
idx=0
for i in allAT:
    
    histy=yt[1:]
    AT=i
    XT=allXT[idx]
    ET=allET[idx]
    yt=allYT[idx]
    y_pred = md(AT,XT,ET,yt).view(-1,1)
    y00.append(y_pred[0].tolist()[0])
    #yt=torch.cat((histy,y_pred[0].view(-1,1)),dim=0)
    idx+=1
    #break
    
plt.plot(y00[0:50],color='red')

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
outputs = model(YT.t())

print(outputs)


loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(30):
    
    idx=0
    loss=[]
    for i in allAT:
        AT=i
        XT=allXT[idx]
        ET=allET[idx]
        YT=allYT[idx]
        yt=allyt[idx].view(-1)
        y_pred = model(YT.t())
        # 计算损失
        #print(y_pred,yt)
        loss0 =loss_fn(y_pred, yt)
        
        loss.append(loss0)
       
       
        idx+=1
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
        if loss1<0.5:
            break
        
    
    
import matplotlib.pyplot as plt
y0=np.sin(T)**2
plt.plot(y0[0:50],color='blue')
yt=y0[0:6]
y00=yt.tolist()
yt0=torch.Tensor(yt).view(-1,1)
idx=0
for i in allAT:
    histy=yt[1:]
    AT=i
    XT=allXT[idx]
    ET=allET[idx]
    yt=allYT[idx]
    y_pred = model(yt.t()).view(-1,1)
    y00.append(y_pred[0].tolist()[0])
    #yt=torch.cat((histy,y_pred[0].view(-1,1)),dim=0)
    idx+=1
    #break
        
plt.plot(y00[0:50],color='green')








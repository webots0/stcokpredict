# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 19:14:53 2022

@author: webot
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
def NormPd(b,name):
    
    b[name]=pd.to_numeric(b[name].replace("None",''))
    if b[name].isnull().any():
        b[name]=b[name].interpolate()
    b[name]=b[name].fillna(b[name].interpolate())
  #  b[name]=(b[name] - b[name].min()) / (b[name].max() - b[name].min())
    b[name]=b[name]-b[name].mean()
    
    return b
# # 把None字符取空
# b["涨跌幅"]=pd.to_numeric(b["涨跌幅"].replace("None",''))
# # 插值带有空字符串的列转为数据
# if b["涨跌幅"].isnull().any():
#     b["涨跌幅"]=b["涨跌幅"].interpolate()

# b["涨跌幅"]=b["涨跌幅"].fillna(b["涨跌幅"].interpolate())

# 归一化
def StockCRmax_8(num):
    name=os.listdir('dataStcok')
    
    fpath=f"dataStcok\{name[num]}"
    a=pd.read_csv(fpath,encoding='gbk')
    b=a;
    b["日期"]=pd.to_datetime(b["日期"])
    s1="涨跌幅"
    s2="总市值"
    s3="收盘价"
    s4="最高价"
    s5="最低价"
    s6="开盘价"
    s7="前收盘"
    s8="涨跌额"
    s9="换手率"
    s10="成交量"
    s11="成交金额"
    s12="流通市值"
    s=[s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12]
    #pd0= pd.DataFrame()
    for i in s:
        b=NormPd(b, i)
      #  pd0=pd.concat(pd0,b,axis=1)
    b=b.fillna(0)
    k=[]
    s0=s8
    for i in s:
        
        k.append(b[s0].corr(b[i]))
    result = sorted(zip(k, range(len(k))), reverse=True)
    
    ak=result[1][0]
    if ak>0.8:
    
        #plt.clf()
        
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.plot(b["日期"],b[s0],label=s0,linewidth=0.5)
        idx=0
        cr=""
        for i,j in result:
            if (idx>0) and (idx<2):
                sj=s[j]
                cr=str(i)
                
                plt.plot(b["日期"],b[sj],label=sj,linewidth=0.5)
                if idx==2:
                    break
            idx=idx+1
           
        
        #ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
        plt.legend()
        
        plt.title(b["名称"][0]+f'相关系数={cr}')
        plt.show()
        return (b["名称"][0],cr,sj,b)
    else:
        return None
NameList=[]
for i in range(10):
    NameList.append(StockCRmax_8(i))
    NameList = list(filter(lambda x: x is not None, NameList))
    



#%%
def getData(num):
    data_all=[]
    
    for j in range(len(NameList)):
        data1=NameList[j][-1]
        
        df = data1.iloc[:, (-15+3):]
        
        a=df.values
        data=torch.from_numpy(a).to(torch.float32)
        a1=data[num:(num+7),0:6]
        data_all.append(a1)
        if j==1:
            Y=data[num+7:num+14,7]
    return (data_all,Y)

import torch
from testMd import Model


(da,y)=getData(0)

md=Model(da[0],da[1],da[2],da[3],da[4],K=36,L1=28,L2=14,out=7)
yp=md(da[0],da[1],da[2],da[3],da[4])

from torch import nn
#loss_fn = nn.NLLLoss()
loss_fn= nn.MSELoss()
optimizer = torch.optim.SGD(md.parameters(), lr=0.1)
#%%

for epoch in range(3000):
    loss=[]
    Y_pred=torch.empty(720,7)
    Yt=torch.empty(720,7)
    for j in range(720):
        (da,yt)=getData(j)
        y_pred=md(da[0],da[1],da[2],da[3],da[4])
        Y_pred[j,:]=y_pred
        Yt[j,:]=yt
        
        
        
      
        # 计算损失
        #print(y_pred,yt)
    loss0 =loss_fn(Y_pred, Yt)
        
    #loss.append(loss0)
   
    optimizer.zero_grad()
    
    
    # 计算梯度
   # loss1=sum(loss)/len(loss)
    loss0.backward()
    
    optimizer.step()
        # print('---',loss0.tolist())
        # break
   
    #print(loss1)
    if (epoch+1)%2==0:
      
        print(f'----{epoch+1}-----',loss0.tolist())
        (da,yt)=getData(76)
        y_p=md(da[0],da[1],da[2],da[3],da[4])
       
        print(y_p,yt)
      

        
#%%

import matplotlib.pyplot as plt
import numpy as np
# data to plot
(da,y)=getData(800)
yp=md(da[0],da[1],da[2],da[3],da[4])
n_groups = 7
means_frank = tuple(y.tolist())
means_guido = tuple(yp.tolist())

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,
alpha=opacity,
color='b',
label='真实')

rects2 = plt.bar(index + bar_width, means_guido, bar_width,
alpha=opacity,
color='g',
label='预测')

plt.xlabel('Person')
plt.ylabel('Scores')
plt.title('Scores by person')
plt.xticks(index + bar_width, ('1', '2', '3', '4','5','6','7'))
plt.legend()

plt.tight_layout()
plt.show()

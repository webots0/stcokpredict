# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 19:16:45 2023

@author: webot
"""
# 每天下午两点开始运行代码，把选择的股票保存在txt文件中


#%% 获取当天股票

import pandas as pd
from pytdx.hq import TdxHq_API
import matplotlib.pyplot as plt
from pytdx.params import TDXParams
import time
import math

api = TdxHq_API()  
import numpy as np
def normalize1(x):
    return (x - x.min()) / (x.max() - x.min())
def normalize2(x):
    return (x - x.mean()) / x.std()

start=0
data_R0=[]

code_df=np.load("code.npy",allow_pickle=True)

if api.connect('106.14.201.131', 7709):
    print('连接成功')
else:
    raise ValueError('连接失败')

end = 10
close_df=[]
code0=[]
idx=0

for code in code_df:
    try:
        data = api.to_df(api.get_security_bars(9,0,code,start,end)) #
        
        open_r=data["open"]
        close_r=data["close"]
        high_r=data["high"]
        low_r=data["low"]
        vloum=data["vol"]
        data_rr=pd.concat([open_r,close_r,low_r,high_r,vloum],axis=1)
        data_r=data_rr.apply(normalize2)
        #cr=corrAB(data_r, C0)
        #data_r0=data_rr.iloc[-1,:]
        #print(data_r)
        data_R0.append(data_r)
       
        #break
    except:
        
        pass
    
    if idx%100==0:
        print(idx,len(data_R0))
        #break
    idx+=1


#%% 80选择
def corrAB(A,B):
    A=A.apply(normalize2)
    B=B.apply(normalize2)
    x1=np.array(A)
    x2=np.array(B)
    
    x1_=np.mean(x1)
    x2_=np.mean(x2)
    x3=np.sum((x1-x1_)*(x2-x2_))
    x4=np.sum((x1-x1_)**2)*np.sum((x2-x2_)**2)
    x34=x3/np.sqrt(x4)
    return x34
cc=np.load("聚类中心_0.npy")
Code_0=[]
Code=[]
for c0 in range(len(cc)):
    print("当前所属的类别是",c0,"类")
    num=c0
    C0=cc[c0,:]
    C0=C0.reshape(10,5)
    C0=pd.DataFrame(C0)
    idx=0
    
    for r0 in data_R0:
        if len(r0)==10:
            cd=code_df[idx]
            cr=corrAB(r0,C0)
            
            if cr>0.8:
                Code.append(cd)
            idx+=1
        
          
    print("选择股票如下：",Code)
    
#% 80 选择20组

import collections

# Count the number of occurrences of each element in the list
counter = collections.Counter(Code)

# Sort the elements by their frequency
sorted_list = sorted(counter.items(), key=lambda x: x[1], reverse=True)

print(sorted_list)
cd=np.array(sorted_list)
cd20=cd[:20,0]
cd_20=cd[-20:,0]
id0=round(len(cd)/2)
cd_z_20=cd[id0:id0+20,0]
np.savetxt('sortCode\\000.txt', cd[:,0], fmt='%s')
np.savetxt('sortCode\\前_20.txt', cd20, fmt='%s')
np.savetxt('sortCode\\后_20.txt', cd_20, fmt='%s')
np.savetxt('sortCode\\中间_20.txt', cd_z_20, fmt='%s')


#%% 95 选择
def corrAB(A,B):
    A=A.apply(normalize2)
    B=B.apply(normalize2)
    x1=np.array(A)
    x2=np.array(B)
    
    x1_=np.mean(x1)
    x2_=np.mean(x2)
    x3=np.sum((x1-x1_)*(x2-x2_))
    x4=np.sum((x1-x1_)**2)*np.sum((x2-x2_)**2)
    x34=x3/np.sqrt(x4)
    return x34
cc=np.load("聚类中心_0.npy")
#code=code_df["code"]
Code_0=[]
Code=[]
for c0 in range(len(cc)):
    print("当前所属的类别是",c0,"类")
    num=c0
    C0=cc[c0,:]
    C0=C0.reshape(10,5)
    C0=pd.DataFrame(C0)
    idx=0
    
    for r0 in data_R0:
        if len(r0)==10:
            cd=code_df[idx]
            cr=corrAB(r0,C0)
            
            if cr>0.95:
                Code.append(cd)
            idx+=1
        
          
    print("选择股票如下：",Code)
# 95 选择10组



import collections

# Count the number of occurrences of each element in the list
counter = collections.Counter(Code)

# Sort the elements by their frequency
sorted_list = sorted(counter.items(), key=lambda x: x[1], reverse=True)

print(sorted_list)
cd=np.array(sorted_list)
cd20=cd[:10,0]
cd_20=cd[-10:,0]
id0=round(len(cd)/2)
cd_z_20=cd[id0:id0+10,0]
np.savetxt('sortCode\\000_10.txt', cd[:,0], fmt='%s')
np.savetxt('sortCode\\前_10.txt', cd20, fmt='%s')
np.savetxt('sortCode\\后_10.txt', cd_20, fmt='%s')
np.savetxt('sortCode\\中间_10.txt', cd_z_20, fmt='%s')
    


#%% 单股测试

start=0;
end=1



if api.connect('106.14.201.131', 7709):
    print('连接成功')
else:
    raise ValueError('连接失败')
    
data = api.to_df(api.get_security_bars(9,0,'002161',start,end)) #
            
#%% 初始资金1万，一年46万，两年2千多万，只需要一天两个点
days=200;
import random
k0=np.arange(0, 1.01, 0.01)
Ka=[]
for r0 in k0:
    a0=0
    for num in range(100):
        
        a1=10000
        b=[]
        
        for i in range(days):
            r=random.uniform(0, 1)
            if r>r0:
                k=-1
            else:
                k=1
            a1=a1*(1+k*0.02)
            a1=a1-a1*0.0001-a1*0.00025*2-a1*0.00001
            b.append(a1)
        a0=a0+a1
    a1=a0/100
    Ka.append(a1)
      
      
import matplotlib.pyplot as plt
plt.plot(k0,Ka)
plt.show()



# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 01:19:35 2023

@author: webot
"""


import pandas as pd
from pytdx.hq import TdxHq_API
import matplotlib.pyplot as plt
from pytdx.params import TDXParams
import time
import math

api = TdxHq_API()  
code_dff=[]
code_location1=(0,1000,1000)
code_location2=(0,2000,1)
code_location3=(0,3000,1)
code_location4=(0,4000,1)
code_locationL=[code_location1,code_location2,code_location3,code_location4]
code_location=code_location1
#for code_location in code_locationL:
if api.connect('106.14.201.131', 7709):
            print('连接成功')
else:
    raise ValueError('连接失败')
code_df=api.to_df(api.get_security_list(code_location[0],code_location[1]))
code_dff.append(code_df["code"].tail(code_location[2]).to_frame())
code_df=pd.concat(code_dff,axis=0)
code_df=code_df.reset_index(drop=True)

#%%
start=0
data_R=[]

for start in range(250):

    if api.connect('106.14.201.131', 7709):
        print('连接成功')
    else:
        raise ValueError('连接失败')
    
    end = 11
    close_df=[]
    code0=[]
    idx=0
    
    for code in code_df["code"]:
        try:
            data = api.to_df(api.get_security_bars(9,0,code,start,end)) #
            rc=data["close"].iloc[-1]
            ro=data["open"].iloc[-1]
            r=(rc-ro)/ro
            if r>0.08:
                open_r=data["open"]
                close_r=data["close"]
                high_r=data["high"]
                low_r=data["low"]
                vloum=data["vol"]
                data_r=pd.concat([open_r,close_r,low_r,high_r,vloum],axis=1)
                data_r=data_r.iloc[:-1,:]
                print(r)
                data_R.append(data_r)
           
                #break
        except:
            
            pass
        
        if idx%100==0:
            print(idx)
        idx+=1
#%% 数据的归一化
import numpy as np
def normalize1(x):
    return (x - x.min()) / (x.max() - x.min())
def normalize2(x):
    return (x - x.mean()) / x.std()

D = np.empty((0,50))
for dr in data_R:
    d0=dr.apply(normalize2)
    d1=d0.to_numpy().reshape(50)
    D=np.concatenate((D,d1[np.newaxis,:]),axis=0)
    
#%%
D1=np.empty((0,50))
for dx in D:

    a1=np.isnan(dx)
    a2=np.alltrue(~a1)
    print(a2)
    if a2:
       print(a2)
        
       D1=np.concatenate((D1,dx[np.newaxis,:]),axis=0)
    
        
        
   
      
#%% Kmeans进行聚类
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
kmeans.fit(D1)
cluster_centers = kmeans.cluster_centers_
labels = kmeans.predict(D1)
print("kmeans 分类数为5")
#%% 传递信息聚类
from sklearn.cluster import AffinityPropagation

clustering = AffinityPropagation(random_state=1).fit(D1)
cluster_centers = clustering.cluster_centers_
labels=clustering.fit_predict(D1)
numC=max(labels)+1
print("自动分开为",numC,"类")
#%%
np.save("聚类中心_0",cluster_centers)
#%%
cc=np.load("聚类中心_0.npy")
#%% 将原始数据进行按照聚类模型的标签进行分类
idx=0
L=[]
for i in range(numC):
    L.append([])
    
   
for i in data_R:
    la=labels[idx]
    L[la].append(i)
    
    idx+=1   
        

#%% 检验聚类的相关性
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

cr=0
L0=L[1]
for l in L0:    
    cr=cr+corrAB(L0[0],l)
cr0=cr/len(L0)
print(cr0)

#%%  

# 画出某一类的k线图
c0=16
print("当前所属的类别是",c0,"类")
num=c0
C0=cluster_centers[c0,:]
C0=C0.reshape(10,5)
C0=pd.DataFrame(C0)
sel=np.random.randint(0,len(L[num]),6)
print("随机选择的组数",sel)
l0=C0.apply(normalize1).add(12)
l1=L[num][sel[0]].apply(normalize1).add(0)
l2=L[num][sel[1]].apply(normalize1).add(2)
l3=L[num][sel[2]].apply(normalize1).add(4)
l4=L[num][sel[3]].apply(normalize1).add(6)
l5=L[num][sel[4]].apply(normalize1).add(8)
l6=L[num][sel[5]].apply(normalize1).add(10)
import pyecharts.options as opts
from pyecharts.charts import Candlestick
y0_data=[]
y1_data=[]
y2_data=[]
y3_data=[]
y4_data=[]
y5_data=[]
y6_data=[]
idx=0
for i in l1.iloc:
    l0_=l0.iloc[idx,:]
    l0_=l0_.tolist()[:-1]
    l1_=l1.iloc[idx,:]
    l1_=l1_.tolist()[:-1]
    l2_=l2.iloc[idx,:]
    l2_=l2_.tolist()[:-1]
    l3_=l3.iloc[idx,:]
    l3_=l3_.tolist()[:-1]
    l4_=l4.iloc[idx,:]
    l4_=l4_.tolist()[:-1]
    l5_=l5.iloc[idx,:]
    l5_=l5_.tolist()[:-1]
    l6_=l6.iloc[idx,:]
    l6_=l6_.tolist()[:-1]
    
    
    
    
    y0_data.append(l0_)
    y1_data.append(l1_)
    y2_data.append(l2_)
    y3_data.append(l3_)
    y4_data.append(l4_)
    y5_data.append(l5_)
    y6_data.append(l6_)
    idx+=1
    

x_data = ["1", "2", "3", "4","5","6","7","8","9","10"]
(
    Candlestick(init_opts=opts.InitOpts(width="1200px", height="1200px"))
    .add_xaxis(xaxis_data=x_data)
    .add_yaxis(series_name="",y_axis=y1_data)
    .add_yaxis(series_name="",y_axis=y2_data)
    .add_yaxis(series_name="",y_axis=y3_data)
    .add_yaxis(series_name="",y_axis=y4_data)
    .add_yaxis(series_name="",y_axis=y5_data)
    .add_yaxis(series_name="",y_axis=y6_data)
    .add_yaxis(series_name="",y_axis=y0_data)
    .set_series_opts()
    .set_global_opts(
        yaxis_opts=opts.AxisOpts(
            splitline_opts=opts.SplitLineOpts(
                is_show=True, linestyle_opts=opts.LineStyleOpts(width=2)
            )
        )
    )
    .render("简单K线图.html")
)

#%% 历史数据测试模型
c0=9
print("当前所属的类别是",c0,"类")
num=c0
C0=cluster_centers[c0,:]
C0=C0.reshape(10,5)
C0=pd.DataFrame(C0)




start=40
data_R0=[]



if api.connect('106.14.201.131', 7709):
    print('连接成功')
else:
    raise ValueError('连接失败')

end = 11
close_df=[]
code0=[]
idx=0

for code in code_df["code"]:
    try:
        data = api.to_df(api.get_security_bars(9,0,code,start,end)) #
        
        open_r=data["open"]
        close_r=data["close"]
        high_r=data["high"]
        low_r=data["low"]
        vloum=data["vol"]
        data_rr=pd.concat([open_r,close_r,low_r,high_r,vloum],axis=1)
        data_r=data_rr.iloc[:-1,:].apply(normalize2)
        cr=corrAB(data_r, C0)
        #data_r0=data_rr.iloc[-1,:]
        if cr>0.8:
            print(cr)
            rc=data["close"].iloc[-1]
            ro=data["open"].iloc[-1]
            r=(rc-ro)/ro
            
            
            data_R0.append(r)
       
        #break
    except:
        
        pass
    
    if idx%100==0:
        print(idx)
    idx+=1


#%% 当天模型测试

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
        break
    idx+=1

#%% 保存选择的股票数据

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
    C0=cluster_centers[c0,:]
    C0=C0.reshape(10,5)
    C0=pd.DataFrame(C0)
    idx=0
    
    for r0 in data_R0:
        if len(r0)==10:
            cd=code[idx]
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
code=code_df["code"]
Code_0=[]
Code=[]
for c0 in range(len(cc)):
    print("当前所属的类别是",c0,"类")
    num=c0
    C0=cluster_centers[c0,:]
    C0=C0.reshape(10,5)
    C0=pd.DataFrame(C0)
    idx=0
    
    for r0 in data_R0:
        if len(r0)==10:
            cd=code[idx]
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
            

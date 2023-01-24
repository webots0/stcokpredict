# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 04:06:54 2023

@author: webot
"""
from datetime import datetime

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


data_R0=[]

code_df=np.load("code.npy",allow_pickle=True)

if api.connect('106.14.201.131', 7709):
    print('连接成功')
else:
    raise ValueError('连接失败')
#%%
upR=[]
downR=[]
idx=0
for start in np.random.randint(0,300,100):
    #start=0
    end = 2
    
    for code in code_df:
        try:
            data = api.to_df(api.get_security_bars(9,0,code,start,end)) #
            close_r=data["close"]
            r0=(close_r[1]-close_r[0])/close_r[0]
            if r0>0.01:
                year=str(data["year"].iloc[0])
                month=data["month"].iloc[0]
                day=data["day"].iloc[0]
                
                if month/10<1:
                   month="0"+str(month)
                else:
                   month=str(month)
                if day/10<1:
                    day="0"+str(day)
                else:
                    day=str(day)
                dy=year+month+day
                dy=int(dy)
                da=api.to_df(api.get_history_minute_time_data(0, code, dy))
                da=list(da["price"])
                upR.append(da)
                
                idx+=1
                print(idx)
                if idx>2000:
                   datar=np.array(upR)
                   datar0=np.array(downR)
                   np.savetxt("dayPrice\\upprice.csv",datar)
                   np.savetxt("dayPrice\\downprice.csv",datar0)
                   
                   break
            else:
               da=api.to_df(api.get_history_minute_time_data(0, code, dy))
               da=list(da["price"])
               downR.append(da)
               
                   
                
        
            
        except:
            
            pass
    if idx>2000:
       break
        
  
#%%

import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6.3])

#归一化
a = (a-np.mean(a)) 
b = (b-np.mean(b)) 
a=a/np.linalg.norm(a)
b=b/np.linalg.norm(b)

#计算平方欧式距离
distance = np.linalg.norm(a-b)**2
print(distance)
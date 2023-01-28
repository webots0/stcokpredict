# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 10:19:21 2023

@author: webot
"""
# 代码每日收盘前运行

# r0=0代表当天执行，r0=1代表明天执行
r0=0










import os

import sys
import pandas as pd
from pytdx.hq import TdxHq_API
import matplotlib.pyplot as plt
from pytdx.params import TDXParams
import time
import math
import matplotlib.pyplot as plt
import datetime
import numpy as np
today = datetime.datetime.now().date()
api = TdxHq_API()  
import numpy as np
from matplotlib.widgets import Button
# [0到13][0,1,(2,3,4,5,6,7,8,9,10,11),12,13]


f=os.listdir("每日分析股票")
Code=[]
Numday=[]
for f0 in f:
    a0=np.loadtxt("每日分析股票\\"+f0,dtype=str).tolist()
    if not isinstance(a0,list):
        a0=[a0]
    
    if len(a0)>0:
        Code.append((int(f0[:-4]),a0))
    
   

    
#%%        
    
def plot0(ax3,pd,numday):
    print('--------------')
    open = pd["open"]
    high = pd["high"]
    low = pd["low"]
    close = pd["close"]
    
    date = np.linspace(1,len(low),len(low))
    # 绘制K线图
   # fig, ax = plt.subplots()
    
    t0=len(date)-(15-numday)
    for i in range(len(date)):
        if close[i] > open[i]:
            color = 'red'
        else:
            color = 'green'
        if i==t0:
            color='blue'
            
            
        
        
        ax3.plot([date[i], date[i]], [low[i], high[i]], color=color)
        ax3.plot([date[i] , date[i]], [open[i], close[i]], color=color, linewidth=10)
    ax3.set_xlim(0,40+1)
    #ax4.plot(lishi)
def plotK(pd,qushi,c0,lishi,weilai,numday):
    
    open = pd["open"]
    high = pd["high"]
    low = pd["low"]
    close = pd["close"]
    
    date = np.linspace(1,len(low),len(low))
    # 绘制K线图
   # fig, ax = plt.subplots()
     
    fig, ((ax, ax3), (ax2, ax4)) = plt.subplots(2, 2)
    #ax.xaxis_date()
   
    t0=len(date)-(15-numday)
    for i in range(len(date)):
        if close[i] > open[i]:
            color = 'red'
        else:
            color = 'green'
        # if i==t0:
        #     color='blue'
            
            
        
        
        ax.plot([date[i], date[i]], [low[i], high[i]], color=color)
        ax.plot([date[i] , date[i]], [open[i], close[i]], color=color, linewidth=10)
        
        if i==t0:
            break
    ax.set_xlim(0,len(low)+1)
    ax.set_title(f"{c0}")
    ax2.plot(qushi)
    ax4.plot(lishi)
    #f1.add_axes([0.5, 0.9, 0.1, 0.04])
    
    bplot = Button(fig.add_axes([0.5, 0.9, 0.1, 0.04]), f'predict{weilai}day')
    bplot.on_clicked(lambda event: plot0(ax3,pd,numday))
    
   
    return bplot


#%%

aa=[]
for numday_c0 in Code:
    c_0=numday_c0[1]
    numday=numday_c0[0]-r0
    start=16-numday
    weilai=15-numday
    for c0 in c_0:
        
        api = TdxHq_API()  
        if api.connect('106.14.201.131', 7709):
            print('连接成功')
        else:
            raise ValueError('连接失败')
        
        pd=api.to_df(api.get_security_bars(9,0,c0,0,40-numday)) #
        d2=api.to_df(api.get_security_bars(9,0,c0,start,1)) #
        date=d2['datetime'][0]
        date_int=int(date[0:4]+date[5:7]+date[8:10])
        
        d1=api.to_df(api.get_history_minute_time_data(0,c0,date_int))
        qushi=np.array(d1["price"])
        lishi=api.to_df(api.get_security_bars(9,0,c0,0,100));
        lishi=np.array(lishi["close"])
        aa.append(plotK(pd,qushi,c0,lishi,weilai,numday))

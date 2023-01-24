# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 01:34:31 2023

@author: webot
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:36:03 2023

@author: webot
"""

print('-----开始趋势匹配---------')
from datetime import datetime

import pandas as pd
from pytdx.hq import TdxHq_API
import matplotlib.pyplot as plt
from pytdx.params import TDXParams
import time
import math
from datetime import datetime
import numpy as np
api = TdxHq_API()  


if api.connect('106.14.201.131', 7709):
    print('连接成功')
else:
    raise ValueError('连接失败')



# 将0，6，3开头的股票代码导入进来
'''
code_df=np.load("code.npy",allow_pickle=True)
code=code_df
code0=api.get_security_list(1, 0)
num=api.get_security_count(0)
x1 = np.array(api.to_df(api.get_security_list(0, 364))["code"])
x2= np.array(api.to_df(api.get_security_list(0, 364+1000))["code"])[0:504]

x12=np.concatenate((x1,x2))

y1=np.array(api.to_df(api.get_security_list(1,18030))["code"])
y2=np.array(api.to_df(api.get_security_list(1,18030+1000))["code"])
y3=np.array(api.to_df(api.get_security_list(1,18030+2000))["code"])[0:172]

y123=np.concatenate((y1,y2,y3))
Y123=[]
for y0 in y123:
    Y123.append([1,y0])
    

z1=np.array(api.to_df(api.get_security_list(0,14739))["code"])
x123=np.concatenate((x12,z1))
X123=[]
for x0 in x123:
    X123.append([0,x0])
    

xy123=X123+Y123    

xy_123=np.array(xy123)

for ij in xy_123:
    print(ij)
    break
np.save("allCode.npy",x12)
'''
#
# 求出涨幅大于0.05的code
code=np.load("allCode.npy",allow_pickle=True)

def get002code(start):
    api = TdxHq_API() 
    if api.connect('106.14.201.131', 7709):
        print('连接成功')
    Code=[]
    idx=0
    for cd in code:
        
        cd0=str(cd)
        try:
            data = api.to_df(api.get_security_bars(9,0,cd0,start,2)) #
            date=data['datetime'][1]
            date_int=int(date[0:4]+date[5:7]+date[8:10])
            close=data["close"]
            open0=data["open"]
            y=close[0]
            t=open0[1]
            ro=(t-y)/y
            
            if ro>=0.02:
                
                if close[1]>open0[1]:
                    Code.append(cd0)
        except:
            pass
        idx+=1
        if idx%100==0:
            print(idx)
            
    return (Code,date_int)
        



C0=np.loadtxt("dayPrice\\标准增长.csv")



def norm(a):
    a = (a-np.mean(a)) 
   
    a=a/np.linalg.norm(a)
    return a
def cor(a,b):
    a = (a-np.mean(a)) 
    b=b[0:len(a)]
    b = (b-np.mean(b)) 
    a=a/np.linalg.norm(a)
    b=b/np.linalg.norm(b)

    #计算平方欧式距离
    distance = np.linalg.norm(a-b)**2
    return distance

def panduanTing(pa):
    r=pa[-50:]
    if sum(abs(r-r.mean()))>(r.mean()/200):
        return True
    else:
        return False
    
def getQushiCode(Code,date_int):
    
    api = TdxHq_API() 
    if api.connect('106.14.201.131', 7709):
        print('连接成功')
    Ds=[]
    for c0 in Code:
        try:
            
            da=api.to_df(api.get_history_minute_time_data(0,c0,date_int))
            pa=np.array(da['price'])
            flag=panduanTing(pa)
            if flag:
            
                ds=cor(pa,C0)
                
                Ds.append([c0,ds])
            #print(ds)
            
        except:
            pass
        #break
    
    
    Dsp=np.array(Ds)    
    ad=Dsp[:,1].astype(np.float)
    idx0=~np.isnan(ad)
    Dsp0=Dsp[idx0,:]
    ad=Dsp0[:,1].astype(np.float)
    sorted_indices = np.argsort(ad)
    sorted_matrix = Dsp0[sorted_indices]
    
    selCd=sorted_matrix
    return selCd



    


def getMinCode(selCd,r0):
    
    minCode=[]
    for c0 in selCd:
        r=c0[1]
        if float(r)<r0:
            minCode.append(str(c0[0]))
            
    return minCode


def plotK(pd,qushi):
    
    open = pd["open"]
    high = pd["high"]
    low = pd["low"]
    close = pd["close"]
    
    date = np.linspace(1,len(low),len(low))
    # 绘制K线图
    #fig, ax = plt.subplots()
    fig, (ax, ax2) = plt.subplots(2, 1)
    ax.xaxis_date()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('K Line')
    #t0=len(date)-15
    for i in range(len(date)):
        if close[i] > open[i]:
            color = 'red'
        else:
            color = 'green'
        # if i==t0-2:
        #     color='yellow'
        ax.plot([date[i], date[i]], [low[i], high[i]], color=color)
        ax.plot([date[i] , date[i]], [open[i], close[i]], color=color, linewidth=10)
    ax2.plot(qushi)
    
    plt.show()

#%%
#Code,date_int=get002code(4)

#%%
#selCd=getQushiCode(Code,date_int)
#minCode=getMinCode(selCd,0.1)



#%%
import threading
import queue

      
def threadGet(i,cd_time):
    try:
   
        Code,date_int=get002code(i)
        selCd=getQushiCode(Code,date_int)
        #minCode=getMinCode(selCd,0.1)
        minCode=selCd
        a=[minCode,date_int,i]
        cd_time.put(a)
    except:
        pass
    
    
    
cd_time=queue.Queue()
threads=[]

selnum=[0]
for i in selnum:
    
    t = threading.Thread(target=threadGet, args=(i, cd_time))
    
    threads.append(t)
    t.start()

for t in threads:
    t.join()
newD=[]
while not cd_time.empty():
    newD.append(cd_time.get())

    
    
#%%
a=()
for i in newD:
    i1=i[0]
    i2=i[1]
    i3=i[2]
    i20=matrix = np.full((len(i1),), str(i2), dtype=np.dtype(object))
    i30=matrix = np.full((len(i1),), str(i3), dtype=np.dtype(object))
    i0 = np.column_stack((i1, i20,i30))
    a=a+(i0,)
newD0=np.concatenate(a)
    

    

from datetime import timedelta,datetime
import matplotlib.pyplot as plt
api = TdxHq_API()  
if api.connect('106.14.201.131', 7709):
    print('连接成功')
else:
    raise ValueError('连接失败')

a=()
for i in newD:
    i1=i[0]
    i2=i[1]
    i3=i[2]
    i20=matrix = np.full((len(i1),), str(i2), dtype=np.dtype(object))
    i30=matrix = np.full((len(i1),), str(i3), dtype=np.dtype(object))
    i0 = np.column_stack((i1, i20,i30))
    a=a+(i0,)
newD0=np.concatenate(a)
    
outCode=[]

c00=norm(C0)
#plt.plot(c00) 
idx=0   
for i in newD0:
    num=int(i[3])
    c0=str(i[0])
    r0=float(i[1])
    
    
    
    d2=api.to_df(api.get_security_bars(9,0,c0,num,1)) #
    close=d2["close"]
    date=d2['datetime'][0]
    date_int=int(date[0:4]+date[5:7]+date[8:10])
    
    
    d1=api.to_df(api.get_history_minute_time_data(0,c0,date_int))
    qushi=np.array(d1["price"])
    
    
    if r0<0.15:
       
        idx+=1
        qus=qushi
        # qushi=norm(qushi)
        # c_0=close[0]
        # c_1=close[1]
        # r_c=(c_1-c_0)/c_0*100;
        # r_c=round(r_c,2)
        # print(f"股票{c0}相似距离是:{r0},第二天股票涨幅是:{r_c}%")
        d2=api.to_df(api.get_security_bars(9,0,c0,0,25)) #
        
        plotK(d2,qus)
        outCode.append(c0)
      
      
        #close=norm(qushi)
        # plt.plot(qushi,label=str(round(r0,3)))
        # plt.show()
        
        if idx==8:
            Qushi=qus
            #break
            
        #print(list(close),r0)
    #break
    

    
# #%%
# import matplotlib.pyplot as plt
# C00=norm(C0)
# C1=C0+np.random.normal(0, 1, size=(240,))*0.1
# C01=norm(C1)
# plt.plot(C00)
# plt.plot(C01)
# dt=cor(C00,C01)
# print(dt)


#%%
#plt.plot(Qushi)
a=np.array(outCode)
np.savetxt('sortCode\\严格趋势.txt',a, fmt='%s')

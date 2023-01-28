# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 01:51:41 2023

@author: webot
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 00:51:47 2023

@author: webot
"""

# -*- coding: utf-8 -*-

# 追涨生成股票

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:10:31 2023

@author: webot
"""



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
numday=11
start=16-numday
#%% 追跌
print('-------三跌一阳涨-------')

def caculate_today(start):
    all_flag=0
    true_flag=0    
    #start=0
    end=10
    
    Code=[]
    for code in code_df:
        try:
            data = api.to_df(api.get_security_bars(9,0,code,start,end)) #
            #data=data0.iloc[:-1,:]
            
            end_price=data.iloc[-1,:]['close']
            start_price=data.iloc[0:-1,:]['close']
            low_true=end_price<start_price
            all_low_true=low_true.all()
            # 如果是你10天内最低价
            if True:
                
                # 判断前三天都在跌
                d1=data.iloc[-2,:]
                true1=d1['open']>d1['close']
                
                if true1:
                    
                    d2=data.iloc[-3,:]
                    true2=d2['open']>d2['close']
                    
                    if true2:
                        d3=data.iloc[-4,:]
                        true3=d3['open']>d3['close']
                        
                        if true3:
                            
                            d4=data.iloc[-5,:]
                            true4=d4['open']>d4['close']
                            
                            
                            if True:
                                # 判断完前四天都在跌
                                #print('前三天都在跌')
                                # 判断当天在充能量
                                # 求当天的时间
                                true5=d3['close']>d2['close']
                                true6=d2['close']>d1['close']
                                if true5 and true6:
                                    
                                    # 当天低开高走
                                    d0=data.iloc[-1,:]
                                    true7=d0['close']>d0['open']
                                    
                                    if true7:
                                        
                                       true8=d1['close']<d0['close']
                                       if true8:
                                       
                                           Code.append(code)
                                 
                            # 如果能量大于0.02,预判明天会涨
                            # if E<-0.003:
                                
                            #     all_flag+=1
                            #     # 统计明天是否会涨
                            #     today=data.iloc[-1,:]
                            #     tomor=data0.iloc[-1,:]
                            #     price_today=today['close']
                            #     price_tomor=tomor['close']
                            #     flag=(price_tomor-price_today)/price_today
                            #     print(f'=========能量满足要求,能量是{E},涨幅是{flag}')
                            #     # 如果涨幅大于0.01
                            #     if flag>0.01:
                            #         print(f'----------涨幅满足要求,涨幅是{flag}')
                            #         true_flag+=1
                                
                                
                            
                                
                                
                                
        except:
                    
            pass
                        
                        
                        
                        
                        
                        
            
            
            
        
        
       
    
    # print(all_flag)
    # print(true_flag)    
    return Code
# 计算昨天三连跌一涨的股票
api = TdxHq_API()  


code_df=np.load("code.npy",allow_pickle=True)

if api.connect('106.14.201.131', 7709):
    print('连接成功')
else:
    raise ValueError('连接失败')
        
Code=caculate_today(start)  
Code_1=np.array(Code)



#%% 追跌，三连跌，两连涨


print('三跌两阳涨')
    

def caculate_today0(start):
    all_flag=0
    true_flag=0    
   # start=0
    end=10
    
    Code=[]
    for code in code_df:
        try:
            data = api.to_df(api.get_security_bars(9,0,code,start,end)) #
            #data=data0.iloc[:-1,:]
            
            end_price=data.iloc[-1,:]['close']
            start_price=data.iloc[0:-1,:]['close']
            low_true=end_price<start_price
            all_low_true=low_true.all()
            # 如果是你10天内最低价
            if True:
                
                # 判断前三天都在跌
                d1=data.iloc[-3,:]
                true1=d1['open']>d1['close']
                
                if true1:
                    
                    d2=data.iloc[-4,:]
                    true2=d2['open']>d2['close']
                    
                    if true2:
                        d3=data.iloc[-5,:]
                        true3=d3['open']>d3['close']
                        
                        if true3:
                            
                            d4=data.iloc[-6,:]
                            true4=d4['open']>d4['close']
                            
                            
                            if True:
                                # 判断完前四天都在跌
                                #print('前三天都在跌')
                                # 判断当天在充能量
                                # 求当天的时间
                                true5=d3['close']>d2['close']
                                true6=d2['close']>d1['close']
                                if true5 and true6:
                                    
                                    # 昨天天低开高走
                                    d01=data.iloc[-2,:]
                                    true7=d01['close']>d01['open']
                                    
                                    if true7:
                                        
                                       true8=d1['close']<d01['close']
                                       if true8:
                                           # 当天低开高走
                                           d02=data.iloc[-1,:]
                                           true9=d02['close']>d02['open']
                                           if true9:
                                               
                                               # 当天比昨天高
                                               true10=d02['close']>d01['close']
                                               if true10:
                                                  Code.append(code)
                                 
                            # 如果能量大于0.02,预判明天会涨
                            # if E<-0.003:
                                
                            #     all_flag+=1
                            #     # 统计明天是否会涨
                            #     today=data.iloc[-1,:]
                            #     tomor=data0.iloc[-1,:]
                            #     price_today=today['close']
                            #     price_tomor=tomor['close']
                            #     flag=(price_tomor-price_today)/price_today
                            #     print(f'=========能量满足要求,能量是{E},涨幅是{flag}')
                            #     # 如果涨幅大于0.01
                            #     if flag>0.01:
                            #         print(f'----------涨幅满足要求,涨幅是{flag}')
                            #         true_flag+=1
                                
                                
                            
                                
                                
                                
        except:
                    
            pass
                        
                        
                        
                        
                        
                        
            
            
            
        
        
       
    
    # print(all_flag)
    # print(true_flag)    
    return Code
# 
api = TdxHq_API()  


code_df=np.load("code.npy",allow_pickle=True)

if api.connect('106.14.201.131', 7709):
    print('连接成功')
else:
    raise ValueError('连接失败')

Code=caculate_today0(start)

Code_2=np.array(Code)



#%% 80选择,追涨
print('---------80追涨-------------')
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

cd=np.array(sorted_list)
cd20=cd[:20,0]
cd_20=cd[-20:,0]
id0=round(len(cd)/2)
cd_z_20=cd[id0:id0+20,0]
Code_3=cd[:,0]

#%%









#%%

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
code=list(set(list(np.concatenate((Code_1,Code_2,Code_3)))))

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
            
            if ro>=0.005:
                
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


def plotK0(pd,qushi):
    
    open = pd["open"]
    high = pd["high"]
    low = pd["low"]
    close = pd["close"]
    
    date = np.linspace(1,len(low),len(low))
    # 绘制K线图
    fig, (ax,ax2) = plt.subplots(2,1)
    ax.xaxis_date()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('K Line')
    t0=len(date)-15
    for i in range(len(date)):
        if close[i] > open[i]:
            color = 'red'
        else:
            color = 'green'
        if i==t0-2:
            color='blue'
        ax.plot([date[i], date[i]], [low[i], high[i]], color=color)
        ax.plot([date[i] , date[i]], [open[i], close[i]], color=color, linewidth=10)
    
    plt.show()



def plot0(ax3,pd):
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
def plotK(pd,qushi,c0,lishi):
    
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
    
    bplot = Button(fig.add_axes([0.5, 0.9, 0.1, 0.04]), 'Click!')
    bplot.on_clicked(lambda event: plot0(ax3,pd))
    
   
    return bplot
    
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

selnum=[start]
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
    i20 = np.full((len(i1),), str(i2), dtype=np.dtype(object))
    i30 = np.full((len(i1),), str(i3), dtype=np.dtype(object))
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
    


c00=norm(C0)
#plt.plot(c00) 
idx=0   
But=[]
for i in newD0:
    num=int(i[3])-1
    c0=str(i[0])
    r0=float(i[1])
    
    
    
    d2=api.to_df(api.get_security_bars(9,0,c0,num,2)) #
    close=d2["close"]
    date=d2['datetime'][0]
    date_int=int(date[0:4]+date[5:7]+date[8:10])
    
    
    d1=api.to_df(api.get_history_minute_time_data(0,c0,date_int))
    qushi=np.array(d1["price"])
    
    
    if r0<0.4:
        
        idx+=1
        qus=qushi
        qushi=norm(qushi)
        c_0=close[0]
        c_1=close[1]
        r_c=(c_1-c_0)/c_0*100;
        r_c=round(r_c,2)
        print(f"股票{c0}相似距离是:{r0},第二天股票涨幅是:{r_c}%")
        pd=api.to_df(api.get_security_bars(9,0,c0,0,40-numday)) #
        close_0=pd['close'].iloc[-2]
        close_1=pd['close'].iloc[-1]
        
        
        if close_1>close_0:
            lishi=api.to_df(api.get_security_bars(9,0,c0,0,100));
            lishi=np.array(lishi["close"])
            But.append(plotK(pd,qushi,c0,lishi))
      
      
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

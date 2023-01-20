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




#%% 追跌
    

def caculate_today():
    all_flag=0
    true_flag=0    
    start=0
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
        
Code=caculate_today()  
Code=np.array(Code)
np.savetxt('sortCode\\三连跌一阳涨.txt',Code,fmt='%s')



#%% 追跌，三连跌，两连涨



    

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

Code=caculate_today0(0)

Code=np.array(Code)
np.savetxt('sortCode\\三连跌两阳涨.txt', Code, fmt='%s')


#%%


#%% 80选择,追涨
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
np.savetxt('sortCode\\追涨.txt', cd[:,0], fmt='%s')



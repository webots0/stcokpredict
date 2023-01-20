# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 23:14:33 2023

@author: webot
"""

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
import matplotlib.pyplot as plt
import datetime
import numpy as np
today = datetime.datetime.now().date()
api = TdxHq_API()  
import numpy as np
def normalize1(x):
    return (x - x.min()) / (x.max() - x.min())
def normalize2(x):
    return (x - x.mean()) / x.std()


# 求当天能量
def get_eneg(code,date):
    
   
        
    data1=api.to_df(api.get_history_minute_time_data(0, code,date))
    x=data1['price']
    y=(x-x[0])/x[0]/240
    y0=y.sum()
    
    return y0


start=0
data_R0=[]

code_df=np.load("code.npy",allow_pickle=True)

if api.connect('106.14.201.131', 7709):
    print('连接成功')
else:
    raise ValueError('连接失败')
    
    

def caculate(start,end):
    all_flag=0
    true_flag=0    
    
    EE=[]
    flag_E=[]
    for code in code_df:
        try:
            data0 = api.to_df(api.get_security_bars(9,0,code,start,end)) #
            data=data0.iloc[:-1,:]
            
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
                                    d4=data.iloc[-1,:]
                                    year=d4['year']
                                    month=d4['month']
                                    day=d4['day']
                                    date=[year,month,day]
                                    date_str = "{:04d}{:02d}{:02d}".format(date[0],date[1],date[2])
                                    
                                    date=int(date_str)
                                    E=get_eneg(code,date)
                                    EE.append(E)
                                    today=data.iloc[-1,:]
                                    tomor=data0.iloc[-1,:]
                                    price_today=today['close']
                                    price_tomor=tomor['close']
                                    flag=(price_tomor-price_today)/price_today
                                    flag_E.append(flag)
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
    return EE,flag_E

    
start=0
end=11
aa=[]
bb=[]
num = np.random.randint(1, 301, 30) 
idx=1
for start in num:
    a,b=caculate(start,end)
    aa.append(a)
    bb.append(b)
    la=len(a)
    
    print(f'第{idx}组-----当前有{la}组数据')
    idx+=1
#%%

   
#%%

import itertools

lst = aa
x = list(itertools.chain.from_iterable(lst))

lst = bb
y = list(itertools.chain.from_iterable(lst))
print(x,y)


plt.scatter(x,y)



#%%

    

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
# 计算当天三连跌一涨的股票
api = TdxHq_API()  


code_df=np.load("code.npy",allow_pickle=True)

if api.connect('106.14.201.131', 7709):
    print('连接成功')
else:
    raise ValueError('连接失败')
        
Code=caculate_today()  
Code=np.array(Code)
np.savetxt('sortCode\\三连跌.txt',Code,fmt='%s')

#%% 计算三连跌两涨

    

def caculate_today():
    all_flag=0
    true_flag=0    
    start=0
    end=10
    
    Code=[]
    for code in code_df:
        try:
            data0 = api.to_df(api.get_security_bars(9,0,code,start,end)) #
            data=data0.iloc[:-1,:]
            
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
                                        
                                        # 计算未来一天也是涨的
                                        d00=data0.iloc[-1,:]
                                        true8=d00['close']>d0['close']
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
# 计算当天三连跌两涨的股票
api = TdxHq_API()  


code_df=np.load("code.npy",allow_pickle=True)

if api.connect('106.14.201.131', 7709):
    print('连接成功')
else:
    raise ValueError('连接失败')
        
Code=caculate_today()  
Code=np.array(Code)
np.savetxt('sortCode\\三连跌两涨.txt',Code,fmt='%s')


#%% 三连跌一涨一跌
 

def caculate_today():
    all_flag=0
    true_flag=0    
    start=0
    end=10
    
    Code=[]
    for code in code_df:
        try:
            data0 = api.to_df(api.get_security_bars(9,0,code,start,end)) #
            data=data0.iloc[:-1,:]
            
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
                                        
                                        # 计算未来一天也是涨的
                                        d00=data0.iloc[-1,:]
                                        true8=d00['close']<d0['close']
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
# 
api = TdxHq_API()  


code_df=np.load("code.npy",allow_pickle=True)

if api.connect('106.14.201.131', 7709):
    print('连接成功')
else:
    raise ValueError('连接失败')
        
Code=caculate_today()  
Code=np.array(Code)
np.savetxt('sortCode\\三连跌一涨一跌.txt',Code,fmt='%s')


#%% 计算前一天的三连跌两涨
  

def caculate_today():
    all_flag=0
    true_flag=0    
    start=1
    end=10
    
    Code=[]
    for code in code_df:
        try:
            data0 = api.to_df(api.get_security_bars(9,0,code,start,end)) #
            data=data0.iloc[:-1,:]
            
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
                                        
                                        # 计算未来一天也是涨的
                                        d00=data0.iloc[-1,:]
                                        true8=d00['close']>d0['close']
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
# 计算昨天三连跌两涨的股票
api = TdxHq_API()  


code_df=np.load("code.npy",allow_pickle=True)

if api.connect('106.14.201.131', 7709):
    print('连接成功')
else:
    raise ValueError('连接失败')
        
Code=caculate_today()  
Code=np.array(Code)
np.savetxt('sortCode\\前一天三连跌两涨.txt',Code,fmt='%s')


#%% 计算昨天三连跌一涨一跌

def caculate_today():
    all_flag=0
    true_flag=0    
    start=1
    end=10
    
    Code=[]
    for code in code_df:
        try:
            data0 = api.to_df(api.get_security_bars(9,0,code,start,end)) #
            data=data0.iloc[:-1,:]
            
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
                                        
                                        # 计算未来一天也是涨的
                                        d00=data0.iloc[-1,:]
                                        true8=d00['close']<d0['close']
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
# 
api = TdxHq_API()  


code_df=np.load("code.npy",allow_pickle=True)

if api.connect('106.14.201.131', 7709):
    print('连接成功')
else:
    raise ValueError('连接失败')
        
Code=caculate_today()  
Code=np.array(Code)
np.savetxt('sortCode\\昨天三连跌一涨一跌.txt',Code,fmt='%s')




  
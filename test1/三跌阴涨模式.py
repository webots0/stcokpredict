# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 01:25:14 2023

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





    

def caculate_today():
    all_flag=0
    true_flag=0    
    start=1
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
                                    true7=d0['close']<d0['open']
                                    
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
np.savetxt('sortCode\\三连跌一阴涨.txt',Code,fmt='%s')



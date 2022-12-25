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
    b[name]=(b[name] - b[name].min()) / (b[name].max() - b[name].min())
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
    for i in s:
        b=NormPd(b, i)
    
    k=[]
    s0=s9
    for i in s:
        
        k.append(b[s0].corr(b[i]))
    result = sorted(zip(k, range(len(k))), reverse=True)
    
    ak=result[1][0]
    if ak<0.5:
    
        #plt.clf()
        
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.plot(b["日期"],b[s0],label=s0,linewidth=0.5)
        idx=0
        cr=""
        for i,j in result:
            if (idx>0) and (idx<3):
                sj=s[j]
                cr=cr+"和"+str(i)
                plt.plot(b["日期"],b[sj],label=sj,linewidth=0.5)
                if idx==2:
                    break
            idx=idx+1
           
        
        #ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
        plt.legend()
        
        plt.title(b["名称"][0]+f'相关系数={cr}')
        plt.show()
        return b["名称"][0]+f'相关系数={cr}'
    else:
        return None
NameList=[]
for i in range(1,300):
    NameList.append(StockCRmax_8(i))
    
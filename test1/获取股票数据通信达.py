# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 01:07:07 2023

@author: webot
"""
import pandas as pd
from pytdx.hq import TdxHq_API
import matplotlib.pyplot as plt
from pytdx.params import TDXParams
import time
import math

api = TdxHq_API()
if api.connect('106.14.201.131', 7709):
    # ... same codes...
    data = api.to_df(api.get_security_bars(9, 0, '000001', 0, 600)) #返回普通list
    d1=api.get_security_count(1)
    d2=api.to_df(api.get_security_list(0,0))
    aa=api.get_history_minute_time_data(TDXParams.MARKET_SH, '000001', 20221209)
    bb=api.get_minute_time_data(1, '000001')
    cc=api.get_history_transaction_data(TDXParams.MARKET_SZ, '000002', 0, 20000, 20221230)
    ff=api.get_company_info_category(TDXParams.MARKET_SZ, '000001')
    gg=api.get_company_info_content(0, '000001', '000001.txt', 3, 8000)
  
    api.disconnect()
   
#%%    
def normalize(x):
    
    return (x - x.min()) / (x.max() - x.min())

# 定义计算相似度的函数
def calc_similarity(x, y):
    return x.corr(y)
api = TdxHq_API()    
class ansysData:
    def __init__(self,code_location=(0,0,200),method=""):
        self.mark=code_location[0]
       
        if api.connect('106.14.201.131', 7709):
            print('连接成功')
        else:
            raise ValueError('连接失败')
        code_df=api.to_df(api.get_security_list(code_location[0],code_location[1]))
        self.code_df=code_df["code"].tail(code_location[2]).to_frame()
        
        self.close_df=[]
        self.jijie_std=[]
        self.yue_std=[]
        self.week_std=[]
        self.code_list=[]
        self.jijie_sel=[]
        self.yue_sel=[]
        self.week_sel=[]
        self.model1_pd=[]
        pass
    
    def get_history_close(self,start,year):
        close =[]
        for code in self.code_df["code"]:
            data = api.to_df(api.get_security_bars(9, self.mark, code,start,365*year)) #返回普通list
            close.append(data["close"])
        self.close_df=pd.concat(close,axis=1)
        self.close_df.fillna(0)
        code_list = [x for sublist in self.code_df.values.tolist() for x in sublist]
        self.code_list=code_list
        self.close_df.columns=code_list
        
        
        
    def std_close(self,method=""):
        # 季节波动 30*3=90
        # 月波动  30
        # 周波动 7
        code_list = [x for sublist in self.code_df.values.tolist() for x in sublist]
        self.code_list=code_list
        group_size = 90 
        num_groups = math.ceil(len(self.close_df) / group_size)  # 计算分组数量
        
        # 循环取数据
        group_list = []  # 存储分组数据的列表
        for i in range(num_groups):
            group = self.close_df[i*group_size : (i+1)*group_size]  # 取一组数据
            group_std=group.std()
            group_list.append(group_std)
        

        self.jijie_std=pd.concat(group_list,axis=1).transpose()
        self.jijie_std.columns=code_list
        
        group_size = 30 
        num_groups = math.ceil(len(self.close_df) / group_size)  # 计算分组数量
        
        # 循环取数据
        group_list = []  # 存储分组数据的列表
        for i in range(num_groups):
            group = self.close_df[i*group_size : (i+1)*group_size]  # 取一组数据
            group_std=group.std()
            group_list.append(group_std)
        

        self.yue_std=pd.concat(group_list,axis=1).transpose()
        self.yue_std.columns=code_list
        
        group_size = 7 
        num_groups = math.ceil(len(self.close_df) / group_size)  # 计算分组数量
        
        # 循环取数据
        group_list = []  # 存储分组数据的列表
        for i in range(num_groups):
            group = self.close_df[i*group_size : (i+1)*group_size]  # 取一组数据
            group_std=group.std()
            group_list.append(group_std)
        

        self.week_std=pd.concat(group_list,axis=1).transpose()
        self.week_std.columns=code_list
        
    def plt_std(self,mehtod=""):
        # 选择波动最大的前6只股票
        jijie_idx=self.jijie_std.sum().sort_values(ascending=False)
        self.jijie_std=self.jijie_std[jijie_idx.index]
        
        self.jijie_sel=jijie_idx.index[0:6]
        self.jijie_std[jijie_idx.index[0:6]].plot.bar()
        plt.show()
        
        yue_idx=self.yue_std.sum().sort_values(ascending=False)
        self.yue_std=self.yue_std[yue_idx.index]
        self.yue_sel=yue_idx.index[0:6]
        self.yue_std[yue_idx.index[0:6]].plot.bar()
        plt.show()
        
        
        week_idx=self.week_std.sum().sort_values(ascending=False)
        self.week_std=self.week_std[week_idx.index]
        self.week_sel=week_idx.index[0:6]
        self.week_std[week_idx.index[0:6]].plot()
        plt.show()
        
        pass
    
    def get_sel_history_close(self,start,year,num):
        # 每年该时间段的基本趋势走向相似度
        close =[]
        for code in self.jijie_sel:
            data = api.to_df(api.get_security_bars(9, self.mark, code,start,365*year)) #返回普通list
            close.append(data["close"])
        close_df_sel=pd.concat(close,axis=1)
        close_df_sel.fillna(0)
        # 选出所有需要进行归一化的行，每365行为一组
        group_size = 365
        num_groups = math.ceil(close_df_sel.shape[0] / group_size)
        groups = [close_df_sel.iloc[i*group_size:(i+1)*group_size] for i in range(num_groups)]
        
        # 对每一组数据进行归一化
        norm_group=[]
        for group in groups:
            normalized = (group - group.min()) / (group.max() - group.min())
            # 将归一化后的值赋回原数据帧
            normalized.columns=self.jijie_sel.tolist()
            normalized.reset_index(inplace=True)
            norm_group.append(normalized)
        plt.figure()   
        for group in norm_group:
            group[self.jijie_sel[num]].plot()
        plt.title(self.jijie_sel[num])
        plt.show()
        
                
              
        pass
    def create_model_fig(self):
        # -----\/模型  (14天，8天平缓，4天降，2天升)
        
        model1=[0,0,0,0,0,0,0,0,0,-0.3,-0.6,-0.9,-0.7,-0.5]
        #model1.reverse()
        #plt.figure()
        
        self.model1_pd=pd.DataFrame({"model1":model1})
        self.model1_pd.plot()
        
        
        # ------/ 模型
        
        
        pass
    
    def get_20_days_close(self):
        close_14_days=self.close_df.iloc[-14:]
        close_20_days=self.close_df.iloc[-20:]
        #close_14_days=close_14_days.apply(normalize)
        mode=self.model1_pd
        #similarity = mode['model1'].apply(calc_similarity, args=(close_14_days,))
        c=[]
        for name in close_14_days.columns:
            r=close_14_days[name].reset_index(drop=True)
            c0=r.corr(mode["model1"])
            if pd.isna(c0):
                c0=0
            c.append(c0)
        #close_14_days.sort_values(c.index)
        sorted_lst = sorted(enumerate(c), key=lambda x: x[1], reverse=True)

        # 提取大小值和索引值
        size_lst = [x[1] for x in sorted_lst]
        index_lst = [x[0] for x in sorted_lst]
        newx=[]
        code0=[]
        for idx in index_lst:
            code=self.code_list[idx]
            code0.append(code)
            newx.append(close_20_days[code])
        aa=pd.concat(newx,axis=1)
        aa=aa.reset_index(drop=True)
      
        fig, ax = plt.subplots(2,2)
        ax1=[ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
        for i in range(4):
            aa[code0[i]][0:14].plot(color='red',ax=ax1[i])
        
            aa[code0[i]][14:].plot(color='green',ax=ax1[i])
            print(code0[i])
        plt.show()

        
        pass
    
    def get_history_10_per(self,start=0,num=1):
        close =[]
        for code in self.code_df["code"]:
            data = api.to_df(api.get_security_bars(9, self.mark, code,start,num)) #返回普通list
            close.append(data["close"]/data["open"]-1)
        self.close_df=pd.concat(close,axis=1)
        self.close_df.fillna(0)
        code_list = [x for sublist in self.code_df.values.tolist() for x in sublist]
        self.code_list=code_list
        self.close_df.columns=code_list
        a2=self.close_df.loc[0].sort_values(ascending=False)
        close = []
        idx=0
        K=5
        for code in a2.index:
            data = api.to_df(api.get_security_bars(9, self.mark, code,0,21)) #返回普通list
            close.append(data["close"])
            idx+=1
            if idx>=K:
                break
            
        name=a2.index[0:K]    
        close_dff=pd.concat(close,axis=1)  
        close_dff.columns=name
        plt.subplot(2,2,1)
        close_dff[name[0]].plot.bar()
        close_dff[name[0]].plot(color='red')
        plt.title(name[0])
        plt.subplot(2,2,2)
        close_dff[name[1]].plot.bar()
        close_dff[name[1]].plot(color='red')
        plt.title(name[1])
        plt.subplot(2,2,3)
        close_dff[name[2]].plot.bar()
        close_dff[name[2]].plot(color='red')
        plt.title(name[2])
        plt.subplot(2,2,4)
        close_dff[name[3]].plot.bar()
        close_dff[name[3]].plot(color='red')
        plt.title(name[3])
        plt.show()
        
        
    
        
a=ansysData(code_location=(0,0,500))        
a.get_history_10_per()        


a1=a.close_df
    

#%%
def myfun(a,b):
    return a+b

aa=(1,2)
c=myfun(aa)


#%%
import matplotlib.pyplot as plt
buy=[]
pric=[]
for i in range(len(cc)):
    if cc[i]["buyorsell"]==1:
        r=-1
    else:
        r=1
        
    buy.append(r*cc[i]["vol"])
    pric.append(cc[i]["price"])
    
    
fig=plt.figure()
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)    
ax1.plot(buy)

ax2.plot(pric)
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 13:51:51 2023

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
def get_close_df(start=0):
    if api.connect('106.14.201.131', 7709):
            print('连接成功')
    else:
        raise ValueError('连接失败')
    
    end = 20
    close_df=[]
    code0=[]
    idx=0
    for code in code_df["code"]:
        try:
            data = api.to_df(api.get_security_bars(9,0,code,start,end)) #
            r=data["close"]
            if len(r)==20:
                close_df.append(r)
                code0.append(code)
                idx+=1
                
            #break
        except:
            
            pass
        
        if idx%100==0:
            print(idx)
        
        #break
    
    close_df=pd.concat(close_df,axis=1)
    close_df.fillna(0)
    
    close_df.columns=code0
    return code0,close_df

x0=[0,0,0,0,0,0,0,0,0,-0.3,-0.6,-0.9,-0.7,-0.5]
#code0,close_df=get_close_df(start=0)
def objc(x0):
    df0=pd.DataFrame({'001':x0})
    a={}
    
    for code in code0:
        a[code]=close_df[code].corr(df0['001'])
        
    b=sorted(a,key=a.get,reverse=True)
    r=0 # 相关系数
    c=0 # 涨幅
    
    for i in range(20):
        code=b[i]
        c0=close_df[code][14]
        c1=close_df[code][15:]
        c2=(c1/c0-1).mean() # 涨幅大小
        r=r+a[code]
        c=c+c2
        
    return -r,-c


#r,c=objc(x0) 
#
import numpy as np
from pymoo.core.problem import ElementwiseProblem
class MyProblem(ElementwiseProblem):

    def __init__(self):
        #self.start=start
        super().__init__(n_var=14,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=2*np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]),
                         xu=2*np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1]))

    def _evaluate(self, x, out, *args, **kwargs):
        (f1,f2)=objc(x)

       
        out["F"] = [f1, f2]
        out["G"] = []

#%%
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination  
from pymoo.optimize import minimize
termination = get_termination("n_gen", 80)
for i in range(14):
    try:
        code0,close_df=get_close_df(start=i)
        problem = MyProblem()
        
            
        
        algorithm = NSGA2(
            pop_size=80,
            n_offsprings=10,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        
       
        
        res = minimize(problem,
                       algorithm,
                       ('n_gen',100),
                       seed=1,
                       save_history=True,
                       verbose=True)
        
        X = res.X
        F = res.F
        xname=f"data_X\\{i}.csv"
        np.savetxt(xname, X,delimiter=',')
        fname=f"data_F\\{i}.csv"
        np.savetxt(fname, F,delimiter=',')
        print(f'保存第{i}成功')
    except:
        
        
        pass
            

#%%

#%%
import matplotlib.pyplot as plt
xl, xu = problem.bounds()
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
#plt.plot(F[:,0],F[:,1])
plt.show()

#%%

plt.figure()
plt.plot(X[9,:])

plt.show()
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 07:52:25 2022

@author: webot
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#%%
# 假设有5个节点，建立邻接矩阵
adj_matrix = np.array([[0, 1, 1, 0, 0],
                       [1, 0, 1, 1, 0],
                       [1, 1, 0, 0, 1],
                       [0, 1, 0, 0, 1],
                       [0, 0, 1, 1, 0]])
adj_matrix1 = np.array([[0, 1, 1, 0, 1],
                       [1, 0, 0, 1, 1],
                       [1, 0, 0, 1, 1],
                       [0, 1, 0, 0, 1],
                       [0, 0, 1, 1, 0]])
# 将邻接矩阵转为图谱
G = nx.Graph(adj_matrix)
G1=nx.from_numpy_matrix(adj_matrix1)
pos = nx.circular_layout(G)
# 画图
edge_colors = ['red' for _ in range(len(G.edges()))]
edge_colors1=['green' for _ in range(len(G.edges()))]
nx.draw(G, pos,with_labels=True,edge_color='red')

nx.draw(G1,pos,with_labels=True,edge_color='blue')
plt.show()
#%%
def plotG(adj_matrix):
    G = nx.Graph(adj_matrix)

    pos = nx.circular_layout(G)
  
    nx.draw(G, pos,with_labels=True,edge_color='red')
    
 
    plt.show()


#%%
from matplotlib.animation import FuncAnimation

# 定义画图函数

# 定义图像数据


# 动态展示图像

import os
def grafW(matrix):
    amx=matrix.max()
    amin=matrix.min()
    c=0.5
    amx*0.5
    matrix = torch.where(matrix > 0, torch.tensor(1.), torch.tensor(0.)) # 将矩阵里面的大于0的变成1，小于等于0的变成0

    
    for i in range(matrix.size(0)):
        matrix[i, i] = 0. # 将矩阵对角线赋值为1
    return matrix

L=os.listdir('WeightData')
idx=0
data=[]
for i in L:


    a1=torch.load('WeightData/'+i)
    #mat=grafW(a1['W4']-torch.mean(a1['W4'])).numpy()
    if idx==0:
        R1=a1['W4']
    else:
        R2=a1['W4']
        
    if idx>0:
        a0=R2-R1
        print(a0)
        
    #print(a1['W4'])
   # plotG(mat)
    idx=idx+1
    r=a1['W6']
    data.append(r)
   
   


#%% a4=grafW(aa).numpy()

    
#fig, ax = plt.subplots()
def update0(num):
    plt.imshow(torch.rand(13,13).mm(data[num]))
    #plt.show()
ani = FuncAnimation(plt.gcf(), update0, frames=len(data), interval=500)
plt.show()
  
#%%

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 定义画图函数
def update(i):
    plt.plot(i, i ** 2, 'o')

# 创建动画
ani = animation.FuncAnimation(plt.gcf(), update, frames=range(100), interval=500)
plt.show()
#%%


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig = plt.figure(figsize=(10, 5))  # 创建图
plt.rcParams["font.family"] = "FangSong"  # 支持中文显示
plt.ylim(-12, 12)  # Y轴取值范围
plt.yticks([-12 + 2 * i for i in range(13)], [-12 + 2 * i for i in range(13)])  # Y轴刻度
plt.xlim(0, 2 * np.pi)  # X轴取值范围
plt.xticks([0.5 * i for i in range(14)], [0.5 * i for i in range(14)])  # X轴刻度
plt.title("函数 y = 10 * sin(x) 在[0,2Π]区间的曲线")   # 标题
plt.xlabel("X轴")  # X轴标签
plt.ylabel("Y轴")  # Y轴标签
x, y = [], []  # 用于保存绘图数据，最开始时什么都没有，默认为空


def update(n):  # 更新函数
    x.append(n)  # 添加X轴坐标
    y.append(10 * np.sin(n))  # 添加Y轴坐标
    plt.plot(x, y, "r--")  # 绘制折线图


ani = FuncAnimation(fig, update, frames=np.arange(0, 2 * np.pi, 0.1), interval=50, blit=False, repeat=False)  # 创建动画效果
plt.show()  # 显示图片 作者：手把手教你学编程 https://www.bilibili.com/read/cv13169116 出处：bilibili    

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig, ax = plt.subplots()  # 创建画布和绘图区
ax.set_axis_off()  # 不显示坐标轴
x = np.arange(0, 2 * np.pi, 0.01)  # 生成X轴坐标序列
line1, = ax.plot(x, np.sin(x))  # 获取折线图对象，逗号不可少，如果没有逗号，得到的是元组
line2, = ax.plot(x, np.cos(x))  # 获取折线图对象，逗号不可少


def update(n):  # 动态更新函数
    line1.set_ydata(np.sin(x + n / 10.0))  # 改变线条y的坐标值
    line2.set_ydata(np.cos(x + n / 10.0))  # 改变线条y的坐标值


ani = FuncAnimation(fig, update, frames=10, interval=50, blit=False, repeat=False)  # 创建动画效果
plt.show()  # 显示图 作者：手把手教你学编程 https://www.bilibili.com/read/cv13169116 出处：bilibili

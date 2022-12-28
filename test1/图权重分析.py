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
import os
def grafW(matrix):
    matrix = torch.where(matrix > 0, torch.tensor(1.), torch.tensor(0.)) # 将矩阵里面的大于0的变成1，小于等于0的变成0

    for i in range(matrix.size(0)):
        matrix[i, i] = 0. # 将矩阵对角线赋值为1
    return matrix

L=os.listdir('WeightData')
idx=0
for i in L:


    a1=torch.load('WeightData/'+i)
    mat=grafW(a1['W4']-torch.mean(a1['W4'])).numpy()
    if idx==0:
        R1=a1['W4']
    else:
        R2=a1['W4']
        
    if idx>0:
        a0=R2-R1
        print(a0)
        
    #print(a1['W4'])
    plotG(mat)
    idx=idx+1
    R1=R2
  
    

#%% a4=grafW(aa).numpy()

    


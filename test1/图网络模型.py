# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 04:52:04 2022

@author: webot
"""

print('----------')


# 新闻数据，随时间变化的一个网络结构

# 股票数据,随时间变化的一个单序列数据

'''
网络： X 特征数据 H 图结构  

股票： Y 特征数据

X-H来发现训练发现与Y特征的相关性,以及预测未来的Y

很合理
'''
import torch

#x=torch.Tensor([1,2,3])

x = torch.Tensor([[1, 2], [3, 4]])
y = torch.Tensor([[5, 6], [7, 8]])
z = x.mul(y) # z = [[15+27, 16+28], [35+47, 36+48]] = [[19, 22], [43, 50]]
print(z)
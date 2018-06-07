# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 19:16:05 2017

@author: user
"""

from smart_grid import *

hh=[0.0001,0.001,0.01,0.1,0.2,0.4,0.6,0.8,1,10]
ll=[1,10]
cc=[]
for ii in hh:
    for jj in ll:
        cc.append([ii,jj])

for i in range(10):
    alpha=0.01 #迭代1的参数
    beta=0.01 #迭代2的参数
    gamma=0.01 #迭代3的参数
    lanmda=0.01 #迭代1的参数
    n=70 #单独电器分解后的维度
    for j in cc:
        h_l=j
        run(alpha,beta,gamma,lanmda,n,h_l)
    i+=1
    

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def load(path):
    with open(path,'r') as f:
        lines=f.readlines()
        data=[eval(line) for line in lines]
   
    return data
    

def draw(x,arr1,arr2,y1,y2,name=None):
    
    fig = plt.figure(figsize=(13, 5), dpi=500)
  

    axis_1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))

    axis_2 = axis_1.twinx()
    axis_1.plot(x,arr1,label='acc',c='r')
    axis_1.axhline(y=y1,c='orange',lw=3, linestyle='--',label='baseline_acc')  
 
    axis_2.plot(x,arr2,label='gene_num',c='b')
    axis_2.axhline(y=y2,c='green',lw=3,linestyle='--',label='baseline_gene_num')
    if name:
        plt.title(name)
    fig.legend()    

import os
import sys
import time
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif

sys.path.append("./")
from utils.preprocess import load
from utils.evaluate import evaluate_by_leiden
from utils.logger import Logger
from scipy.stats import norm


class HRG:

    def __init__(self, X, y,taskname,**arg):  
        self.X = X
        self.y = y
        self.taskname=taskname
        

    def run(self, n_selected=None,coefficient=1e4): 
        path=f"model/HRG_result/{self.taskname}.txt"
        if not os.path.exists(path):
            print('Please run HRG.R first!')
            raise NameError
        
        with open(path,'r') as f:
            lines=f.readlines()
            s=lines[0].strip().replace('Feature','').replace(' ',',')
            indexs=list(map(int,eval(s)))
      
        return indexs

        
    def log(self, logger):
        pass


if __name__ == "__main__":

    task_name = "Chu1"
    X, y = load(task_name)
    hrg = HRG(X, y,taskname=task_name)
    index = hrg.run(300)

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
from sklearn.preprocessing import MinMaxScaler


class Genebasis:

    def __init__(self, X, y,taskname,**arg):  
        self.X = X
        self.y = y
        self.taskname=taskname
        

    def run(self, n_selected=None,coefficient=1e4): 
        path=f"model/GeneBasis_result/{self.taskname}.txt"
        if not os.path.exists(path):
            print('Please run Genebasis.R first!')
            raise NameError
        
        with open(path,'r') as f:
            lines=f.readlines()
            s=lines[0].replace('1:50,c','')
            indexs=list(map(int,eval(s)))
            

        feature_importances=np.arange(len(indexs))
        mean = np.mean(feature_importances)
        std = np.std(feature_importances)
        standardized_data = (feature_importances - mean) / std
        dist = norm(loc=0, scale=1)
        cdf_values = dist.cdf(standardized_data)
        scores = cdf_values[::-1]
        self.scores=np.zeros(self.X.shape[1])
        self.scores[indexs]=scores
        scaler = MinMaxScaler()
        self.scores = scaler.fit_transform(self.scores.reshape(-1,1)).reshape(-1)/coefficient
        return indexs

        
    def log(self, logger):
        pass


if __name__ == "__main__":

    task_name = "Chu1"
    X, y = load(task_name)
    genebasis = Genebasis(X, y,taskname=task_name)
    index = genebasis.run(300)
    print(index,genebasis.scores)

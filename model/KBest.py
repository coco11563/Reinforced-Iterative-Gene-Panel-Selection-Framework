import sys
import time
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif

sys.path.append("./")
from utils.preprocess import load
from utils.evaluate import evaluate_by_leiden
from utils.logger import Logger
from sklearn.preprocessing import MinMaxScaler


class KBest:

    def __init__(self, X, y,**arg):  
        self.X = X
        self.y = y

    def run(self, n_selected=None,coefficient=1e4): 
        if n_selected == None:
            n_selected = self.X.shape[1] // 20
        skb = SelectKBest(score_func=f_classif, k=int(n_selected))
        skb.fit(self.X, self.y)
        choice = skb.get_support()
        self.selected_features=np.argwhere(choice).flatten()
        self.scores=np.array(list(skb.scores_))
        self.scores=np.nan_to_num(self.scores)
        scaler = MinMaxScaler()
        self.scores = scaler.fit_transform(self.scores.reshape(-1,1)).reshape(-1)/coefficient

        return self.selected_features
        
    def log(self, logger):
        pass


if __name__ == "__main__":

    task_name = "Chu1"
    X, y = load(task_name)
    kbest = KBest(X, y)
    index = kbest.run(300)
    kbest.scores[6]=0
    print(np.min(kbest.scores))

    
    # try:
    #     X_ = X.iloc[:, index]
    # except:
    #     X_ = X.loc[:, index]
    # print(evaluate_by_leiden(X_, y.flatten()))

import sys
import time

import pandas as pd
import numpy as np
import scanpy as sc

from mrmr import mrmr_classif
from sklearn.datasets import make_classification

sys.path.append('./')
from utils.preprocess import load
from utils.evaluate import evaluate_by_leiden


class mRMR:

    def __init__(self, X, y,**arg): 
        self.X = X
        self.y = y

    def run(self, n_selected=None):
        if n_selected==None:
            n_selected=50
        self.selected_features = mrmr_classif(X=self.X, y=self.y, K=int(n_selected),show_progress=False)
        
        return self.selected_features
        

    def log(self, logger):
        pass


if __name__ == "__main__":
    
    task_name='Leng'
    X, y = load(task_name)
    adata = sc.AnnData(X)
    sc.pp.normalize_per_cell(adata,min_counts=0)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata,use_rep='X_pca')
    resolution=1
    sc.tl.leiden(adata,resolution=resolution)
    labels_predict=adata.obs['leiden']
    labels_predict=np.array(labels_predict).reshape(-1,1)
    start=time.time()
    mrmr=mRMR(X,labels_predict)
    index=mrmr.run(10)
    print(index)
    try:
        X_=X.iloc[:,index]
    except:
        X_=X.loc[:,index]
    print(evaluate_by_leiden(X_, y.flatten()),time.time()-start)
    

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

sys.path.append('./')
from utils.preprocess import load
from utils.evaluate import evaluate_by_leiden


class LASSO:

    def __init__(self, X, y,**arg): 
        self.X = X
        self.y = y

    def run(self,n_selected=None):  
        if n_selected==None:
            n_selected=self.X.shape[1]//20
        lasso = Lasso(alpha=0.1)  
  
 
        lasso.fit(self.X, self.y)  
        
      
        coef = lasso.coef_  
        
 
        self.selected_features = np.where(coef != 0)[0]  
        
        return self.selected_features

    def log(self, logger):
        pass


if __name__ == "__main__":
    
  
    task_name='Cao'
    X, y = load(task_name)
    lasso=LASSO(X,y)
    index=lasso.run(200)
    print(index)
    try:
        X_=X.iloc[:,index]
    except:
        X_=X.loc[:,index]
    print(evaluate_by_leiden(X_, y.flatten()))
    
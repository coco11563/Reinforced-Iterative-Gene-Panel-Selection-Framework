import sys
import sklearn

sys.path.append('./')
from utils.preprocess import load
from utils.evaluate import evaluate_by_leiden
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn  import feature_selection

import numpy as np


#Recursive Feature Elimination
class RFE:

    def __init__(self, X, y,**arg):
        self.X = X
        self.y = y
        self.arg=arg

    def run(self,n_selected=None,coefficient=1e4):  
        if n_selected==None:
            n_selected=self.X.shape[1]//20

        seed=self.arg['seed'] if self.arg.get('seed') else 1
        estimator = RandomForestClassifier(random_state=seed, n_jobs=128)
        step=(self.X.shape[1]-n_selected)//20
        selector = feature_selection.RFE(estimator, n_features_to_select=n_selected, step=step)
        selector.fit(self.X, self.y.flatten())
        supports=selector.get_support()
        indexs=np.where(supports)[0]
        scaler = MinMaxScaler()
        self.scores = scaler.fit_transform(selector.ranking_.reshape(-1,1)).reshape(-1)/coefficient
   
        return indexs




if __name__ == "__main__":
    

    task_name='Chu1'
    X, y = load(task_name)
    rfe=RFE(X,y)
    index=rfe.run()
    print(index,max(rfe.scores))
    # try:
    #     X_=X.iloc[:,index]
    # except:
    #     X_=X.loc[:,index]
    # print(evaluate_by_leiden(X_, y.flatten()))
    

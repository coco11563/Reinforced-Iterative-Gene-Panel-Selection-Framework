import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC

sys.path.append('./')
from utils.preprocess import load
from utils.evaluate import evaluate_by_leiden
from sklearn.preprocessing import MinMaxScaler


class SVM:

    def __init__(self, X, y,**arg): 
        self.X = X
        self.y = y
        self.arg=arg

    def run(self,n_selected=None,coefficient=1e4):  
        if n_selected==None:
            n_selected=self.X.shape[1]//20

        seed=self.arg['seed'] if self.arg.get('seed') else 1
        svm_model = SVC(kernel='linear', C=0.5, random_state=seed)
        svm_model.fit(self.X, self.y.flatten())
        indexs_svc=np.argmax(svm_model.coef_,axis=1)
        scaler = MinMaxScaler()
        self.scores = scaler.fit_transform(svm_model.coef_[0,:].reshape(-1,1)).reshape(-1)/coefficient
        
        return indexs_svc
    

if __name__ == "__main__":
    
    task_name='Cao'
    X, y = load(task_name)
    svm=SVM(X,y)
    index=svm.run(200)
    print(index,svm.scores.shape)
    # try:
    #     X_=X.iloc[:,index]
    # except:
    #     X_=X.loc[:,index]
    # print(evaluate_by_leiden(X_, y.flatten()))
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.append("./")
from utils.preprocess import load
from utils.evaluate import evaluate_by_leiden
from sklearn.preprocessing import MinMaxScaler


class RandomForest:

    def __init__(self, X, y, **arg):
        self.X = X
        self.y = y
        self.arg = arg

    def run(self, n_selected=None):
        if n_selected == None:
            n_selected = self.X.shape[1] // 20

        seed = self.arg["seed"] if self.arg.get("seed") else 1

        rf = RandomForestClassifier(
            n_estimators=1000, n_jobs=-1, bootstrap=False, random_state=seed
        )
        rf.fit(self.X, self.y.flatten())
        feature_importances = rf.feature_importances_
        scaler = MinMaxScaler()
        self.scores = scaler.fit_transform(rf.feature_importances_.reshape(-1,1)).reshape(-1)
        importance_idx = np.argsort(feature_importances)
        std = np.std(feature_importances)
        mean = np.mean(feature_importances)
        mean2sigma = mean + 2 * std
        N_feature = np.sum(feature_importances >= mean2sigma)
        indexs_rf = importance_idx[-N_feature:]

        return indexs_rf


if __name__ == "__main__":

    task_name = "Chu1"
    X, y = load(task_name)
    rf = RandomForest(X, y)
    index = rf.run(200)
    print(index, rf.scores)

    # try:
    #     X_=X.iloc[:,index]
    # except:
    #     X_=X.loc[:,index]
    # print(evaluate_by_leiden(X_, y.flatten()))

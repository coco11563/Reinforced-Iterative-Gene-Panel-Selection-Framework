import os
import sys
import time
import argparse
import traceback
import warnings

import numpy as np
import scanpy as sc
import pandas as pd

from gap_statistic import OptimalK
from model.CellBRF import CellBRF, cellbrf
from model.RandomForest import RandomForest
from model.RFE import RFE
from model.SVM import SVM
from model.KBest import KBest
from model.HRG import HRG
from model.mRMR import mRMR
from model.Genebasis import Genebasis
from utils.networks import DQN
from utils.preprocess import load
from utils.evaluate import evaluate_by_leiden
from utils.networks import Feature_AE
from utils.logger import Logger
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

params = {
    "LEARN_STEPS": 50,
    "EXPLORE_STEPS": 50,
    "LEARN_EPSILON": 0.999,
    "EXPLORE_EPSILON": 0.6,
    "N_STATES": 64,
    "N_ACTIONS": 2,
    "TARGET_REPLACE_ITER": 100,
    "MEMORY_CAPACITY": 800,
    "seed": 1,
    "filter_model": ["RandomForest", "SVM", "KBest"],
    "prior_model": ["mRMR", "KBest", "cellbrf"],
    "INJECTION_NUMBER": 800,
    "filter": True,
}


def dopca(X, dim):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10


def predict_label(data, k_clusters, seed, pca=50, k=15):
    adj = get_adj(count=data, pca=pca, k=k)
    label_predict = SpectralClustering(
        n_clusters=k_clusters,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=seed,
    ).fit_predict(adj)
    return label_predict


def get_adj(count, k=15, pca=50, mode="connectivity"):
    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()
    return adj


class RIGPS:

    def __init__(
        self,
        taskname,
        data,
        y_true,
        params,
        show=True,
    ):
        self.X = data
        self.y = y_true
        self.taskname = taskname
        self.params = params
        self.show = show

    def run(self):
        df = []
        indexs = np.arange(self.X.shape[1])

        adata = sc.AnnData(self.X)
        sc.pp.normalize_per_cell(adata, min_counts=0)
        sc.pp.log1p(adata)
        sc.tl.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, use_rep="X_pca")
        resolution = 1
        sc.tl.leiden(adata, resolution=resolution)
        label_predict = adata.obs["leiden"]
        label_predict = np.array(label_predict).reshape(-1, 1)

        # Gene Pre-Filtering
        if self.params["filter"]:
            start = time.time()
            importances = np.zeros(self.X.shape[1])
            for model in self.params["filter_model"]:
                model = eval(model)
                model = model(
                    self.X, self.y, taskname=self.taskname, seed=self.params["seed"]
                )
                indexs_temp = model.run()
                scores = model.scores
                X_ = self.X.iloc[:, indexs_temp]
                metrics = evaluate_by_leiden(X_, self.y.flatten())
                importances_temp = scores * metrics["NMI"]
                importances += importances_temp

            importance_idx = np.argsort(importances)
            std = np.std(importances)
            mean = np.mean(importances)
            mean2sigma = mean + 2 * std
            N_feature = np.sum(importances >= mean2sigma)
            indexs_prefilter = importance_idx[-N_feature:]
            time_filter = time.time() - start  
            if self.show:
                print(
                    f"Gene pre-filtering completed! time:{time_filter},n_selected_feature:{len(indexs_prefilter)}"
                )

        # Knowledge Injection
        is_prior_konwledge = (
            self.params["prior_model"] != None and len(self.params["prior_model"]) != 0
        )
        if is_prior_konwledge:
            start = time.time()
            prior_knowledge = {}
            for prior_model_ in self.params["prior_model"]:
                prior_model = eval(prior_model_)
                model = prior_model(
                    self.X, self.y, taskname=self.taskname, seed=self.params["seed"]
                )
                indexs_prior_knowledge = model.run()
                X_ = self.X.iloc[:, indexs_prior_knowledge]
                metrics = evaluate_by_leiden(X_, label_predict.flatten())
                state = Feature_AE(X_, N_HIDDEN=self.params["N_STATES"])
                actions = np.zeros(self.X.shape[1])
                actions[indexs_prior_knowledge] = 1
                reward = self.rewards(metrics["NMI"], actions)
                prior_knowledge[prior_model_] = {
                    "genes": indexs_prior_knowledge,
                    "reward": reward,
                    "state": state,
                }

            time_prior = time.time() - start
            metrics = evaluate_by_leiden(X_, label_predict.flatten())
            if self.params["filter"]:
                indexs = np.union1d(indexs_prefilter, indexs_prior_knowledge)

            if self.show:
                print(
                    f"Preparation of prior knowledge completed! time:{time_prior},n_selected_feature:{len(indexs)}"
                )

        # Reinforced Iteration
        start = time.time()
        self.N_feature = len(indexs)
        X_ = self.X.iloc[:, indexs]
        self.agents = []
        params_ = ["N_STATES", "N_ACTIONS", "MEMORY_CAPACITY", "TARGET_REPLACE_ITER"]
        for idx in indexs:
            agent = DQN(**{k: self.params[k] for k in params_})
            if is_prior_konwledge:
                for key in self.params["prior_model"]:
                    state = prior_knowledge[key]["state"]
                    reward = prior_knowledge[key]["reward"]
                    for _ in range(self.params["INJECTION_NUMBER"]):
                        agent.store_transition(
                            state,
                            1 if idx in prior_knowledge[key]["genes"] else 0,
                            reward,
                            state.copy(),
                        )
            self.agents.append(agent)

        initial_actions = np.random.randint(2, size=self.N_feature)
        choiced_data = X_.iloc[:, initial_actions == 1]
        state = Feature_AE(choiced_data, N_HIDDEN=self.params["N_STATES"])

        start = time.time()
        params_ = ["LEARN_STEPS", "EXPLORE_STEPS", "EXPLORE_EPSILON", "LEARN_EPSILON"]
        LEARN_STEPS, EXPLORE_STEPS, EXPLORE_EPSILON, LEARN_EPSILON = (
            self.params[k] for k in params_
        )
        for i in range(LEARN_STEPS + EXPLORE_STEPS):
            actions = np.zeros(X_.shape[1])

            is_explore = i < EXPLORE_STEPS

            for agent, dqn in enumerate(self.agents):

                actions[agent] = dqn.choose_action(
                    state, EPSILON=EXPLORE_EPSILON if is_explore else LEARN_EPSILON
                )

            choiced_data = X_.iloc[:, actions == 1].astype(np.float64)
            state_ = Feature_AE(choiced_data, N_HIDDEN=self.params["N_STATES"])
            metrics_true = evaluate_by_leiden(choiced_data, self.y.flatten())
            metrics = evaluate_by_leiden(choiced_data, label_predict.flatten())
            reward = self.rewards(metrics["NMI"], actions)

            for index, agent in enumerate(self.agents):
                agent.store_transition(state, actions[index], reward, state_)

            if not is_explore:
                for agent in self.agents:
                    agent.learn()

            state = state_

            row = {
                "time": time.time() - start,
                "feature_selected": indexs[actions == 1],
                "n_feature_selected": actions.sum(),
                "n_feature": self.X.shape[1],
                "cells": self.X.shape[0],
                "task": self.taskname,
                "stage": "explore" if is_explore else "learn",
                "step": i,
            }
            if self.params["filter"]:
                row["time_filter"] = time_filter
            if is_prior_konwledge:
                row["time_prior"] = time_prior

            row.update(metrics)
            row["NMI"] = metrics_true["NMI"]
            row["ARI"] = metrics_true["ARI"]
            row["SI"] = metrics_true["SI"]
            df.append(row)

            if self.show:
                print(
                    f" step {i} {'explore' if is_explore else 'exploit'} dataset:{self.taskname}  n_select_feature:{actions.sum()} NMI {metrics_true['NMI']}"
                )

        optimal_gene_set= sorted(df, key=lambda x: x["NMI"], reverse=True)[0]

        if self.show:
            print(
                f"\noptimal gene panel \nn_select_feature:{len(df[0]['feature_selected'])} NMI {df[0]['NMI']}"
            )
            print(f"gene panel performance saved in result/{self.taskname}_performance.txt")
            print(f"gene panel saved in result/{self.taskname}_gene_panel.txt")

        os.makedirs('./result', exist_ok=True)
        with open(f"result/{self.taskname}_performance.txt", "w") as f:
            for key in df[0]:
                if key in ["feature_selected", "step", "stage"]:
                    continue
                f.write(f"{key}:{df[0][key]}\n")

        with open(f"result/{self.taskname}_gene_panel.txt", "w") as f:
            f.write(str(df[0]["feature_selected"]))

        return optimal_gene_set

    def rewards(self, point, actions):

        N_feature = len(actions)
        reward = (
            point + (N_feature - actions.sum()) / (N_feature + actions.sum())
        ) * 10

        return reward



if __name__ == "__main__":

    dataset='Chu1'
    X, y = load(dataset)
    rigps = RIGPS(dataset, X, y, params)
    optimal_gene_set = rigps.run()
    print(optimal_gene_set)

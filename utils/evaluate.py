import os
import torch

import numpy as np
import pandas as pd
import scanpy as sc  

from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score,silhouette_score
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.mixture import GaussianMixture


def evaluate(data,y_pred,y_true):
    nmi=normalized_mutual_info_score(y_true,y_pred)
    ari=adjusted_rand_score(y_true,y_pred)
    si=silhouette_score(data,y_pred)

    metrics={
        'NMI':nmi,
        'ARI':ari,
        'SI':si
    }
    return metrics


def evaluate_by_kmeans(data,y_true):
    n_clusters=len(np.unique(y_true))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    y_pred = kmeans.labels_  

    return  evaluate(data,y_pred,y_true)


def evaluate_by_agglo(data,y_true):
    n_clusters=len(np.unique(y_true))
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    agglo.fit(data)
    y_pred = agglo.labels_

    return  evaluate(data,y_pred,y_true)


def evaluate_by_GMM(data,y_true):
    n_clusters=len(np.unique(y_true))
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    gmm.fit(data)
    y_pred = gmm.predict(data)

    return  evaluate(data,y_pred,y_true)


def evaluate_by_leiden(data,y_true,is_pca=False):

    resolution=1
    adata=sc.AnnData(data)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    if is_pca:
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata,use_rep='X_pca')
    else:
        sc.pp.neighbors(adata,use_rep='X')  
    sc.tl.leiden(adata,resolution=resolution)
    y_pred=adata.obs['leiden']

    return  evaluate(data,y_pred,y_true)


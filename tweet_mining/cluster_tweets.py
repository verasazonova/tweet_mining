__author__ = 'verasazonova'

import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def crp_clusters(vecs):
    """
    Clusters vector data using chineese restaurant process
    :param vecs: an array of vectors
    :return: an array of cluster indices
    """
    cluster_vec = []  # tracks sum of vectors in a cluster
    cluster_idx = []  # array of index arrays. e.g. [[1, 3, 5], [2, 4, 6]]
    ncluster = 0
    # probablity to create a new table if new customer
    # is not strongly "similar" to any existing table
    pnew = 1.0 / (1 + ncluster)
    n_vecs = len(vecs)
    logging.info("CRP with %i vectors" % n_vecs)
    rands = np.random.random(n_vecs)

    print rands
    for i in range(n_vecs):
        max_sim = -np.inf
        max_idx = 0
        v = vecs[i]
        for j in range(ncluster):
            sim = cosine_similarity(v, cluster_vec[j])
            if sim < max_sim:
                max_idx = j
                max_sim = sim
            if max_sim < pnew:
                if rands(i) < pnew:
                    cluster_vec[ncluster] = v
                    cluster_idx[ncluster] = [i]
                    ncluster += 1
                    pnew = 1.0 / (1 + ncluster)
                continue
        print max_idx
        cluster_vec[max_idx] = cluster_vec[max_idx] + v
        cluster_idx[max_idx].append(i)
    return cluster_idx


import numpy as np
import scipy.stats
import scipy.spatial.distance
import random
from sklearn.cluster import KMeans

import os

shared = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared')

# def calc_entropy(pred):
#     e = -np.sum(np.log(pred) * pred, axis=1)
#     return e

"""
score: bigger is more informative
"""


def get_max_k(scores, k):
    unsorted = np.argpartition(-scores, k)[:k]
    return unsorted[np.argsort(-scores[unsorted])]


def uncertainly_sampling(pred, strategy):
    if strategy == 'entropy':
        scores = scipy.stats.entropy(pred.T)
    elif strategy == 'least_confident':
        scores = -np.max(pred, axis=1)
    elif strategy == 'margin':
        sorted_pred = np.sort(pred, axis=1)
        scores = -(sorted_pred[:, -1] - sorted_pred[:, -2])
    else:
        raise ValueError('invalid stare')
    return scores


def perform_clustering(feat, kmeans_cache=None):
    if kmeans_cache is not None and os.path.exists(kmeans_cache):
        kmeans = np.load(kmeans_cache)
    else:
        kmeans = KMeans(n_clusters=100, random_state=0, n_init=5).fit(feat)
    cluster_indices = kmeans.predict(feat)
    dist = kmeans.transform(feat)
    return cluster_indices, dist


def get_centroid(cluster_indices, dist):
    candidates = dict()
    for idx, cluster_id in enumerate(cluster_indices):
        if cluster_id not in candidates:
            candidates[cluster_id] = idx
        else:
            rep_idx = candidates[cluster_id]
            if dist[idx, cluster_id] < dist[rep_idx, cluster_id]:
                candidates[cluster_id] = idx

    return np.array(list(candidates.values()))


def query_by_committee(preds, strategy):
    if strategy == 'vote_entropy':
        vote = np.argmax(preds, axis=2)
        vote_count = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), axis=0, arr=vote)
        vote_entropy = scipy.stats.entropy(vote_count)
        scores = vote_entropy
    elif strategy == 'average_kl':
        mean_p = np.mean(preds, axis=0)
        average_kl = np.mean(np.sum(preds * np.log(preds / mean_p), axis=2), axis=0)
        scores = average_kl
    else:
        raise ValueError('invalid strategy')

    return scores


def information_density(feat, indices, beta):
    normalized_feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
    perm = np.random.permutation(len(feat))[:10000]
    densities = np.mean(np.dot(normalized_feat[indices], normalized_feat[perm].T), axis=1)
    densities **= beta
    return densities


def query_dataset(labeled_data, unlabeled_data, feat, query_indices):
    labeled_data = labeled_data + [unlabeled_data[idx] for idx in query_indices]
    unlabeled_data = [unlabeled_data[idx] for idx in range(len(unlabeled_data)) if idx not in query_indices]
    feat = np.delete(feat, query_indices, 0)
    return labeled_data, unlabeled_data, feat


def initialize_labeled_dataset(opt, unlabeled_data, feat):
    if opt.label_init == 'clustering':
        if os.path.exists(opt.initial_label_cache):
            query_indices = np.load(opt.initial_label_cache)
        else:
            cluster_indices, dist = perform_clustering(feat, opt.kmeans_cache)
            indices = get_centroid(cluster_indices, dist)
            query_indices = indices[np.random.permutation(len(indices))[:opt.init_size]]
            np.save(opt.initial_label_cache, query_indices)
    else:
        query_indices = np.random.permutation(len(unlabeled_data[0]))[:opt.init_size]

    labeled_data = [unlabeled_data[idx] for idx in query_indices]
    unlabeled_data = [unlabeled_data[idx] for idx in range(len(unlabeled_data)) if idx not in query_indices]
    feat = np.delete(feat, query_indices, 0)
    return labeled_data, unlabeled_data, feat


def active_annotation(preds, feat, opt):

    scores = np.zeros(preds.shape[1])

    if opt.uncertain:
        scores += uncertainly_sampling(preds[0], opt.uncertain_strategy)

    if opt.uncertain_with_dropout:
        scores += uncertainly_sampling(np.mean(preds, axis=0), opt.uncertain_strategy)

    elif opt.query_by_committee:
        scores += query_by_committee(preds, opt.qbc_strategy)

    cluster_indices, dist = perform_clustering(feat, kmeans_cache=opt.kmeans_cache)

    D = set()
    Q = set()
    for idx in np.argsort(-scores):
        cluster_id = cluster_indices[idx]
        if cluster_id not in D:
            Q.add(idx)
            D.add(cluster_id)

        if len(Q) == opt.active_sample_size:
            break

    return np.array(list(Q))

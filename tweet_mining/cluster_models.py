__author__ = 'verasazonova'

import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import DPGMM
import numpy as np
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from operator import itemgetter
from gensim.matutils import corpus2dense
import os.path
import pickle
import w2v_models


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


def load_dpggm(dpggm_model_name="dpggm_30_5"):
    clf = None
    if os.path.isfile(dpggm_model_name):
        clf = pickle.load(open(dpggm_model_name, 'rb'))
        logging.info("Loaded from file")
    return clf


def load_build_dpggm(dpggm_model_name, x_data):
    if os.path.isfile(dpggm_model_name):
        clf = load_dpggm(dpggm_model_name)
    else:
        clf = DPGMM(n_components=30, covariance_type='diag', alpha=5,  n_iter=1000)
        logging.info("Fitting with DPGMM")
        clf.fit(x_data)
        pickle.dump(clf, open(dpggm_model_name, 'wb'))
        logging.info("Fitted")
        print clf.converged_
    return clf


def print_centers(x_words, y_,  clf, w2v_model=None, min_size=5, min_size_english=800):
    cluster_to_recluster = []
    for i, cluster_center in enumerate(clf.means_):
        cluster = x_words[y_ == i]
        cluster_size = len(cluster)
        if cluster_size > min_size:
            print "%i, %i   :" % (i, cluster_size)
            if w2v_model is not None:
                central_words = [word for word, _ in w2v_model.most_similar_cosmul(positive=[cluster_center], topn=20)]
                print repr(central_words)
        if cluster_size > min_size_english:
            cluster_to_recluster.append(i)
    return cluster_to_recluster


def print_first_words(clf, w2v_model):
    for i, cluster_center in enumerate(clf.means_):
        if w2v_model is not None:
            central_words = [word for word, _ in w2v_model.most_similar_cosmul(positive=[cluster_center], topn=20)]
            print i, repr(" ".join(central_words))[2:-1]


#TODO: Pass all words, cluster on median freq, then predict all.
def DPGGM_clustering(x_data, x_words, min_size=5, w2v_model=None, min_size_english=800):

    dpggm_model_name = "dpggm_30_5"
    clf = load_dpggm(dpggm_model_name)

    y_ = clf.predict(x_data)
    cluster_to_recluster = print_centers(x_words, y_, clf, w2v_model,
                                         min_size=min_size, min_size_english=min_size_english)

    ind_to_recluster = y_ == cluster_to_recluster[0]
    for cluster in cluster_to_recluster[1:]:
        ind_to_recluster = y_ == cluster | ind_to_recluster

    x_data = x_data[ind_to_recluster]
    x_words = x_words[ind_to_recluster]

    print x_data.shape, x_words.shape

    def try_covar(type_str, x_words):
        clf = DPGMM(n_components=20, covariance_type=type_str, alpha=30,  n_iter=1000)
        clf.fit(x_data)
        y_ = clf.predict(x_data)
        print type_str
        print_centers(x_words, y_, clf)
        print

    try_covar("diag", x_words)
    try_covar("spherical", x_words)
    try_covar("tied", x_words)
    try_covar("full", x_words)


def assign_language(tweets, w2v_model, ids):

    dpggm_model_name = "dpggm_30_5"
    clf = load_dpggm(dpggm_model_name)

    #print_first_words(clf, w2v_model)

    id2lang = {}
    with open("word_clusters_identified.txt", 'r') as f:
        for line in f:
            id2lang[int(line.split(':')[0])] = line.split(':')[1].strip()

    def most_common(lst):
        return sorted(set(lst), key=lst.count, reverse=True)

    for i, (tweet, tweet_id) in enumerate(zip(tweets, ids)):
        tweet_vector = w2v_models.vectorize_tweet_corpus(w2v_model, tweet)
        clusters = clf.predict(tweet_vector)
        languages = most_common([id2lang[cluster] for cluster in clusters])
        if len(languages) > 1:
            languages = languages[1:]
        print "%s, %s" % (tweet_id, " ".join(languages))

        if i % 100 == 0:
            logging.info("Processed %i tweets" % i)


# **************** Cluster relating functions ******************************

def build_clusters(counts, topics, thresh):

    topic_sims = calculate_topics_similarity(topics)
    clusters = cluster_by_similarity(topic_sims, thresh=thresh)
    new_counts, new_labels = update_counts_labels_by_cluster(counts, topics, clusters)
    return new_counts, new_labels, clusters


def cluster_by_similarity(similarity_matrix, thresh=0.15):
    """
    Creates clusters based on pairwise similarity.  Two vectors are in the same cluster
    if they has similarity larger than a threshhold.
    :param similarity_matrix:
    :param thresh:
    :return: a list of clusters defined by ids
    """
    logging.info("Calculating most probable clusters with threshhold %d " % thresh)
    n_topics = len(similarity_matrix)
    clusters = {}
    n_clusters = 0
    for i in range(n_topics):
        if i not in clusters:
            clusters[i] = n_clusters
            n_clusters += 1
        for j in range(n_topics):
            if i != j:
                if similarity_matrix[i][j] > thresh or similarity_matrix[j][i] > thresh:
                    clusters[j] = clusters[i]
    inv_clusters = {}
    for k, v in clusters.iteritems():
        inv_clusters[v] = inv_clusters.get(v, [])
        inv_clusters[v].append(k)

    logging.info("Clusters found: %s " % list(inv_clusters.itervalues()))
    return list(inv_clusters.itervalues())


def update_counts_labels_by_cluster(counts, topics, clusters):

    n_clusters = len(clusters)

    new_counts = np.zeros((n_clusters, counts.shape[1]))
    new_labels = [[] for _ in range(len(clusters))]

    n_words = 20
    # sort the words in topics by weight
    for i in range(len(topics)):
        topics[i] = sorted(topics[i], key=itemgetter(1))

    for i, cluster in enumerate(clusters):
        for j in cluster:
            new_counts[i, :] += counts[j, :]
            # augment the label only if there is place
            if len(new_labels[i]) < n_words:
                new_labels[i] += [word for word, _ in topics[j]]

        new_labels[i] = set(new_labels[i])

    return new_counts, new_labels

# **************** Similarity relating functions ******************************


def calculate_topics_similarity(topics):
    """
    Calculates mutual similarity between topics, and tries to find clusters
    :param topics: a list of topics defined by tuples (word, weight)
    :return:
    """

    # Create a topic word corpus - each topic is one text
    topics_txt = [[word for word, _ in topic] for topic in topics]
    topics_dict = Dictionary(topics_txt)
    # Create the BOW model for topics
    bow_topics = [topics_dict.doc2bow(text) for text in topics_txt]

    # Inverse dictionary lookup for topic_corpus dictionary
    id2token = dict((v, k) for k, v in topics_dict.token2id.iteritems())

    # Update the counts with weights from the topic definition
    new_bow_topics = []
    for i, text in enumerate(bow_topics):
        bow_weights = []
        weight_dict = dict(topics[i])
        for word_id, count in text:
            token = id2token[word_id]
            weight = float(weight_dict[token])
            bow_weights.append((word_id, count*weight))
        new_bow_topics.append(bow_weights)

    tfidf_model = TfidfModel(new_bow_topics, id2word=id2token, dictionary=topics_dict)
    topics_tfidf_data = corpus2dense(tfidf_model[new_bow_topics], num_terms=len(topics_dict),
                                     num_docs=len(bow_topics)).T

    # Calculate pairwise cosine similarity between topics.
    topic_similarities = cosine_similarity(topics_tfidf_data)

    logging.info("Topic similarities: %s" % topic_similarities)
    return topic_similarities

__author__ = 'verasazonova'

import numpy as np

from six import string_types
from tweet_mining.utils import textutils as tu
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Word2Vec, Doc2Vec
import re
import collections
import os.path

import logging


# **************** W2V relating functions ******************************

def make_w2v_model_name(dataname, size, window, min_count, corpus_length):
    return "w2v_model_%s_%i_%i_%i_%.2g" % (dataname, size, window, min_count, corpus_length)


def load_w2v(w2v_model_name):
    if os.path.isfile(w2v_model_name):
        w2v_model = Word2Vec.load(w2v_model_name)
        logging.info("Model %s loaded" % w2v_model_name)
        return w2v_model
    return None


def build_word2vec(text_corpus, size=100, window=10, min_count=2, dataname="none", shuffle=False):
    """
    Given a text corpus build a word2vec model
    :param size:
    :param window:
    :param dataname:
    :return:
    """

    #w2v_model = Word2Vec(sentences=text_corpus, size=size, alpha=0.025, window=window, min_count=min_count, iter=20,
    #                     sample=1e-3, seed=1, workers=4, hs=1, min_alpha=0.0001, sg=1, negative=1e-4, cbow_mean=0)

    if shuffle:
        w2v_model = Word2Vec(size=size, alpha=0.05, window=window, min_count=min_count, iter=1,
                             sample=1e-3, seed=1, workers=4, hs=1, min_alpha=0.0001, sg=1, cbow_mean=0)
        w2v_model.build_vocab(text_corpus)
        w2v_model.iter = 1
        for epoch in range(20):
            perm = np.random.permutation(text_corpus.shape[0])
            w2v_model.train(text_corpus[perm])
    else:
        w2v_model = Word2Vec(sentences=text_corpus, size=size, alpha=0.05, window=window, min_count=min_count, iter=20,
                             sample=1e-3, seed=1, workers=4, hs=1, min_alpha=0.0001, sg=1, cbow_mean=0)

    logging.info("%s" % w2v_model)
    w2v_model_name = make_w2v_model_name(dataname, size, window, min_count, len(text_corpus))
    w2v_model.save(w2v_model_name)

    return w2v_model


# test the quality of the w2v model by extracting mist similar words to ensemble of words
def test_word2vec(w2v_model, word_list=None, neg_list=None):
    if word_list is None or not word_list:
        return []
    else:
        pos_list = [word for word in word_list if word in w2v_model]
        if neg_list is None or not neg_list:
            list_similar = w2v_model.most_similar_cosmul(positive=pos_list)
        else:
            neg_list_checked = [word for word in neg_list if word in w2v_model]
            list_similar = w2v_model.most_similar_cosmul(positive=pos_list, negative=neg_list_checked, topn=10)
        return list_similar
#------------------------------


def make_d2v_model_name(dataname, size, window, type_str):
    return "d2v_model_%s_%s_%i_%i" % (dataname, type_str, size, window)


def build_doc2vec(dataset, size=100, window=10, dataname="none"):
    """
    Given a text corpus build a word2vec model
    :param size:
    :param window:
    :param dataname:
    :return:
    """

    # dataset a class of KenyaCSMessage, a list of tweets, sorted by date.

    # Extract date and text.
    # Clean, tokenize it
    # Build a BOW model.
    text_pos = dataset.text_pos
    id_pos = dataset.id_pos
    data = np.array(dataset.data)

    text_data, text_dict, text_bow = tu.process_text(data[:, text_pos], stoplist=dataset.stoplist, keep_all=True)

    labeled_text_data = np.array([LabeledSentence(tweet, ["id_"+str(id_str)])
                                  for tweet, id_str in zip(text_data, data[:, id_pos])])

    logging.info("Text processed")
    logging.info("Building d2v ")

    d2v_model_dm = Doc2Vec(min_count=1, window=window, size=size, sample=1e-3, negative=5, workers=4)
    d2v_model_dbow = Doc2Vec(min_count=1, window=window, size=size, sample=1e-3, negative=5, dm=0, workers=4)

    #build vocab over all reviews
    d2v_model_dm.build_vocab(labeled_text_data)
    d2v_model_dbow.build_vocab(labeled_text_data)

    #We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
    for epoch in range(10):
        perm = np.random.permutation(labeled_text_data.shape[0])
        d2v_model_dm.train(labeled_text_data[perm])
        d2v_model_dbow.train(labeled_text_data[perm])

    d2v_model_dm_name = make_d2v_model_name(dataname, size, window, 'dm')
    d2v_model_dbow_name = make_d2v_model_name(dataname, size, window, 'dbow')
    d2v_model_dm.save(d2v_model_dm_name)
    d2v_model_dbow.save(d2v_model_dbow_name)

    return d2v_model_dm, d2v_model_dbow

#------------------------------


def vectorize_tweet(w2v_model, tweet, dpgmm=None):
    size = w2v_model.layer1_size

    # allow one word vectorization without encapsulation by an array
    if isinstance(tweet, string_types):
        tweet = [tweet]

    # get a matrix of w2v vectors
    vec_list = [w2v_model[word].reshape((1, size)) for word in tweet if word in w2v_model]

    if not vec_list:
        data = np.zeros(size).reshape((1, size))
    else:
        data = np.concatenate(vec_list)

    # [avg, std, diff]
    features = [np.median(data, axis=0), np.std(data, axis=0), np.median(np.diff(data, axis=1), axis=0)]

    # [cluster info]
    features.append(dpgmm.transform(tweet))
    vec = np.concatenate(features)

    vec = vec.reshape((1, len(vec)))

    return vec


def vectorize_tweet_corpus(w2v_model, tweet_corpus, dpgmm=None):
    logging.info("Vectorizing a corpus")
    size = w2v_model.layer1_size
    if len(tweet_corpus) > 0:
        vecs = np.concatenate([vectorize_tweet(w2v_model, z, dpgmm=dpgmm) for z in tweet_corpus])
    else:
        vecs = np.zeros(size).reshape((1, size))
    return vecs
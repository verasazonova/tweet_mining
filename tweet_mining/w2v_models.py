__author__ = 'verasazonova'

import numpy as np
import os
import os.path

from six import string_types
from sklearn.preprocessing import StandardScaler
import codecs

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import skew
from sklearn.mixture import DPGMM
import pickle
from tweet_mining.utils import textutils as tu
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Word2Vec, Doc2Vec
import re
import collections

from gensim.corpora import Dictionary

import logging


# **************** W2V relating functions ******************************

def make_w2v_model_name(dataname, size, window, min_count, corpus_length):
    return "w2v_model_%s_%i_%i_%i_%.2g" % (dataname, size, window, min_count, corpus_length)


def make_d2v_model_name(dataname, size, window, type_str):
    return "d2v_model_%s_%s_%i_%i" % (dataname, type_str, size, window)


def build_word2vec(text_corpus, size=100, window=10, min_count=2, dataname="none"):
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

    #text_data, text_dict, text_bow = tu.process_text(text_corpus, stoplist=None, keep_all=True)
    #logging.info("Text processed")
    #logging.info("Building w2v ")

    #w2v_model = Word2Vec(sentences=text_corpus, size=size, alpha=0.025, window=window, min_count=min_count, iter=20,
    #                     sample=1e-3, seed=1, workers=4, hs=1, min_alpha=0.0001, sg=1, negative=1e-4, cbow_mean=0)


    w2v_model = Word2Vec(sentences=text_corpus, size=size, alpha=0.05, window=window, min_count=min_count, iter=20,
                         sample=1e-3, seed=1, workers=4, hs=1, min_alpha=0.0001, sg=1, negative=0, cbow_mean=0)
    logging.info("%s" % w2v_model)
    w2v_model_name = make_w2v_model_name(dataname, size, window, min_count, len(text_corpus))
    w2v_model.save(w2v_model_name)

    return w2v_model


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


def load_w2v(w2v_model_name):
    if os.path.isfile(w2v_model_name):
        w2v_model = Word2Vec.load(w2v_model_name)
        logging.info("Model %s loaded" % w2v_model_name)
        return w2v_model
    return None


def apply_w2v(word_list, w2v_model=None):
    """
    If the W2v model exists, apply it to the BOW corpus and return topic assignments
    :param word_list: list of words to investigate
    :param w2v_model: lda_model
    :return: a list of topic assignments
    """

    if w2v_model is not None:
        for word in word_list:
            if word in w2v_model:
                print "%s:\t\t%s" % (word,
                                     ", ".join([word for word, _ in w2v_model.most_similar(positive=[word], topn=10)]))


def test_w2v(w2v_model=None, word_list=None, neg_list=None):

    if w2v_model is None:
        logging.info("No model supplied")
        return None

    if word_list is not None:
        word_list = [word for word in word_list if word in w2v_model]
        print " ".join(word_list),
    else:
        word_list = []
    if neg_list is not None:
        print " - " + " ".join(neg_list),
        print ":"
        neg_list = [word for word in neg_list if word in w2v_model]
    else:
        neg_list = []

    if word_list:
        return w2v_model.most_similar_cosmul(positive=word_list, negative=neg_list, topn=10)
    else:
        return []


def create_word_vecs(word_list, size=100, w2v_model=None):
    word_vecs = np.zeros((len(word_list), size))
    for i, word in enumerate(word_list):
        if word in w2v_model:
            word_vecs[i] = w2v_model[word]
    return word_vecs


def vectorize_tweet_old(w2v_model, tweet, weight_dict=None):
    size = w2v_model.layer1_size
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    # allow one word vectorization without encapsulation by an array
    if isinstance(tweet, string_types):
        tweet = [tweet]
    for word in tweet:
        try:
            if weight_dict is None:
                vec += w2v_model[word].reshape((1, size))
                count += 1.
            else:
                if word in weight_dict:
                    weight = weight_dict[word]
                else:
                    weight = 1
                vec += w2v_model[word].reshape((1, size)) * weight
                count += weight

        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def vectorize_tweet(w2v_model, tweet, type="avg", clusterers=None, scaler=None):
    size = w2v_model.layer1_size
    #vec_size = 2*size
    #vec = np.zeros(size).reshape((1, size))
    #count = 0.
    # allow one word vectorization without encapsulation by an array
    if isinstance(tweet, string_types):
        tweet = [tweet]

    # get a matrix of w2v vectors
    vec_list = [w2v_model[word].reshape((1,size)) for word in tweet if word in w2v_model]

    if not vec_list:
        data = np.zeros(size).reshape((1, size))

    else:
        data = np.concatenate(vec_list)

    if type == "std":
        vec = np.concatenate([data.mean(axis=0), data.std(axis=0)])  #, skew(data, axis=0)])
    elif type == "cluster":
        features = [data.mean(axis=0), data.std(axis=0)]
        for clusterer in clusterers:
            predictions = clusterer.predict(scaler.transform(data))
            features.append(np.bincount(predictions, minlength=clusterer.n_components))
        vec = np.concatenate(features)
    elif type == "sim":
        #distances = cosine_similarity(data)
        features = [data.mean(axis=0), data.std(axis=0), len(vec_list)]
                    #[distances.mean()],
                    #[distances.std()], [np.amin(distances)], [np.amax(distances)]]
        #for clusterer in clusterers:
        #    predictions = clusterer.predict(data)
        #    features.append(np.bincount(predictions, minlength=clusterer.n_components))
        vec = np.concatenate(features)
    else:
        vec = np.concatenate([data.mean(axis=0)])  #, skew(data, axis=0)])

    vec = vec.reshape((1, len(vec)))

    return vec


def vectorize_tweet_corpus(w2v_model, tweet_corpus, weight_dict=None, dictionary=None, tfidf=None, type=None,
                           clusterers=None, scaler=None):
    logging.info("Vectorizing a corpus with %s" % type)
    size = w2v_model.layer1_size
    if len(tweet_corpus) > 0:
        if dictionary is not None:
            vecs = np.concatenate([vectorize_bow_text(w2v_model, z, dictionary, tfidf=tfidf) for z in tweet_corpus])
        else:
            vecs = np.concatenate([vectorize_tweet(w2v_model, z, type=type, clusterers=clusterers, scaler=scaler) for z in tweet_corpus])
    else:
        vecs = np.zeros(size).reshape((1, size))
    return vecs


def vectorize_bow_text(w2v_model, text, dictionary, tfidf=None):
    size = w2v_model.layer1_size
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    if tfidf is not None:
        bow = tfidf[dictionary.doc2bow(text)]
    else:
        bow = dictionary.doc2bow(text)
    for weight, word_id in bow:
        try:
            word = dictionary[word_id]
            vec += w2v_model[word].reshape((1, size)) * weight
            count += weight
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def build_dpgmm_model(w2v_corpus, w2v_model=None, n_components=None, no_above=0.8, no_below=2, dataname=""):

    dictionary = Dictionary(w2v_corpus)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=9000)

    size = w2v_model.layer1_size

    word_list = [word for word in dictionary.token2id.iterkeys() if word in w2v_model]

    # saving word representations
    with codecs.open("%s_%f_%f_w2v_rep.txt" % (dataname, no_above, no_below), 'w', encoding="utf-8") as fout:
        for word in word_list:
            fout.write(word + "," + ",".join(["%.8f" % x for x in w2v_model[word]]) + "\n")


    vec_list = [w2v_model[word] for word in word_list]
    scaler = StandardScaler()
    scaler.fit(np.array(vec_list))
    vecs = scaler.transform(np.array(vec_list))

    cluster__name_base = "dpggm_model"

    cluster_model_names = []

    for n_comp in [n_components]:
        cluster_model_name = "%s-%i" % (cluster__name_base, n_comp)

        cluster = DPGMM(n_components=n_comp, covariance_type='diag', alpha=5, n_iter=1000, verbose=0)
        cluster.fit(vecs)
        y_ = cluster.predict(vecs)
        for i, cluster_center in enumerate(cluster.means_):
            cluster_x = vecs[y_ == i]
            cluster_x_size = len(cluster_x)
            if cluster_x_size > 0:
                central_words = [word for word, _ in w2v_model.most_similar_cosmul(positive=[cluster_center], topn=5)]
                print "%i, %i   : %s" % (i, cluster_x_size, repr(central_words))

        pickle.dump(cluster, open(cluster_model_name, 'wb'))
        logging.info("Clusterer build %s" % cluster)

        cluster_model_names.append(cluster_model_name)

    scaler_model_name = cluster__name_base+"-scaler"
    pickle.dump(scaler, open(scaler_model_name, 'wb'))

    return cluster_model_names, scaler_model_name


def save_w2v_words_representations(w2v_model, corpus, no_above=0.9, no_below=2, dataname=""):

    dictionary = Dictionary(corpus)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=9000)

    word_list = [word for word in dictionary.token2id.iterkeys() if word in w2v_model]

    with open("%s_%f_%f_w2v_rep.txt" % (dataname, no_above, no_below), 'w') as fout:
        for word in word_list:
            fout.write(word + "," + ",".join(w2v_model[word])+"\n")



# **************** Spell checking relating functions ******************************

def words(text):
    return re.findall('[a-z]+', text.lower())


def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


#big_file = "/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/big.txt"
NWORDS = [] #train(words(file(big_file).read()))
alphabet = 'abcdefghijklmnopqrstuvwxyz'


def edits1(word):
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts    = [a + c + b     for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)


def levenshtein(s, t):
    ''' From Wikipedia article; Iterative with two matrix rows. '''
    if s == t: return 0
    elif len(s) == 0: return len(t)
    elif len(t) == 0: return len(s)
    v0 = [None] * (len(t) + 1)
    v1 = [None] * (len(t) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(s)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]

    return v1[len(t)]


def ismisspelled(word1, word2):
    if levenshtein(word1, word2) <= 1:
        return True
    return False


def check_spelling(w2v_model=None, dictionary=None):
    if w2v_model is None or dictionary is None:
        logging.error("No model or dictionary supplied")
        return None

    w2id = dict([ (w, dictionary.token2id[w] ) for w in dictionary.token2id.keys() if re.match('[a-z]+', w)])
    print len(w2id)
    idlen = np.ones((len(dictionary), 1))

    for word1 in w2id.keys():
        id = w2id[word1]
        if word1 in w2v_model:
            close_words = [word for word, sim in w2v_model.most_similar([word1], topn=10) if sim > 0.6 ]
            for word2 in close_words:
                if word2 in w2id:
                    id2 = w2id[word2]
                    if ismisspelled(word1, word2):
                        if idlen[id] < idlen[id2]:
                            id = id2
                        w2id[word1] = id
                        w2id[word2] = id
                        idlen[id] += 1
                        print id, word1, word2, idlen[id]

#    for k,v in groupby(sorted(w2id.items()),key=itemgetter(0)):
#        print k,list(v)



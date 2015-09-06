__author__ = 'verasazonova'

import numpy as np

from six import string_types
from sklearn.preprocessing import StandardScaler
import codecs

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


def build_word2vec(text_corpus, size=100, window=10, min_count=2, dataname="none", shuffle=False):
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
def test_w2v(w2v_model, word_list=None, neg_list=None):
    if word_list is None or not word_list:
        return []
    elif neg_list is None or not neg_list:
        list_similar = w2v_model.most_similar_cosmul(positive=word_list)
    else:
        list_similar = w2v_model.most_similar_cosmul(positive=word_list, negative=neg_list)
    return list_similar


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


def vectorize_tweet(w2v_model, tweet, dpgmm=None, scaler=None):
    size = w2v_model.layer1_size
    #vec_size = 2*size
    #vec = np.zeros(size).reshape((1, size))
    #count = 0.
    # allow one word vectorization without encapsulation by an array
    if isinstance(tweet, string_types):
        tweet = [tweet]

    # get a matrix of w2v vectors
    vec_list = [w2v_model[word].reshape((1, size)) for word in tweet if word in w2v_model]

    if not vec_list:
        data = np.zeros(size).reshape((1, size))

    else:
        data = np.concatenate(vec_list)

    #data_shifted[:,0:2] = 0
    features = [np.median(data, axis=0), np.std(data, axis=0), np.median(np.diff(data, axis=1), axis=0)]
    predictions = dpgmm.predict(scaler.transform(data))
    features.append(np.bincount(predictions, minlength=dpgmm.n_components))
    vec = np.concatenate(features)

    vec = vec.reshape((1, len(vec)))

    return vec


def vectorize_tweet_corpus(w2v_model, tweet_corpus, dictionary=None, tfidf=None,
                           dpgmm=None, scaler=None):
    logging.info("Vectorizing a corpus")
    size = w2v_model.layer1_size
    if len(tweet_corpus) > 0:
        if dictionary is not None:
            vecs = np.concatenate([vectorize_bow_text(w2v_model, z, dictionary, tfidf=tfidf) for z in tweet_corpus])
        else:
            vecs = np.concatenate([vectorize_tweet(w2v_model, z, dpgmm=dpgmm, scaler=scaler) for z in tweet_corpus])
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


def build_dpgmm_model(w2v_corpus, w2v_model=None, n_components=None, no_above=0.9, no_below=8, dataname="",
                      stoplist=None):

    dictionary = Dictionary(w2v_corpus)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=9000)

    # construct a list of words to cluster
    # remove rare and frequent words
    # remove words of length 1
    # remove stopwords
    word_list = np.array([word for word in dictionary.token2id.iterkeys()
                          if word in w2v_model and len(word) > 1 and (stoplist is None or word not in stoplist)])

    # saving word representations
    # word, w2v vector
    with codecs.open("w2v_vocab_%s_%f_%f.lcsv" % (dataname, no_above, no_below), 'w', encoding="utf-8") as fout:
        for word in word_list:
            fout.write(word + "," + ",".join(["%.8f" % x for x in w2v_model[word]]) + "\n")

    vec_list = [w2v_model[word] for word in word_list]
    scaler = StandardScaler()
    vecs = scaler.fit_transform(np.array(vec_list))

    cluster_name_base = "dpggm_model_%s" % dataname

    cluster_model_name = "%s-%i" % (cluster_name_base, n_components)

    clusterer = DPGMM(n_components=n_components, covariance_type='diag', alpha=10, n_iter=100, tol=0.0001)
    clusterer.fit(vecs)
    print clusterer.converged_
    y_ = clusterer.predict(vecs)
    with codecs.open("clusters_%s_%i_%.2f_%.0f.txt" % (dataname, n_components, no_above, no_below), 'w',
                     encoding="utf-8") as fout:
        for i, cluster_center in enumerate(clusterer.means_):
            cluster_x = vecs[y_ == i]
            words = word_list[y_ == i]
            cluster_x_size = len(cluster_x)
            if cluster_x_size > 0:
                cluster_center_original = scaler.inverse_transform(cluster_center)
                central_words = [word for word, _ in w2v_model.most_similar_cosmul(positive=[cluster_center_original],
                                                                                   topn=cluster_x_size) if word in words]

                fout.write("%2i, %5i,   : " % (i, cluster_x_size))
                for j, word in enumerate(central_words):
                    if j < 10:
                        fout.write("%s " % word)
                fout.write("\n")

    pickle.dump(clusterer, open(cluster_model_name, 'wb'))
    logging.info("Clusterer build %s" % clusterer)

    scaler_model_name = cluster_name_base+"-scaler"
    pickle.dump(scaler, open(scaler_model_name, 'wb'))

    return clusterer, scaler


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



__author__ = 'verasazonova'

import logging
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import utils.textutils as tu
import utils.ioutils as io
from sklearn.mixture import DPGMM
from sklearn.preprocessing import StandardScaler
from gensim import corpora, models, matutils
import re
import w2v_models


# A BOW Model encompassing sklearn transformer interface
class BOWModel(BaseEstimator, TransformerMixin):
    def __init__(self, no_below=2, no_above=0.9, stoplist=None):
        self.no_below = no_below
        self.no_above = no_above
        self.dictionary = None
        self.tfidf = None
        self.stoplist = stoplist
        logging.info("BOW classifier: initialized with no_below %s and no_above %s and stoplist %s "
                     % (self.no_below, self.no_above, self.stoplist))

    # build dictionary and the tfidf model for the data
    # X is an array of non-tokenized sentences
    def fit(self, X, y=None):
        x_clean = tu.clean_and_tokenize(X, stoplist=self.stoplist)
        self.dictionary = corpora.Dictionary(x_clean)
        self.dictionary.filter_extremes(no_above=self.no_above, no_below=self.no_below)
        self.tfidf = models.TfidfModel([self.dictionary.doc2bow(text) for text in x_clean],
                                       id2word=self.dictionary, normalize=True)
        return self

    # return tfidf model for X
    def transform(self, X):
        x_clean = tu.clean_and_tokenize(X, stoplist=self.stoplist)
        x_tfidf = self.tfidf[[self.dictionary.doc2bow(text) for text in x_clean]]
        x_data = matutils.corpus2dense(x_tfidf, num_terms=len(self.dictionary)).T
        logging.info("Returning data of shape %s " % (x_data.shape,))
        return x_data


# A class encompassing a W2V representation for texts through a sklearn transforme interface
class W2VTextModel(BaseEstimator, TransformerMixin):

    def __init__(self, w2v_model=None, dpgmm=None, scaler=None, no_below=2, no_above=0.9, stoplist=None):
        self.w2v_model = w2v_model
        self.dpgmm = dpgmm
        self.dictionary = None
        self.no_above = no_above
        self.no_below = no_below
        self.stoplist = stoplist

        self.no_dictionary = False
        logging.info("W2v based text classifier with model %s and dpgmm %s " % (self.w2v_model, self.dpgmm))

    # the w2v model should be provided if not it will be built from the input data
    # builds dictionary to filter out frequent and rare words
    # loads a scaler for clustering (the clusters and scaler should be built)
    def fit(self, X, y=None):

        logging.info("Loaded from file")

        x_clean = [tu.normalize_punctuation(text).split() for text in X]

        if self.w2v_model is None:
            self.w2v_model = w2v_models.build_word2vec(x_clean, size=100, window=10, min_count=1, dataname="test")

        if self.dpgmm is None:
            logging.info("No dpgmm provided - building")
            #self.dpgmm, self.scaler = w2v_models.build_dpgmm_model(x_clean, w2v_model=self.w2v_model,
            #                                                       n_components=30,
            #                                                       stoplist=self.stoplist)
            self.dpgmm = DPGMMClusterModel(self.w2v_model, n_components=30, dataname="test", stoplist=self.stoplist)

        if self.no_below == 1 and self.no_above == 1:
            self.no_dictionary = True
        else:
            self.dictionary = corpora.Dictionary(x_clean)
            self.dictionary.filter_extremes(no_above=self.no_above, no_below=self.no_below)

        logging.info("W2V: got a model %s " % (self.w2v_model,))
        logging.info("W2V: got a cluster %s " % (self.dpgmm,))
        return self

    # X is an array of sentences (texts)
    # Pre-process, filter frequent and rare words, and vectorize
    def transform(self, X):

        # Text pre-processing
        x_clean = [tu.normalize_punctuation(text).split() for text in X]
        logging.info("Text prepocessed")

        # Text processing: remove words outside the dictionary frequency boundaries
        # To check whether word is in the dictionary need to convert it to id first!!!!
        if self.no_dictionary:
            x_processed = x_clean
        else:
            x_processed = [[word for word in text if word in self.dictionary.token2id and word not in self.stoplist]
                            for text in x_clean]

        # Vectorize using W2V model, and clusterer
        logging.info("Clusterer %s " % self.dpgmm)
        if self.w2v_model is not None:
            x_vector = w2v_models.vectorize_tweet_corpus(self.w2v_model, x_processed, dpgmm=self.dpgmm)
            logging.info("W2V Averaged: returning pre-processed data of shape %s" % (x_vector.shape, ))
        else:
            logging.info("W2V Averaged: no model was provided.")
            x_vector = np.zeros((len(X), 1))
        return x_vector


class DPGMMClusterModel(BaseEstimator, TransformerMixin):

    def __init__(self, w2v_model, n_components=None, no_above=0.9, no_below=8, dataname="", stoplist=None,
                 dictionary=None, recluster_thresh=1000):
        self.w2v_model = w2v_model
        self.no_above = no_above
        self.no_below = no_below
        self.n_components = n_components
        self.stoplist = stoplist
        self.dataname = dataname
        self.dictionary = dictionary
        self.dpgmm = None
        self.scaler = None
        self.cluster_info = None
        # a list of sub-clusterer
        self.subdpgmms = []
        self.reclustered = []
        self.recluster_thresh = recluster_thresh

    def should_cluster_word(self, word):
        return (word in self.w2v_model) and (word in self.dictionary.token2id) and (len(word) > 1) and \
               (self.stoplist is None or word not in self.stoplist)


    def fit(self, X, y=None):
        # either consturct a dictionary from X, trim it
        if self.dictionary is None:
            self.dictionary = corpora.Dictionary(X)
            word_list = np.array([word for word in self.dictionary.token2id.iterkeys() if self.should_cluster_word(word)])
        # or use an existing dictionary and trim the given set of words
        else:
            word_list = np.array([word for word in X if self.should_cluster_word(word)])

        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above, keep_n=9000)

        # construct a list of words to cluster
        # remove rare and frequent words
        # remove words of length 1
        # remove stopwords
        vec_list = [self.w2v_model[word] for word in word_list]

        logging.info("DPGMM received %i words" % len(vec_list))

        # save word representations
        filename = "w2v_vocab_%s_%.1f_%.0f.lcsv" % (self.dataname, self.no_above, self.no_below)
        io.save_words_representations(filename, word_list, vec_list)

        self.scaler = StandardScaler()
        vecs = self.scaler.fit_transform(np.array(vec_list))

        self.dpgmm = DPGMM(n_components=self.n_components, covariance_type='diag', alpha=10, n_iter=100, tol=0.0001)
        self.dpgmm.fit(vecs)
        logging.info("DPGMM converged: %s" % self.dpgmm.converged_)

        # save information about found clusters
        self.cluster_info = []
        y_ = self.dpgmm.predict(vecs)
        for i, cluster_center in enumerate(self.dpgmm.means_):
            cluster_words = word_list[y_ == i]
            cluster_size = len(cluster_words)
            if cluster_size > self.recluster_thresh and self.recluster_thresh > 0:
                logging.info("DPGMM: reclustering %i words for cluster %i" % (len(cluster_words), i))
                sub_dpgmm = DPGMMClusterModel(self.w2v_model, n_components=self.n_components,
                                              dictionary=self.dictionary,
                                              dataname="%s-%i" % (self.dataname, i), stoplist=self.stoplist)
                sub_dpgmm.fit(cluster_words)
                self.subdpgmms.append(sub_dpgmm)
                self.reclustered.append(i)
            if cluster_size > 0:
                cluster_center_original = self.scaler.inverse_transform(cluster_center)
                similar_words = self.w2v_model.most_similar_cosmul(positive=[cluster_center_original], topn=cluster_size)
                central_words = [word for word, _ in similar_words if word in cluster_words]
            else:
                central_words = []
            self.cluster_info.append({'cnt': i, 'size': cluster_size, 'words': central_words})

        filename = "clusters_%s_%i_%.1f_%.0f.txt" % (self.dataname, self.n_components, self.no_above, self.no_below)
        io.save_cluster_info(filename, self.cluster_info)

        return self

    def transform(self, X):

        word_list = np.array([word for word in X if self.should_cluster_word(word)])
        vec_list = [self.w2v_model[word] for word in word_list]

        if vec_list:
            # assign words to clusters
            predictions = self.dpgmm.predict(self.scaler.transform(np.array(vec_list)))
            global_bincount = np.bincount(predictions, minlength=self.dpgmm.n_components)
            # re-assign words in large clusters
            bincounts = [global_bincount]
            for i, subdpgmm in zip(self.reclustered, self.subdpgmms):
                words_torecluster = word_list[predictions == i]
                bincounts.append(subdpgmm.transform(words_torecluster))

            vec = np.concatenate(bincounts)

        else:
            vec = np.zeros((self.dpgmm.n_components*(len(self.reclustered)+1),))

        return vec

# A class that augments a given text using similar words from a W2V
# model.  Follows sklearn transform interface
class W2VAugmentModel(BaseEstimator, TransformerMixin):

    def __init__(self, topn=None, w2v_model=None):
        self.topn = topn
        self.w2v_model = w2v_model
        logging.info("W2v stacked classifier")

    def fit(self, X, y=None):
        logging.info("W2V: building a model")
        logging.info("W2V: model assigned %s with topn %s" % (self.w2v_model, self.topn))
        return self

    def transform(self, X):
        logging.info("W2V Augmented: augmenting the text %s" % (X.shape, ))
        if self.w2v_model is None:
            augmented_corpus = X[:]
        else:
            augmented_corpus = []
            for text in X:
                words_in_model = [word for word in text if word in self.w2v_model]
                augmented_text = [word for word in text]
                sim_words = [re.sub(r"\W", "_", tup[0]) for
                             tup in self.w2v_model.most_similar(positive=words_in_model, topn=self.topn)
                             if tu.word_valid(tup[0])]
                augmented_text += sim_words
                augmented_corpus.append(augmented_text)
        a = np.array(augmented_corpus)
        print a.shape
        return a


# a class builds LDA topic assingments.
# Follows sklearn transformers interface
class LDAModel(BaseEstimator, TransformerMixin):

    def __init__(self, topn=None, no_below=1, no_above=1, mallet=True):
        self.topn = topn
        self.no_above = no_above
        self.no_below = no_below
        self.mallet = mallet
        self.mallet_path = "/Users/verasazonova/no-backup/JARS/mallet-2.0.7/bin/mallet"
        self.dictionary = None
        self.model = None

    def fit(self, X, y=None):
        self.dictionary = corpora.Dictionary(X)
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)

        bow_corpus = [self.dictionary.doc2bow(text) for text in X]

        if self.mallet:
            self.model = models.LdaMallet(self.mallet_path, corpus=bow_corpus, num_topics=self.topn,
                                          id2word=self.dictionary, workers=4,
                                          optimize_interval=10, iterations=1000)
        else:
            self.model = models.LdaModel(bow_corpus, id2word=self.dictionary, num_topics=self.topn,
                                         distributed=False,
                                         chunksize=2000, passes=1, update_every=5, alpha='auto',
                                         eta=None, decay=0.5, eval_every=10, iterations=50, gamma_threshold=0.001)

        for topic in self.model.show_topics(num_topics=self.topn, num_words=20, formatted=False):
            for word in topic:
                print word[1] + " ",
            print ""
        return self


    def transform(self, X):

        bow_corpus = [self.dictionary.doc2bow(text) for text in X]
        x_data = matutils.corpus2dense(self.model[bow_corpus], num_terms=self.topn).T
        return x_data
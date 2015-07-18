__author__ = 'verasazonova'

import logging
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import utils.textutils as tu
from sklearn.mixture import DPGMM
from sklearn.preprocessing import scale
from gensim import corpora, models, matutils
import re
import w2v_models


class BOWModel(BaseEstimator, TransformerMixin):
    def __init__(self, no_below=2, no_above=0.9, stoplist=None):
        self.no_below = no_below
        self.no_above = no_above
        self.dictionary = None
        self.tfidf = None
        self.stoplist = stoplist
        logging.info("BOW classifier: initialized with no_below %s and no_above %s and stoplist %s "
                     % (self.no_below, self.no_above, self.stoplist))


    def fit(self, X, y=None):
        x_clean = tu.clean_and_tokenize(X, stoplist=self.stoplist)
        self.dictionary = corpora.Dictionary(x_clean)
        self.dictionary.filter_extremes(no_above=self.no_above, no_below=self.no_below)
        self.tfidf = models.TfidfModel([self.dictionary.doc2bow(text) for text in x_clean],
                                       id2word=self.dictionary, normalize=True)
        return self


    def transform(self, X):
        x_clean = tu.clean_and_tokenize(X, stoplist=self.stoplist)
        x_tfidf = self.tfidf[[self.dictionary.doc2bow(text) for text in x_clean]]
        x_data = matutils.corpus2dense(x_tfidf, num_terms=len(self.dictionary)).T
        logging.info("Returning data of shape %s " % (x_data.shape,))
        return x_data

class W2VWeightedAveragedModel(BaseEstimator, TransformerMixin):

    def __init__(self, w2v_model=None, dictionary=None, no_below=2, no_above=0.9, stoplist=None):
        self.w2v_model = w2v_model
        self.dictionary = dictionary
        self.no_above = no_above
        self.no_below = no_below
        self.stoplist = stoplist
        logging.info("W2v averaged classifier %s " % self.w2v_model)

    def fit(self, X, y=None):
        x_clean = [tu.normalize_punctuation(text).split() for text in X]

        if self.w2v_model is None:
            self.w2v_model = w2v_models.build_word2vec(x_clean, size=100, window=10, min_count=1, dataname="test")

        if self.dictionary is None:
            self.dictionary = corpora.Dictionary(x_clean)
            self.dictionary.filter_extremes(no_above=self.no_above, no_below=self.no_below)

        self.tfidf = models.TfidfModel([self.dictionary.doc2bow(text) for text in x_clean],
                                       id2word=self.dictionary, normalize=True)

        logging.info("W2V: got a model %s " % (self.w2v_model,))
        return self

    # X is an array of tweets.  Just text
    def transform(self, X):

        # Text pre-processing
        x_clean = [tu.normalize_punctuation(text).split() for text in X]
        logging.info("Text prepocessed")

        # Text processing: remove words outside the dictionary frequency boundaries
        # To check whether word is in the dictionary need to convert it to id first!!!!
        x_processed = [[word for word in text if word in self.dictionary.token2id and word not in self.stoplist]
                       for text in x_clean]

        # W2V vectors averaging
        if self.w2v_model is not None:
            x_vector = w2v_models.vectorize_tweet_corpus(self.w2v_model, x_processed,
                                                         dictionary=self.dictionary, tfidf=None)
            logging.info("W2V Averaged: returning pre-processed data of shape %s" % (x_vector.shape, ))
        else:
            logging.info("W2V Averaged: no model was provided.")
            x_vector = np.zeros((len(X), 1))
        return x_vector


class W2VAveragedModel(BaseEstimator, TransformerMixin):

    def __init__(self, w2v_model=None, no_below=2, no_above=0.9, stoplist=None, type=None):
        self.w2v_model = w2v_model
        self.dictionary = None
        self.no_above = no_above
        self.no_below = no_below
        self.stoplist = stoplist
        self.type = type
        self.no_dictionary = False
        logging.info("W2v averaged classifier type %s model %s " % (self.type, self.w2v_model))

    def fit(self, X, y=None):
        x_clean = [tu.normalize_punctuation(text).split() for text in X]

        if self.w2v_model is None:
            self.w2v_model = w2v_models.build_word2vec(x_clean, size=100, window=10, min_count=1, dataname="test")

        if self.no_below == 1 and self.no_above == 1:
            self.no_dictionary = True
        else:
            self.dictionary = corpora.Dictionary(x_clean)
            self.dictionary.filter_extremes(no_above=self.no_above, no_below=self.no_below)

        #self.cluster = DPGMM(n_components=30, covariance_type='diag', alpha=5,  n_iter=1000)


        logging.info("W2V: got a model %s " % (self.w2v_model,))
        return self

    # X is an array of tweets.  Just text
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

        # W2V vectors averaging
        if self.w2v_model is not None:
            x_vector = w2v_models.vectorize_tweet_corpus(self.w2v_model, x_processed, type=self.type)
            logging.info("W2V Averaged: returning pre-processed data of shape %s" % (x_vector.shape, ))
        else:
            logging.info("W2V Averaged: no model was provided.")
            x_vector = np.zeros((len(X), 1))
        return x_vector


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


class LDAModel(BaseEstimator, TransformerMixin):

    def __init__(self, topn=None, no_below=1, no_above=1, mallet=True):
        self.topn = topn
        self.no_above = no_above
        self.no_below = no_below
        self.mallet = mallet
        self.mallet_path = "/Users/verasazonova/no-backup/JARS/mallet-2.0.7/bin/mallet"


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

        for topic in  self.model.show_topics(num_topics=self.topn, num_words=20, formatted=False):
            for word in topic:
                print word[1] + " ",
            print ""
        return self


    def transform(self, X):

        bow_corpus = [self.dictionary.doc2bow(text) for text in X]
        x_data = matutils.corpus2dense(self.model[bow_corpus], num_terms=self.topn).T
        return x_data

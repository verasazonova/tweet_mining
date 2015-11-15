__author__ = 'verasazonova'

import logging
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import utils.textutils as tu
import utils.ioutils as io
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import DPGMM
from sklearn.preprocessing import StandardScaler
from gensim import corpora, models, matutils
import re
import gc
from sklearn.cluster import DBSCAN
import w2v_models
from six import string_types


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
        #x_data = matutils.corpus2dense(x_tfidf, num_terms=len(self.dictionary)).T
        #logging.info("Returning data of shape %s " % (len(x_data)))
        #returning a csr matrix
        return x_tfidf


# A class encompassing a W2V representation for texts through a sklearn transforme interface
class W2VTextModel(BaseEstimator, TransformerMixin):

    def __init__(self, w2v_model=None, no_below=2, no_above=0.9, stoplist=None, diffmax0=6, diffmax1=6):
        self.w2v_model = w2v_model
        self.dictionary = None
        self.no_above = no_above
        self.no_below = no_below
        self.stoplist = stoplist
        self.diffmax0 = diffmax0
        self.diffmax1 = diffmax1
        self.feature_crd = {}
        self.length = 0

        self.no_dictionary = False
        logging.info("W2v based text classifier with model %s " % (self.w2v_model))

    # the w2v model should be provided if not it will be built from the input data
    # builds dictionary to filter out frequent and rare words
    # loads a scaler for clustering (the clusters and scaler should be built)
    # X here is a list of texts (tweets).  Every
    def fit(self, X, y=None):

        logging.info("Loaded from file")

        x_clean = [tu.normalize_punctuation(text).split() for text in X]

        if self.w2v_model is None:
            self.w2v_model = w2v_models.build_word2vec(x_clean, size=100, window=10, min_count=1, dataname="test")

        if self.no_below == 1 and self.no_above == 1:
            self.no_dictionary = True
        else:
            self.dictionary = corpora.Dictionary(x_clean)
            self.dictionary.filter_extremes(no_above=self.no_above, no_below=self.no_below)

        # setting the coordinates for different models (start, stop)
        size = self.w2v_model.layer1_size
        self.feature_crd = {'00_avg': (0, size),
                            '01_std': (size, 2*size)}
        feature_cnt = 2
        start = 2*size
        l = size
        for i in range(1,self.diffmax0):
            name = "%02d_diff0_%i" % (feature_cnt, i)
            feature_cnt += 1
            val = (start, start + l)
            self.feature_crd[name] = val
            start += l
            name = "%02d_diff0_std_%i" % (feature_cnt, i)
            feature_cnt += 1
            val = (start, start + l)
            self.feature_crd[name] = val
        for i in range(1,self.diffmax1):
            name = "%02d_diff1_%i" % (feature_cnt, i)
            feature_cnt += 1
            val = (start, start + l)
            self.feature_crd[name] = val
            start += l
            name = "%02d_diff1_std_%i" % (feature_cnt, i)
            feature_cnt += 1
            val = (start, start + l)
            self.feature_crd[name] = val
            start += l
        self.length = start
        logging.info("Total feature length %i " % self.length )
        logging.info("W2V: got a model %s " % (self.w2v_model,))
        return self


    def vectorize_text(self, text):
        size = self.w2v_model.layer1_size

        # allow one word vectorization without encapsulation by an array
        if isinstance(text, string_types):
            text = [text]

        # get a matrix of w2v vectors
        vec_list = [self.w2v_model[word].reshape((1, size)) for word in text if word in self.w2v_model]

        if not vec_list:
            data = np.zeros(size).reshape((1, size))
        else:
            data = np.concatenate(vec_list)

        # [avg, std]
        features = [np.median(data, axis=0), np.std(data, axis=0)]

        for i in range(1, self.diffmax0):
            if len(data) > i:
                data_diff = data - np.roll(data, i, axis=0)
                features.append(np.median(np.diff(data_diff, axis=0), axis=0))
                features.append(np.std(np.diff(data_diff, axis=0), axis=0))
            else:
                features.append(np.zeros((2*size,)))

        # these features are not differences between words, they are differences between columns !!!!!
        # I don't know what it corresponds to
        for i in range(1, self.diffmax1):
            data_diff = data - np.roll(data, i, axis=1)
            features.append(np.median(data_diff, axis=0))
            features.append(np.std(data_diff, axis=0))


        vec = np.concatenate(features)

        return vec.reshape((1, len(vec)))


    def pre_process(self, text):
        if self.no_dictionary:
            x_processed = tu.normalize_punctuation(text).split()
        else:
            x_processed = [word for word in tu.normalize_punctuation(text).split()
                           if word in self.dictionary.token2id and word not in self.stoplist]

        return x_processed


    # X is an array of sentences (texts)
    # Pre-process, filter frequent and rare words, and return an array of dictionary of different feature vectors
    def transform(self, X):

        # Text pre-processing
        #x_clean = [tu.normalize_punctuation(text).split() for text in X]
        #logging.info("Text prepocessed")

        # Text processing: remove words outside the dictionary frequency boundaries
        # To check whether word is in the dictionary need to convert it to id first!!!!
        #if self.no_dictionary:
        #    x_processed = x_clean
        #else:
        #    x_processed = [[word for word in text if word in self.dictionary.token2id and word not in self.stoplist]
        #                    for text in x_clean]

        # Vectorize using W2V model


        if self.w2v_model is not None:
            logging.info("W2V: vectorizing a corpus")
            #vecs = np.memmap('vectors.dat', dtype='float32', mode='w+', shape=(len(X), self.length))
            vecs =  np.zeros((len(X), self.length))
            size = self.w2v_model.layer1_size
            if len(X) > 0:
                for i, text in enumerate(X):
                    vec = self.vectorize_text(self.pre_process(text))
                    vecs[i, :] = vec
                    if i % 1000 == 0 :
                        gc.collect()
                        logging.info("Processeed %i texts" % i)
            else:
                vecs = np.zeros(size).reshape((1, size))
            logging.info("W2V Text Model: returning pre-processed data of shape %s" % (vecs.shape, ))
        else:
            logging.info("W2V Averaged: no model was provided.")
            vecs = np.zeros((len(X), 1))

        return vecs


class DPGMMClusterModel(BaseEstimator, TransformerMixin):

    def __init__(self, w2v_model=None, n_components=None, no_above=0.9, no_below=8, dataname="", stoplist=None,
                 dictionary=None, recluster_thresh=1000, alpha=5):
        self.w2v_model = w2v_model
        self.no_above = no_above
        self.no_below = no_below
        self.alpha = alpha
        self.n_components = n_components
        self.n_sub_components = int(n_components / 2)
        self.stoplist = stoplist
        self.dataname = dataname
        self.dictionary = dictionary
        self.dpgmm = None
        self.scaler = None
        self.cluster_info = None
        # a list of sub-clusterer
        self.feature_crd = {}
        self.subdpgmms = []
        self.reclustered = []
        self.recluster_thresh = recluster_thresh

    def should_cluster_word(self, word):
        return (word in self.dictionary.token2id) and (len(word) > 1) and \
               (self.w2v_model is None or word in self.w2v_model) and \
               (self.stoplist is None or word not in self.stoplist)

    # constructs a dictionary and a DPGMM model on 9000 middle frequency words from X
    # X is a sequence of texts
    def fit(self, X, y=None):
        # either consturct a dictionary from X, trim it
        if self.dictionary is None:
            self.dictionary = corpora.Dictionary(X)
        # or use an existing dictionary and trim the given set of words
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above, keep_n=9000)

        if self.w2v_model is None:
            w2v_corpus = [[word for word in text if self.should_cluster_word(word)] for text in X]
            self.w2v_model = w2v_models.build_word2vec(w2v_corpus, size=100, window=10, min_count=self.no_below,
                                                       dataname=self.dataname+"_dpgmm")

        word_list = np.array([word for word in self.dictionary.token2id.iterkeys() if self.should_cluster_word(word)])

        # This was  reclustering clause - I need to re-write this
        # else:
        #    # note the double loop here!!
        #    word_list = np.array([word for text in X for word in text if self.should_cluster_word(word)])

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

        self.dpgmm = DPGMM(n_components=self.n_components, covariance_type='diag', alpha=self.alpha,
                           n_iter=1000, tol=0.0001)
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
                sub_dpgmm = DPGMMClusterModel(w2v_model=self.w2v_model,
                                              n_components=self.n_sub_components,
                                              dictionary=self.dictionary,
                                              dataname="%s-%i" % (self.dataname, i), stoplist=self.stoplist)
                # recluster words.  Note the double array
                sub_dpgmm.fit([cluster_words])
                self.subdpgmms.append(sub_dpgmm)
                self.reclustered.append(i)
            if cluster_size > 0:
                #cluster_center_original = self.scaler.inverse_transform(cluster_center)
                #similar_words = self.w2v_model.most_similar_cosmul(positive=[cluster_center_original], topn=cluster_size)
                #central_words = [word for word, _ in similar_words if word in cluster_words]
                central_words = cluster_words[0:10]
            else:
                central_words = []
            self.cluster_info.append({'cnt': i, 'size': cluster_size, 'words': central_words})

        filename = "clusters_%s_%i_%.1f_%.0f.txt" % (self.dataname, self.n_components, self.no_above, self.no_below)
        io.save_cluster_info(filename, self.cluster_info)

        # setting up the coordinates for the features
        self.feature_crd = {'global': range(0, self.n_components),
                            'reclustered': [i for i in range(0, self.n_components + self.n_sub_components*len(self.reclustered))
                                            if i not in self.reclustered]}

        return self

    # calculate cluster counts for one text
    def clusterize(self, text):
        word_list = [word for word in text if self.should_cluster_word(word)]
        vec_list = np.array([self.w2v_model[word] for word in word_list])
        bincounts = np.zeros((self.n_components+self.n_sub_components*len(self.reclustered),))

        if len(vec_list) > 0:
            # assign words to clusters
            predictions = self.dpgmm.predict(self.scaler.transform(np.array(vec_list)))
            global_bincount = np.bincount(predictions, minlength=self.n_components)
            # re-assign words in large clusters
            bincounts[0:self.n_components] = global_bincount #reshape((1,len(global_bincount)))
            start = self.n_components
            for i, subdpgmm in zip(self.reclustered, self.subdpgmms):
                # if words in respective clusters exists - recluster them
                vecs_torecluster = vec_list[predictions == i]
                if len(vecs_torecluster) > 0:
                    predictions = subdpgmm.dpgmm.predict(subdpgmm.scaler.transform(np.array(vecs_torecluster)))
                    bincounts[start:start+subdpgmm.dpgmm.n_components] = \
                        np.bincount(predictions, minlength=subdpgmm.dpgmm.n_components) #.reshape((1, subdpgmm.n_components))
                start += subdpgmm.dpgmm.n_components
                # erase the count inthe global counts

        # returns a vector of cluster bin counts: [ global, reclustered1, reclustered2, ...]
        return bincounts.reshape((1, len(bincounts)))


    # for a  text, constructs a bincount of clusters present in the sentence
    # X is a list of texts.  One text is one string! Not tokenized
    def transform(self, X):

        # Text pre-processing
        x_clean = [tu.normalize_punctuation(text).split() for text in X]
        logging.info("DPGGM: Text prepocessed")

        # Vectorize using W2V model
        if self.dpgmm is not None:
            logging.info("Vectorizing a corpus")
            size = self.w2v_model.layer1_size
            if len(X) > 0:
                vecs = np.concatenate([self.clusterize(z) for z in x_clean], axis=0)
            else:
                vecs = np.zeros(size).reshape((1, size))
            logging.info("DPGMM: returning pre-processed data of shape %s" % (vecs.shape, ))
        else:
            logging.info("W2V Averaged: no model was provided.")
            vecs = np.zeros((len(X), 1))

        return vecs


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

__author__ = 'verasazonova'

import argparse
import logging
from operator import itemgetter
import os
import numpy as np
import dateutil.parser
import re
from six import string_types
import time
from tweet_mining.utils import ioutils, plotutils
from tweet_mining.utils import textutils as tu
import w2v_models
import lda_models
import transformers
import cluster_models
from sklearn import cross_validation, grid_search
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion


def calculate_and_plot_lda(filename, ntopics, dataname):
    stop_path = "/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/en_swahili.txt"

    # Load dataset
    dataset = ioutils.KenyanCSVMessage(filename, fields=["id_str", "text", "created_at"], stop_path=stop_path)

    # Unless the counts and topic definitions have already been extracted
    if not os.path.isfile(dataname+"_cnts.txt"):
        # Create the histogram of LDA topics by date
        lda_models.bin_tweets_by_date_and_lda(dataset, n_topics=ntopics, mallet=False, dataname=dataname)

    # Read the resulting counts, date bins, and topics
    counts, bins, topics = ioutils.read_counts_bins_labels(dataname)

    # Figure out which topics to cluster together
    clustered_counts, clustered_labels, clusters = cluster_models.build_clusters(counts, topics, thresh=0.09)

    # Plot the clustered histogram
    plotutils.plot_tweets(counts=clustered_counts, dates=bins, clusters=clusters,
                          labels=clustered_labels, dataname=dataname)

    flattened_topic_list = [word for topic in topics for word, weight in topic]
    print len(flattened_topic_list)


def build_and_test_w2v(filenames, size, window, dataname):

    model_name = w2v_models.make_w2v_model_name(dataname, size, window)

    if isinstance(filenames, string_types):
        filenames = [filenames]

    data = []
    for filename in filenames:
        logging.info("Processing: %s" % filename)
        dataset = ioutils.KenyanCSVMessage(filename, fields=["text"])
        text_pos = dataset.text_pos
        data.append(dataset.data[:, text_pos])

    data = np.concatenate(data)
    logging.info("Read text corpus of size %s" % data.shape)

    if not os.path.isfile(model_name):
        # Load dataset
        model = w2v_models.build_word2vec(data, size=size, window=window, dataname=dataname)
        logging.info("Model created")
    else:
        model = w2v_models.load_w2v(model_name)
        logging.info("Model loaded")

    test_words = open("/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/tests.txt", 'r').readlines()
    for word_list in test_words:
        pos_words = word_list.split(':')[0].split()
        neg_words = word_list.split(':')[1].split()
        list_similar = w2v_models.test_w2v(model, word_list=pos_words, neg_list=neg_words)

        for word, similarity in list_similar:
            print similarity, repr(word)


def extract_political_tweets(filename, size, window, dataname):
    w2v_model_name = w2v_models.make_w2v_model_name(dataname, size, window)
    model = None
    if  os.path.isfile(w2v_model_name):
        model = w2v_models.load_w2v(w2v_model_name)
        logging.info("Model loaded")

    dataset = ioutils.KenyanCSVMessage(filename, fields=["id_str", "text", "created_at"])
    tweets = dataset.data[:, dataset.text_pos]
    ids = dataset.data[:, dataset.id_pos]
    dates = dataset.data[:, dataset.date_pos]

    text_corpus, dictionary, bow = tu.process_text(tweets, stoplist=dataset.stoplist,
                                                   bigrams=None, trigrams=None, keep_all=False,
                                                   no_below=5, no_above=0.9)

    political_words = ["president", "kenyatta", "uhuru"]

    phrase = ["does", "he", "even", "care"]
    phrase = ["do", "something", "to", "save", "your", "people"]

    sim_tweets = []

    for tweet in tweets:
        tweet_lst = [word for word in tweet if word in model]
        for word in political_words:
            if word in tweet:
                sim = model.n_similarity(phrase, tweet_lst)
                sim_tweets.append( (tweet, sim))
                break

    sim_tweets = sorted(sim_tweets, key=itemgetter(1), reverse=True)
    print sim_tweets[0:5]
    print sim_tweets[-5:-1]



def extract_phrases(tweet_text_corpus, stoplist):
    for thresh in [6000, 7000, 8000, 10000]:
        print "Threshhold %i " % thresh
        text_corpus, dictionary, bow = tu.process_text(tweet_text_corpus, stoplist=stoplist,
                                                       bigrams=thresh, trigrams=None, keep_all=False,
                                                       no_below=10, no_above=0.8)

        bigrams = [word for word in dictionary.token2id.keys() if re.search("_", word)]
        print len(bigrams)
        print ", ".join(bigrams)

        print


def assign_cluster(text, words_clusters_dict):
    weights = np.zeros((30,))
    for word in text:
        if word in words_clusters_dict:
            weights[words_clusters_dict[word]] += 1

    return weights


def try_w2v_tweet_clustering(filename, size, window, dataname):

    w2v_model_name = w2v_models.make_w2v_model_name(dataname, size, window)
    w2v_model = w2v_models.load_w2v(w2v_model_name)

    # Load dataset
    start_date = dateutil.parser.parse("Mon Jun 16 00:00:00 +0000 2014")
    end_date = dateutil.parser.parse("Thu Jul 10 00:00:00 +0000 2014")
    stop_path = "/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/utils/en_swahili.txt"

    dataset = ioutils.KenyanCSVMessage(filename, fields=["id_str", "text", "created_at"],
                                       #start_date=start_date, end_date=end_date,
                                       stop_path=stop_path)

    tweet_text_corpus = [tweet[dataset.text_pos] for tweet in dataset]
    ids = [tweet[dataset.id_pos] for tweet in dataset]

    text_corpus, dictionary, bow = tu.process_text(tweet_text_corpus, stoplist=dataset.stoplist,
                                                   bigrams=None, trigrams=None, keep_all=False,
                                                   no_below=5, no_above=0.9)

    cluster_models.assign_language(text_corpus, w2v_model, ids)


#-------------------
def run_grid_search(x, y, clf=None, parameters=None, fit_parameters=None):
    if clf is None:
        raise Exception("No classifier passed")
    if parameters is None:
        raise Exception("No parameters passed")
    print parameters
    grid_clf = grid_search.GridSearchCV(clf, param_grid=parameters, fit_params=fit_parameters,
                                        scoring='accuracy',
                                        iid=False, cv=2, refit=True)
    grid_clf.fit(x, y)
    print grid_clf.grid_scores_
    print grid_clf.best_params_
    print grid_clf.best_score_

    return grid_clf.best_score_

def run_cv_classifier(x, y, clf=None, fit_parameters=None, n_trials=10, n_cv=5):
    scores = np.zeros((n_trials * n_cv))
    for n in range(n_trials):
        logging.info("Testing: trial %i or %i" % (n, n_trials))

        x_shuffled, y_shuffled = shuffle(x, y, random_state=n)
        skf = cross_validation.StratifiedKFold(y_shuffled, n_folds=n_cv)  # random_state=n, shuffle=True)
        scores[n * n_cv:(n + 1) * n_cv] = cross_validation.cross_val_score(clf, x_shuffled, y_shuffled, cv=skf,
                                                                           scoring='f1',
                                                                           fit_params=fit_parameters,
                                                                           verbose=0, n_jobs=1)
    print scores, scores.mean(), scores.std()
    return scores.mean()

#------------------

def make_x_y(filename):
    stop_path = "/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/utils/en_swahili.txt"

    dataset = ioutils.KenyanCSVMessage(filename, fields=["text", "label"], stop_path=stop_path)

    tweet_text_corpus = [tweet[dataset.text_pos] for tweet in dataset]
    labels = [tweet[dataset.label_pos] for tweet in dataset]
    classes, indices = np.unique(labels, return_inverse=True)
    print classes
    return tweet_text_corpus, indices, dataset.stoplist


def tweet_classification(filename, size, window, dataname, filename2=None):

    x_full, y_full, stoplist = make_x_y(filename)

    n_trials = 5
    ps = [0.001, 0.01, 0.1]
    threshs = [0, 0.1, 0.2, 0.4, 0.6, 0.8]

    clf = LogisticRegression(C=1)
    #clf = SVC(kernel='linear', C=1)

    for p in ps: #

        for n in range(n_trials):

            x_unlabeled, x_cv, y_unlabeled, y_cv = train_test_split(x_full, y_full, test_size=p, random_state=n)

            for thresh in threshs:


                x_other, x_w2v = train_test_split(x_unlabeled, test_size=thresh)

                w2v_corpus = [tu.normalize_punctuation(text).split() for text in np.concatenate([x_cv, x_w2v])]
                w2v_model = w2v_models.build_word2vec(w2v_corpus, size=size, window=window, min_count=1, dataname=dataname)
                logging.info("Model created")

                clf_pipeline = Pipeline([
                        ('w2v_avg', transformers.W2VAveragedModel(w2v_model=w2v_model, no_above=0.99, no_below=1, stoplist=[])),
                        ('clf', clf) ])

                mean = run_cv_classifier(x_cv, y_cv, clf=clf_pipeline, n_trials=5, n_cv=5)
                print n, "w2v", size, p, thresh, len(x_cv), len(w2v_corpus), mean

            clf_pipeline = Pipeline([
                        ('bow', transformers.BOWModel(no_above=0.8, no_below=2, stoplist=stoplist)),
                        ('clf', clf) ])

            mean = run_cv_classifier(x_cv, y_cv, clf=clf_pipeline, n_trials=5, n_cv=5)
            print n, "bow", p, len(x_cv),  mean



def print_tweets(filename):

    ioutils.clean_save_tweet_text(filename, ["id_str"])

    #data = ioutils.KenyanCSVMessage(filename, ["id_str", "text", "created_at"])
    #for row in data:
    #    print row[data.text_pos]


def __main__():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', nargs='+', help='Filename')
    parser.add_argument('--dname', action='store', dest='dataname', default="log", help='Output filename')
    parser.add_argument('-n', action='store', dest='ntopics', default='10', help='Number of LDA topics')
    parser.add_argument('--size', action='store', dest='size', default='100', help='Size w2v of LDA topics')
    parser.add_argument('--window', action='store', dest='window', default='10', help='Number of LDA topics')

    arguments = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename=arguments.dataname+"_log.txt")

    #test_print_tweets(arguments.filename)
    #clean_save_tweets(arguments.filename)
    #get_statistics(arguments.filename)

    #calculate_and_plot_lda(arguments.filename, int(arguments.ntopics), arguments.dataname)

    #build_and_test_w2v(arguments.filename, int(arguments.size), int(arguments.window), arguments.dataname)

    #extract_political_tweets(arguments.filename[0], int(arguments.size), int(arguments.window), arguments.dataname)


    #print_tweets(arguments.filename)

    #try_w2v_tweet_clustering(arguments.filename[0], int(arguments.size), int(arguments.window), arguments.dataname)

    #compare_language_identification(arguments.filename,  "word_clusters_identified.txt",  "word_clusters.txt")

    tweet_classification(arguments.filename[0], int(arguments.size), int(arguments.window), arguments.dataname)

if __name__ == "__main__":
    __main__()
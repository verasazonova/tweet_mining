
__author__ = 'verasazonova'

import pickle
import argparse
import logging
from sklearn.preprocessing import StandardScaler
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
    #print scores, scores.mean(), scores.std()
    return scores

#------------------

def make_x_y(filename, fields=None):
    stop_path = "/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/utils/en_swahili.txt"

    dataset = ioutils.KenyanCSVMessage(filename, fields=fields, stop_path=stop_path)

    tweet_text_corpus = [tweet[dataset.text_pos] for tweet in dataset]
    if dataset.label_pos is not None:
        labels = [tweet[dataset.label_pos] for tweet in dataset]
        classes, indices = np.unique(labels, return_inverse=True)
        print classes
    else:
        indices = None
    return tweet_text_corpus, indices, dataset.stoplist


def tweet_classification(filename, size, window, dataname, per=None, thr=None, ntrial=None, clf_name='w2v',
                         unlabeled_filenames=None, clf_base="lr"):

    x_full, y_full, stoplist = make_x_y(filename, ["text", "label"])

    if unlabeled_filenames is not None:
        x_unlabeled = []
        for unlabeled in unlabeled_filenames:
            x, _, _ = make_x_y(unlabeled, ["text"])
            x_unlabeled += x
    else:
        x_unlabeled = []

    if ntrial is None or ntrial == -1:
        n_trials = [0]
    else:
        n_trials = [ntrial]
    if per is None or per == -1:
        ps = [1]
    else:
        ps = [per]
    if thr is None or thr == -1:
        threshs = [1]#  [0, 0.1, 0.4, 0.8]
    else:
        threshs = [thr]


    n_components = 30

    logging.info("Classifing for p= %s" % ps)
    logging.info("Classifing for ntrials = %s" % n_trials)
    logging.info("Classifing for threshs = %s" % threshs)

    if clf_base == "lr":
        clf = LogisticRegression(C=1)
    else:
        clf = SVC(kernel='linear', C=1)

    for p in ps: #

        for n in n_trials:
            print n
            if p == 1:
                x_labeled = x_full
                y_labeled = y_full

            else:

                x_unlabeled, x_labeled, y_unlabeled, y_labeled = train_test_split(x_full, y_full, test_size=p, random_state=n)

            if clf_name == 'w2v':

                for thresh in threshs:

                    if thresh == 1:
                        x_unlabeled_for_w2v = x_unlabeled
                    else:
                        x_unused, x_unlabeled_for_w2v = train_test_split(x_unlabeled, test_size=thresh, random_state=0)

                    scores_avg, scores_std, scores_cluster = w2v_classify_tweets(x_data=x_labeled,
                                                                                 y_data=y_labeled,
                                                                                 unlabeled_data=x_unlabeled_for_w2v,
                                                                                 window=window,
                                                                                 size=size,
                                                                                 dataname=dataname,
                                                                                 n_components=n_components,
                                                                                 clf=clf,
                                                                                 rebuild=True)

                    with open(dataname + "_" + clf_base + "_fscore.txt", 'a') as f:
                        for i, score in enumerate(scores_avg):
                            f.write("%i, %i,  %s, %i, %f, %f, %i, %i, %f, %i \n" %
                                (n, i, "avg", size, p, thresh, len(x_labeled), len(x_unlabeled_for_w2v), score, n_components))
                        for i, score in enumerate(scores_std):
                            f.write("%i, %i,  %s, %i, %f, %f, %i, %i, %f, %i \n" %
                                (n, i, "std", size, p, thresh, len(x_labeled), len(x_unlabeled_for_w2v), score, n_components))
                        for i, score in enumerate(scores_cluster):
                            f.write("%i, %i,  %s, %i, %f, %f, %i, %i, %f, %i \n" %
                                (n, i, "cluster", size, p, thresh, len(x_labeled), len(x_unlabeled_for_w2v), score, n_components))

            else:

                clf_pipeline = Pipeline([
                        ('bow', transformers.BOWModel(no_above=0.8, no_below=2, stoplist=stoplist)),
                        ('clf', clf) ])

                scores = run_cv_classifier(x_labeled, y_labeled, clf=clf_pipeline, n_trials=1, n_cv=5)
                with open(dataname + "_" + clf_base + "_fscore.txt", 'a') as f:
                    for i, score in enumerate(scores):
                        f.write("%i, %i, %s, %i, %f, %f, %i, %i, %f, %i \n" %
                            (n, i, "bow", -1, p, -1, len(x_labeled), -1, score, -1))

                print n, "bow", p,len(x_labeled),  scores.mean()


def w2v_classify_tweets(x_data=None, y_data=None, unlabeled_data=None, window=0, size=0, dataname="", n_components=0,
                        clf=None, rebuild=True):

    w2v_corpus = [tu.normalize_punctuation(text).split() for text in np.concatenate([x_data, unlabeled_data])]

    print len(w2v_corpus)

    w2v_model_name = w2v_models.make_w2v_model_name(dataname=dataname, size=size, window=window,
                                                    min_count=1, corpus_length=len(w2v_corpus))
    logging.info("Looking for model %s" % w2v_model_name)
    if not rebuild and os.path.isfile(w2v_model_name):
        w2v_model = w2v_models.load_w2v(w2v_model_name)
        logging.info("Model Loaded")
    else:
        w2v_model = w2v_models.build_word2vec(w2v_corpus, size=size, window=window, min_count=1, dataname=dataname)
    logging.info("Model created")
    w2v_model.init_sims(replace=True)

    cluster_model_names, scaler_name = w2v_models.build_dpgmm_model(w2v_corpus, w2v_model=w2v_model,
                                                                    n_components=n_components, dataname=dataname)
    #cluster_model_names = ["dpggm_model-30"]
    #scaler_name = "dpggm_model-scaler"

    x_data = transformers.W2VAveragedModel(w2v_model=w2v_model,
                                             cluster_model_names=cluster_model_names,
                                             no_above=1.0, no_below=1,
                                             scaler_name=scaler_name,
                                             stoplist=[], type="cluster").fit_transform(x_data)

    # scale averages
    x_data_avg = StandardScaler().fit_transform(x_data[:, 0:size])
    # scale standard deviations separately
    x_data_std = np.concatenate([x_data_avg, StandardScaler().fit_transform(x_data[:, size:2*size])], axis=1)
    # scale clusters
    x_data_cluster = np.concatenate([x_data_std, x_data[:, 2*size:2*size+30]], axis=1)

    scores_avg = run_cv_classifier(x_data_avg, y_data, clf=clf, n_trials=1, n_cv=5)
    scores_std = run_cv_classifier(x_data_std, y_data, clf=clf, n_trials=1, n_cv=5)
    scores_cluster = run_cv_classifier(x_data_cluster, y_data, clf=clf, n_trials=1, n_cv=5)

    print "Scaled: ", scores_avg.mean(), scores_std.mean(), scores_cluster.mean()

    return scores_avg, scores_std, scores_cluster

def print_tweets(filename):

    ioutils.clean_save_tweet_text(filename, ["id_str"])

    #data = ioutils.KenyanCSVMessage(filename, ["id_str", "text", "created_at"])
    #for row in data:
    #    print row[data.text_pos]


def plot_scores():
   # plotutils.plot_curves_baseslines()
    plotutils.plot_kenyan_data()



def __main__():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', nargs='+', help='Filename')
    parser.add_argument('--dname', action='store', dest='dataname', default="log", help='Output filename')
    parser.add_argument('-n', action='store', dest='ntopics', default='10', help='Number of LDA topics')
    parser.add_argument('--size', action='store', dest='size', default='100', help='Size w2v of LDA topics')
    parser.add_argument('--window', action='store', dest='window', default='10', help='Number of LDA topics')
    parser.add_argument('--p', action='store', dest='p', default='-1', help='Fraction of labeled data')
    parser.add_argument('--thresh', action='store', dest='thresh', default='-1', help='Fraction of unlabelled data')
    parser.add_argument('--ntrial', action='store', dest='ntrial', default='-1', help='Number of the trial')
    parser.add_argument('--clfbase', action='store', dest='clfbase', default='lr', help='Number of the trial')
    parser.add_argument('--clfname', action='store', dest='clfname', default='w2v', help='Number of the trial')
    parser.add_argument('--action', action='store', dest='action', default='plot', help='Number of the trial')


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

    if arguments.action == "classify":
        if len(arguments.filename) > 1:
            tweet_classification(arguments.filename[0], int(arguments.size), int(arguments.window), arguments.dataname,
                             per=float(arguments.p), thr=float(arguments.thresh), ntrial=int(arguments.ntrial),
                             clf_name=arguments.clfname, unlabeled_filenames=arguments.filename[1:],
                             clf_base=arguments.clfbase)
        else:
            tweet_classification(arguments.filename[0], int(arguments.size), int(arguments.window), arguments.dataname,
                             per=float(arguments.p), thr=float(arguments.thresh), ntrial=int(arguments.ntrial),
                             clf_name=arguments.clfname, unlabeled_filenames=None,
                             clf_base=arguments.clfbase)

    elif arguments.action == "plot":
        plot_scores()

    elif arguments.action == "test":
        ioutils.make_positive_labeled_kenyan_data(arguments.dataname)

if __name__ == "__main__":
    __main__()
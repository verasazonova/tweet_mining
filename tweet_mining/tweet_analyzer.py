
__author__ = 'verasazonova'

import argparse
import sklearn.metrics
import logging
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import re
import pickle
from tweet_mining.utils import ioutils, plotutils
from tweet_mining.utils import textutils as tu
import w2v_models
import lda_models
import transformers
import cluster_models
from sklearn import cross_validation, grid_search
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline


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


def explore_classifier(x, y, clf=None, n_trials=1):
    for n in range(n_trials):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=n)
        print x_train[0]
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(sklearn.metrics.confusion_matrix(y_test, y_pred))
        print(sklearn.metrics.classification_report(y_test, y_pred))
        print sklearn.metrics.f1_score(y_test, y_pred)
        print sklearn.metrics.f1_score(y_test, y_pred, average='weighted')
#        print sklearn.metrics.f1_score(y_test, y_pred, average='micro')


#------------------
def make_x_y(filename, fields=None):
    stop_path = "/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/utils/en_swahili.txt"

    dataset = ioutils.KenyanCSVMessage(filename, fields=fields, stop_path=stop_path)

    tweet_text_corpus = [tweet[dataset.text_pos] for tweet in dataset]
    if dataset.label_pos is not None:
        labels = [tweet[dataset.label_pos] for tweet in dataset]
        classes, indices = np.unique(labels, return_inverse=True)
        # a hack to change the order
        #indices = -1*(indices - 1)
        print classes
        print np.bincount(indices)
    else:
        indices = None
    return tweet_text_corpus, indices, dataset.stoplist
#------------------


def tweet_classification(filename, size, window, dataname, per=None, thr=None, ntrial=None, clf_name='w2v',
                         unlabeled_filenames=None, clf_base="lr", explore=False, rebuild=False, min_count=1,
                         recluster_thresh=0):

    x_full, y_full, stoplist = make_x_y(filename, ["text", "label"])

    if unlabeled_filenames is not None:
        x_unlabeled = []
        for unlabeled in unlabeled_filenames:
            x, _, _ = make_x_y(unlabeled, ["text"])
            x_unlabeled += x
    else:
        x_unlabeled = []

    if ntrial is None or ntrial == -1:
        n_trials = range(5)
    else:
        n_trials = [ntrial]
    if per is None or per == -1:
        ps = [1]
    else:
        ps = [per]
    if thr is None or thr == -1:
        threshs = [1]  # [0, 0.1, 0.4, 0.8]
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

    for p in ps:  #

        for n in n_trials:
            print n
            if p == 1:
                x_labeled = x_full
                y_labeled = y_full

            else:

                x_unlabeled, x_labeled, y_unlabeled, y_labeled = train_test_split(x_full, y_full, test_size=p,
                                                                                  random_state=n)

            if clf_name == 'w2v':

                for thresh in threshs:

                    if thresh == 1:
                        x_unlabeled_for_w2v = x_unlabeled
                    else:
                        x_unused, x_unlabeled_for_w2v = train_test_split(x_unlabeled, test_size=thresh, random_state=0)

                    names, scores_array = w2v_classify_tweets(x_data=x_labeled,
                                                       y_data=y_labeled,
                                                       unlabeled_data=x_unlabeled_for_w2v,
                                                       window=window,
                                                       size=size,
                                                       min_count=min_count,
                                                       dataname=dataname,
                                                       n_components=n_components,
                                                       clf=clf,
                                                       stoplist=stoplist,
                                                       explore=explore,
                                                       recluster_thresh=recluster_thresh,
                                                       rebuild=rebuild)

                    with open(dataname + "_" + clf_base + "_fscore.txt", 'a') as f:
                        for scores, name in zip(scores_array, names):
                            for i, score in enumerate(scores):
                                f.write("%i, %i,  %s, %i, %f, %f, %i, %i, %f, %i \n" %
                                       (n, i, name, size, p, thresh, len(x_labeled),
                                        len(x_unlabeled_for_w2v), score, n_components))

            else:

                clf_pipeline = Pipeline([
                                         ('bow', transformers.BOWModel(no_above=0.8, no_below=2, stoplist=stoplist)),
                                         ('clf', clf)])

                if explore:
                    explore_classifier(x_labeled, y_labeled, clf_pipeline, n_trials=1)
                else:
                    scores = run_cv_classifier(x_labeled, y_labeled, clf=clf_pipeline, n_trials=1, n_cv=5)
                    with open(dataname + "_" + clf_base + "_fscore.txt", 'a') as f:
                        for i, score in enumerate(scores):
                            f.write("%i, %i, %s, %i, %f, %f, %i, %i, %f, %i \n" %
                                (n, i, "bow", -1, p, -1, len(x_labeled), -1, score, -1))

                    print n, "bow", p, len(x_labeled),  scores.mean()


def build_w2v_model(w2v_corpus, dataname="", window=0, size=0, min_count=0, rebuild=False, explore=False):
    w2v_model_name = w2v_models.make_w2v_model_name(dataname=dataname, size=size, window=window,
                                                    min_count=min_count, corpus_length=len(w2v_corpus))
    logging.info("Looking for model %s" % w2v_model_name)
    if (not rebuild or explore) and os.path.isfile(w2v_model_name):
        w2v_model = w2v_models.load_w2v(w2v_model_name)
        logging.info("Model Loaded")
    else:
        w2v_model = w2v_models.build_word2vec(w2v_corpus, size=size, window=window, min_count=min_count, dataname=dataname)
        logging.info("Model created")
    w2v_model.init_sims(replace=True)

    check_w2v_model(w2v_model=w2v_model)
    return w2v_model


def build_dpgmm_model(w2v_corpus, w2v_model=None, n_components=0, dataname="", stoplist=None, recluster_thresh=0,
                      rebuild=False, alpha=5, no_below=6, no_above=0.9):

    model_name = w2v_models.make_dpgmm_model_name(dataname=dataname,n_components=n_components, n_below=no_below,
                                                  n_above=no_above, alpha=alpha)
    logging.info("Looking for model %s" % model_name)
    if not rebuild and os.path.isfile(model_name):
        dpgmm = pickle.load(open(model_name, 'rb'))
    else:
        dpgmm = transformers.DPGMMClusterModel(w2v_model=w2v_model, n_components=n_components, dataname=dataname,
                                               stoplist=stoplist, recluster_thresh=recluster_thresh, alpha=alpha,
                                               no_below=no_below, no_above=no_above)
        dpgmm.fit(w2v_corpus)
        pickle.dump(dpgmm, open(model_name, 'wb'))
    return dpgmm


def w2v_classify_tweets(x_data=None, y_data=None, unlabeled_data=None, window=0, size=0, dataname="", n_components=0,
                        clf=None, rebuild=False, explore=False, stoplist=None, min_count=1, recluster_thresh=0,
                        no_above=0.9, no_below=5):

    w2v_corpus = np.array([tu.normalize_punctuation(text).split() for text in np.concatenate([x_data, unlabeled_data])])

    logging.info("Classifying %s, %i, %i, %i, %i" % (dataname, len(w2v_corpus), min_count, recluster_thresh, n_components))

    # build models
    w2v_model = build_w2v_model(w2v_corpus, dataname=dataname, window=window, size=size, min_count=min_count,
                                rebuild=rebuild, explore=explore)

    # get features from models
    w2v = transformers.W2VTextModel(w2v_model=w2v_model, no_above=1.0, no_below=1, diffmax0=4, diffmax1=4)

    #dpgmm = build_dpgmm_model(w2v_corpus, w2v_model=w2v_model, n_components=n_components, dataname=dataname,
    #                          stoplist=stoplist, recluster_thresh=recluster_thresh, alpha=5, no_above=no_above,
    #                          no_below=no_below)

    # get matrices of features from x_data
    w2v_data = w2v.fit_transform(x_data)
    #dpgmm_data = dpgmm.transform(x_data)

    print w2v_data.shape
    #print dpgmm_data.shape


    dpgmm = transformers.DPGMMClusterModel(w2v_model=None, n_components=n_components, dataname=dataname,
                                           stoplist=stoplist, recluster_thresh=0, no_above=no_above, no_below=no_below,
                                           alpha=5)
    dpgmm.fit(w2v_corpus)
    dpgmm_data = dpgmm.transform(x_data)

    # scale features
    for name, inds in w2v.feature_crd.items():
        w2v_data[:, inds] = StandardScaler().fit_transform(w2v_data[:, inds])

    for name, inds in dpgmm.feature_crd.items():
        dpgmm_data[:, inds] = StandardScaler().fit_transform(dpgmm_data[:, inds])

    names = []
    experiments = []


#['0_avg', '1_std', '4_diff0_3', '7_diff1_3']:   #
    for name in sorted(w2v.feature_crd.keys()):
        print name
        names.append(name)
        if len(experiments) > 0:
            experiments.append(np.concatenate([experiments[-1], w2v_data[:, w2v.feature_crd[name]]], axis=1))
        else:
            experiments.append(w2v_data[:, w2v.feature_crd[name]])

    names.append("cluster")
    experiments.append(np.concatenate([experiments[-1], dpgmm_data[:, dpgmm.feature_crd['global']]], axis=1))

    #x_data_cluster = np.concatenate([x_data_std_diff2, dpgmm_data[:, dpgmm.feature_crd['global']]], axis=1)

    #names = ['avg', 'std', 'diff0_1', 'diff0_2', 'diff0_3', 'all']
    #experiments = [x_data_avg, x_data_std, x_data_std_diff, x_data_std_diff2, x_data_std_diff3, x_data_std_all]

    if explore:

        for experiment in experiments:
            explore_classifier(experiment, y_data, clf=clf, n_trials=1)

        return [], []
    else:

        results = []
        for name, experiment in zip(names, experiments):
            scores = run_cv_classifier(experiment, y_data, clf=clf, n_trials=1, n_cv=5)
            print name, scores, scores.mean()
            results.append(scores)

        return names, results


def w2v_cluster_tweet_vocab(filename, window=0, size=0, dataname="", n_components=0, min_count=1,
                            rebuild=False):

    print "Clustering"
    x_data, y_data, stoplist = make_x_y(filename, ["text"])
    w2v_corpus = np.array([tu.normalize_punctuation(text).split() for text in x_data])

    #w2v_model = build_w2v_model(w2v_corpus, dataname=dataname, window=window, size=size, min_count=min_count,
    #                            rebuild=rebuild, explore=False)

    dpgmm = transformers.DPGMMClusterModel(w2v_model=None, n_components=n_components, dataname=dataname,
                                           stoplist=stoplist, recluster_thresh=0, no_above=0.9, no_below=5,
                                           alpha=5)
    dpgmm.fit(w2v_corpus)

    #print dpgmm.dpgmm.precs_.shape


def check_w2v_model(filename="", w2v_model=None, window=0, size=0, min_count=1, dataname="", rebuild=True):

    print "Checking model for consistency"

    if w2v_model is None:
        x_data, y_data, stoplist = make_x_y(filename, ["text"])

        w2v_corpus = np.array([tu.normalize_punctuation(text).split() for text in x_data])

        logging.info("Pre-processing 2 done")
        logging.info("First line: %s" % w2v_corpus[0])
        logging.info("Last line: %s" % w2v_corpus[-1])

        w2v_model = build_w2v_model(w2v_corpus, dataname=dataname, window=window, size=size, min_count=min_count,
                                    rebuild=rebuild, explore=False)

    test_words = open("/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/tests.txt", 'r').readlines()
    for word_list in test_words:
        pos_words = word_list.split(':')[0].split()
        neg_words = word_list.split(':')[1].split()
        list_similar = w2v_models.test_word2vec(w2v_model, word_list=pos_words, neg_list=neg_words)
        print "%s - %s" % (pos_words, neg_words)
        for word, similarity in list_similar:
            print similarity, repr(word)


def print_tweets(filename):

    ioutils.clean_save_tweet_text(filename, ["id_str"])

    #data = ioutils.KenyanCSVMessage(filename, ["id_str", "text", "created_at"])
    #for row in data:
    #    print row[data.text_pos]


def plot_scores(dataname):
    plotutils.plot_kenyan_data(dataname)


def __main__():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', nargs='+', help='Filename')
    parser.add_argument('--dname', action='store', dest='dataname', default="log", help='Output filename')
    parser.add_argument('-n', action='store', dest='ntopics', default='10', help='Number of LDA topics')
    parser.add_argument('--size', action='store', dest='size', default='100', help='Size w2v of LDA topics')
    parser.add_argument('--window', action='store', dest='window', default='10', help='Number of LDA topics')
    parser.add_argument('--min', action='store', dest='min', default='1', help='Number of LDA topics')
    parser.add_argument('--nclusters', action='store', dest='nclusters', default='30', help='Number of LDA topics')
    parser.add_argument('--clusthresh', action='store', dest='clusthresh', default='0', help='Threshold for reclustering')
    parser.add_argument('--p', action='store', dest='p', default='-1', help='Fraction of labeled data')
    parser.add_argument('--thresh', action='store', dest='thresh', default='-1', help='Fraction of unlabelled data')
    parser.add_argument('--ntrial', action='store', dest='ntrial', default='-1', help='Number of the trial')
    parser.add_argument('--clfbase', action='store', dest='clfbase', default='lr', help='Number of the trial')
    parser.add_argument('--clfname', action='store', dest='clfname', default='w2v', help='Number of the trial')
    parser.add_argument('--action', action='store', dest='action', default='plot', help='Number of the trial')
    parser.add_argument('--rebuild', action='store_true', dest='rebuild', help='Number of the trial')


    arguments = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename=arguments.dataname+"_log.txt")

    # parameters for w2v model
    min_count = int(arguments.min)
    n_components = int(arguments.nclusters)
    size=int(arguments.size)
    window=int(arguments.window)
    recluster_thresh=int(arguments.clusthresh)

    # parameters for large datasets
    ntrial = int(arguments.ntrial)
    threshhold = float(arguments.thresh)
    percentage = float(arguments.p)


    # runs a classification experiement a given file
    if arguments.action == "classify":
        if len(arguments.filename) > 1:
            tweet_classification(arguments.filename[0], size=size, window=window, dataname=arguments.dataname,
                             per=percentage, thr=threshhold, ntrial=ntrial, min_count=min_count,
                             clf_name=arguments.clfname, unlabeled_filenames=arguments.filename[1:],
                             clf_base=arguments.clfbase, recluster_thresh=recluster_thresh,
                             rebuild=arguments.rebuild)
        else:
            tweet_classification(arguments.filename[0], size=size, window=window, dataname=arguments.dataname,
                             per=percentage, thr=threshhold, ntrial=ntrial, min_count=min_count,
                             clf_name=arguments.clfname, unlabeled_filenames=None,
                             clf_base=arguments.clfbase, recluster_thresh=recluster_thresh,
                             rebuild=arguments.rebuild)

    # debugs a classification experiement a given file
    elif arguments.action == "explore":
        tweet_classification(arguments.filename[0], size=size, window=window, dataname=arguments.dataname,
                             per=percentage, thr=threshhold, ntrial=ntrial, min_count=min_count,
                             clf_name=arguments.clfname, unlabeled_filenames=arguments.filename[1:],
                             clf_base=arguments.clfbase, explore=True, recluster_thresh=recluster_thresh,
                             rebuild=arguments.rebuild)

    # clusters the vocabulary of a given file accoding to the w2v model constructed on the same file
    elif arguments.action == "cluster":
        w2v_cluster_tweet_vocab(arguments.filename[0],
                                size=size,
                                window=window,
                                dataname=arguments.dataname,
                                n_components=n_components,
                                rebuild=arguments.rebuild,
                                min_count=min_count)

    # given a testfile of words, print most similar word from the model constructed on the file
    elif arguments.action == "check":

        check_w2v_model(filename=arguments.filename[0],
                        size=size,
                        window=window,
                        dataname=arguments.dataname,
                        min_count=min_count,
                        rebuild=arguments.rebuild)

    # plot results of a classification experiment for a certain dataname
    elif arguments.action == "plot":
        plot_scores(arguments.dataname)

    # merge a unlabeled dataset, with positive labels to produce a positively labeled dataset
    elif arguments.action == "make_labels":
        ioutils.make_positive_labeled_kenyan_data(arguments.dataname)

if __name__ == "__main__":
    __main__()
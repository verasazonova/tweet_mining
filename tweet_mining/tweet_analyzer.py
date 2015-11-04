
__author__ = 'verasazonova'

import argparse
import sklearn.metrics
import logging
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import re
from tweet_mining.utils import ioutils, plotutils
from tweet_mining.utils import textutils as tu
import w2v_models
import pickle
import lda_models
import transformers
import cluster_models
from sklearn import cross_validation, grid_search
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
import time
from scipy.sparse import csr_matrix

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


def run_train_test_classifier(x, y, train_end, start, stop, clf=None):
    #print x_train.shape, y_train.shape, x_test.shape, y_test.shape
    scores = np.zeros((1, 4))
    MAX = 2e6
    # if we can fit the whole array in memory.

    if train_end < MAX:
        clf.fit(csr_matrix(x[0:train_end, start:stop]), y[0:train_end])
    # if not, go by batches.

    else:
        batch_size = 10000
        n_batches = int(train_end/batch_size)
        all_classes = np.unique(y)
        print "Learning by batches: %i " % n_batches
        logging.info("Learning by batches: %i " % n_batches )

        # cycle over the data 5 times, shuffling the order of training
        for r in range(5):
            inds = shuffle(range(train_end), random_state=r)
            for i in range(n_batches):
                logging.info("Run  %i %i " % (r, i))
                batch_inds = inds[i*batch_size:(i+1)*batch_size]
                clf.partial_fit(csr_matrix(x[batch_inds, start:stop]), y[batch_inds], classes=all_classes)

            # last batch
            batch_inds = inds[n_batches*batch_size:]
            if batch_inds:
                clf.partial_fit(csr_matrix(x[batch_inds, start:stop]), y[batch_inds], classes=all_classes)

    predictions = clf.predict(csr_matrix(x[train_end:, start:stop]))
    for i, metr in enumerate([sklearn.metrics.accuracy_score, sklearn.metrics.precision_score,
                              sklearn.metrics.recall_score, sklearn.metrics.f1_score]):
        sc = metr(y[train_end:], predictions)
        scores[0, i] = sc

    return scores


def run_cv_classifier(x, y, clf=None, fit_parameters=None, n_trials=10, n_cv=5):
    # all cv will be averaged out together
    scores = np.zeros((n_trials * n_cv, 4))
    for n in range(n_trials):
        logging.info("Testing: trial %i or %i" % (n, n_trials))

        x_shuffled, y_shuffled = shuffle(x, y, random_state=n)
        skf = cross_validation.StratifiedKFold(y_shuffled, n_folds=n_cv)  # random_state=n, shuffle=True)
        #score_names = ['accuracy', 'f1', 'precision', 'recall']
        #metrics = {score: cross_validation.cross_val_score(dt,x_data_tfidf.toarray(), target_arr, cv=cv,scoring=score) for score in scores}
#            sc = cross_validation.cross_val_score(clf, x_shuffled, y=y_shuffled, cv=skf,
#                                              scoring=scoring_name,
#                                              fit_params=fit_parameters,
#                                              verbose=2, n_jobs=1)
        predictions = cross_validation.cross_val_predict(clf, x_shuffled, y=y_shuffled, cv=skf, n_jobs=1, verbose=2)
        n_fold = 0
        for _, test_ind in skf:
            for i, metr in enumerate([sklearn.metrics.accuracy_score, sklearn.metrics.precision_score   ,
                                        sklearn.metrics.recall_score, sklearn.metrics.f1_score]):

                sc = metr(y_shuffled[test_ind], predictions[test_ind])
                scores[n * n_cv + n_fold, i] = sc
            n_fold += 1
    #print scores, scores.mean(), scores.std()
    return scores


def explore_classifier(x, y, clf=None, n_trials=1, orig_data=None):
    false_positives = []
    false_negatives = []
    positives = []
    print np.bincount(y)
    for n in range(n_trials):
        skf = cross_validation.StratifiedKFold(y, n_folds=5)  # random_state=n, shuffle=True)
        predictions = cross_validation.cross_val_predict(clf, x, y=y, cv=skf, n_jobs=1, verbose=2)

        print len(predictions)
        print np.bincount(predictions)
#        x_train, x_test, y_train, y_test, _, orig_test = train_test_split(x, y, orig_data, test_size=0.2, random_state=n)
#        #print x_train[0]
#        clf.fit(x_train, y_train)
#        y_pred = clf.predict(x_test)

        print(sklearn.metrics.confusion_matrix(y, predictions))
        print(sklearn.metrics.classification_report(y, predictions))
        print(sklearn.metrics.precision_score(y, predictions))
        print
        for i in range(len(predictions)):
            if predictions[i] != y[i]:
                if y[i] == 1:
                    false_negatives.append(orig_data[i])
                else:
                    false_positives.append(orig_data[i])
            if y[i] == 1:
                positives.append(orig_data[i])

    print len(false_negatives), len(false_positives)
    false_positives = list(set(false_positives))
    false_negatives = list(set(false_negatives))
    print "False positives:", len(false_positives)
    ioutils.save_positives(false_positives, dataname="false")
    ioutils.save_positives(false_negatives, dataname="false_negatives")
    ioutils.save_positives(positives, dataname="positives")
    print
    print "False negatives:", len(false_negatives)

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

    if dataset.id_pos is not None:
        ids = [tweet[dataset.id_pos] for tweet in dataset]
    else:
        ids = None
    return tweet_text_corpus, indices, dataset.stoplist, ids
#------------------

def read_and_split_data(filename, p=1, thresh=0, n_trial=0, unlabeled_filenames=None, dataname=""):
    x_full, y_full, stoplist, ids = make_x_y(filename, ["text", "label", "id_str"])

    if unlabeled_filenames is not None:
        x_unlabeled = []
        for unlabeled in unlabeled_filenames:
            x, _, _, _ = make_x_y(unlabeled, ["text"])
            x_unlabeled += x
    else:
        x_unlabeled = []

    logging.info("Classifing for p= %s" % p)
    logging.info("Classifing for ntrials = %s" % n_trial)
    logging.info("Classifing for threshs = %s" % thresh)

    if p == 1:
        x_labeled = x_full
        y_labeled = y_full
        ids_l = ids
    else:
        x_unlabeled, x_labeled, y_unlabeled, y_labeled, _, ids_l = train_test_split(x_full, y_full, ids, test_size=p,
                                                                                    random_state=n_trial)

    if thresh == 1:
        x_unlabeled_for_w2v = x_unlabeled
    else:
        x_unused, x_unlabeled_for_w2v = train_test_split(x_unlabeled, test_size=thresh, random_state=0)

    experiment_name = "%s_%0.3f_%0.1f_%i" % (dataname, p, thresh, n_trial)

    return x_labeled, y_labeled, x_unlabeled_for_w2v, experiment_name, stoplist, ids_l


def tweet_classification(filename, size, window, dataname, p=None, thresh=None, n_trial=None, clf_name='w2v',
                         unlabeled_filenames=None, clf_base="lr", action="classify", rebuild=False, min_count=1,
                         recluster_thresh=0, n_components=30, experiment_nums=None, test_filename=None,
                         diff1_max=3, diff0_max=1):

    experiment_name = "%s_%0.3f_%0.1f_%i" % (dataname, p, thresh, n_trial)

    start_time = time.time()


    w2v_data_name = dataname+"_w2v_data"
    w2v_data_scaled_name = "%s_%i_%i_%i_scaled_w2v_data" % (experiment_name, size, window, min_count)
    y_data_name = "%s_%i_%i_%i_y_data" % (experiment_name, size, window, min_count)
    w2v_feature_crd_name = "%s_%i_%i_%i_w2v_f_crd" % (experiment_name, size, window, min_count)
    ids = []
    x_data = []

    if not os.path.isfile(w2v_data_scaled_name+".npy"):

        x_data, y_data, unlabeled_data, run_dataname, stoplist, ids = read_and_split_data(filename=filename, p=p, thresh=thresh,
                                                                                  n_trial=n_trial, dataname=dataname)

        train_data_end = len(y_data)

        if test_filename is not None:
            x_test, y_test, _, _ = make_x_y(test_filename,["text", "label", "id_str"])
            x_data = np.concatenate([x_data, x_test])
            y_data = np.concatenate([y_data, y_test])
            print x_data.shape, y_data.shape


        #x_data, y_data, unlabeled_data, run_dataname, stoplist = read_and_split_data(filename=filename, p=p, thresh=thresh,
        #                                                                      n_trial=n_trial, dataname=dataname)

        # should make this into a separate process to release memory afterwards
        w2v_data, w2v_feature_crd = build_and_vectorize_w2v(x_data=x_data, y_data=y_data,
                                                              unlabeled_data=unlabeled_data, window=window,
                                                              size=size, dataname=run_dataname,
                                                              rebuild=rebuild,action=action,
                                                              stoplist=stoplist, min_count=min_count,
                                                              diff1_max=diff1_max, diff0_max=diff0_max)

        # scale
        print "Vectorized.  Saving"

        np.save(w2v_data_name, np.ascontiguousarray(w2v_data))

        pickle.dump(w2v_feature_crd, open(w2v_feature_crd_name, 'wb'))

        print "Scaling"

        w2v_data = scale_features(w2v_data, w2v_feature_crd)
        #dpgmm_data = scale_features(dpgmm_data, dpgmm_feature_crd)
        print "Scaled. Saving"

        if os.path.isfile(w2v_data_name+".npy"):
            os.remove(w2v_data_name+".npy")
        np.save(w2v_data_scaled_name, np.ascontiguousarray(w2v_data))
        np.save(y_data_name, np.ascontiguousarray(y_data))

        print "Building experiments"

    else:

        print("%s s: " % (time.time() - start_time))
        w2v_data = np.load(w2v_data_scaled_name+".npy", mmap_mode='c')
        # this need to be C-order array.
        y_data = np.load(y_data_name+".npy")

        print "loaded data %s" % w2v_data
        print("%s s: " % (time.time() - start_time))
        w2v_feature_crd = pickle.load(open(w2v_feature_crd_name, 'rb'))
        print "Loaded feature crd %s" % w2v_feature_crd
        train_data_end = int(p*1600000)
        logging.info("Loaded data, features.  %s " %  str(w2v_data.shape))



    names, experiments = build_experiments(w2v_feature_crd, experiment_nums=experiment_nums)
    print("%s s: " % (time.time() - start_time))

    print "Built experiments: ", names
    print experiments
    print action
    print train_data_end, w2v_data.shape

    with open(dataname + "_" + clf_base + "_fscore.txt", 'a') as f:
        for name, experiment in zip(names, experiments):
            print("%s s: " % (time.time() - start_time))
            print name, experiment
            logging.info("Experiment %s %s" % (name, str(experiment)))
            #inds = []
            #for start, stop in experiment:
            #    inds += (range(start, stop))
            # we will assume for the memory sake that the experiment is continious
            start = experiment[0][0]
            stop = experiment[0][1]

            if clf_base == "lr":
                clf = sklearn.linear_model.SGDClassifier(loss='log', penalty="l2",alpha=0.005, n_iter=5, shuffle=True)
            else:
                clf = SVC(kernel='linear', C=1)

            if action == "classify":

                if test_filename is not None:
                    scores = run_train_test_classifier(w2v_data, y_data, train_data_end, start, stop, clf=clf)

                    #scores = run_train_test_classifier(w2v_data[0:train_data_end, start:stop], y_data[0:train_data_end],
                    #                                   w2v_data[train_data_end:, start:stop], y_data[train_data_end:], clf=clf)
                else:
                    scores = run_cv_classifier(w2v_data[:, start:stop], y_data, clf=clf, n_trials=1, n_cv=5)
                print name, scores, scores.shape

                for i, score in enumerate(scores):
                    f.write("%i, %i,  %s, %i, %f, %f, %i, %i, %f, %f, %f, %f, %i \n" %
                           (n_trial, i, name, size, p, thresh, w2v_data.shape[0], w2v_data.shape[0]*(p+thresh-p+thresh),
                            score[0], score[1], score[2], score[3], n_components))
                f.flush()

            elif action == "explore":

                print np.bincount(y_data)
                explore_classifier(w2v_data[:, start:stop], y_data, clf=clf, n_trials=1, orig_data=zip(x_data, ids))

            elif action == "save":

                ioutils.save_liblinear_format_data (dataname + name+"_libl.txt", w2v_data[:, start:stop], y_data)


def build_w2v_model(w2v_corpus_list, dataname="", window=0, size=0, min_count=0, rebuild=False, explore=False):
    w2v_model_name = w2v_models.make_w2v_model_name(dataname=dataname, size=size, window=window,
                                                    min_count=min_count)
    logging.info("Looking for model %s" % w2v_model_name)
    if (not rebuild or explore) and os.path.isfile(w2v_model_name):
        w2v_model = w2v_models.load_w2v(w2v_model_name)
        logging.info("Model Loaded")
    else:
        w2v_corpus = np.array([tu.normalize_punctuation(text).split() for text in np.concatenate(w2v_corpus_list)])
        w2v_model = w2v_models.build_word2vec(w2v_corpus, size=size, window=window, min_count=min_count, dataname=dataname)
        logging.info("Model created")
    w2v_model.init_sims(replace=True)

    #check_w2v_model(w2v_model=w2v_model)
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




def build_and_vectorize_w2v(x_data=None, y_data=None, unlabeled_data=None, window=0, size=0, dataname="",
                        rebuild=False, action="classify", stoplist=None, min_count=1,
                        diff1_max=3, diff0_max=1):

    w2v_corpus = [x_data, unlabeled_data]
    if action == "explore":
        explore = True
    else:
        explore = False

    logging.info("Classifying %s, %i, %i" % (dataname, len(w2v_corpus), min_count,))
    # build models
    w2v_model = build_w2v_model(w2v_corpus, dataname=dataname, window=window, size=size, min_count=min_count,
                                rebuild=rebuild, explore=explore)

    # get features from models
    w2v = transformers.W2VTextModel(w2v_model=w2v_model, no_above=1.0, no_below=1, diffmax0=diff0_max, diffmax1=diff1_max)

    # get matrices of features from x_data
    w2v_data = w2v.fit_transform(x_data)

    print w2v_data.shape

    return w2v_data, w2v.feature_crd


def build_and_vectorize_dpgmm(x_data=None, y_data=None, unlabeled_data=None, dataname="", n_components=0,
                        rebuild=False, action="classify", stoplist=None, min_count=1, recluster_thresh=0,
                        no_above=0.9, no_below=5):

    w2v_corpus = np.array([tu.normalize_punctuation(text).split() for text in np.concatenate([x_data, unlabeled_data])])

    dpgmm = transformers.DPGMMClusterModel(w2v_model=None, n_components=n_components, dataname=dataname,
                                           stoplist=stoplist, recluster_thresh=0, no_above=no_above, no_below=no_below,
                                           alpha=5)

    pickle.dump(dpgmm, open(dataname+"_dpgmm", 'wb'))

    dpgmm.fit(w2v_corpus)
    dpgmm_data = dpgmm.transform(x_data)

    pickle.dump(dpgmm_data, open(dataname+"_dpgmm_data", 'wb'))


    return dpgmm_data, dpgmm.feature_crd


def scale_features(data, feature_crd):
    # scale features
#    for name, (start, end) in feature_crd.items():
#        data[:, start:end] = StandardScaler().fit_transform(data[:, start:end])

    data = StandardScaler(copy=False).fit_transform(data)
    print "scaled"
    return data

# experiment_nums = is a list of feature #s to add to experiments.
# experiment_nums = [0,1] to add [avg, std] to the list
def build_experiments(feature_crd, names_orig=None, experiment_nums=None):
    if names_orig is None:
        names_orig = sorted(feature_crd.keys())
    experiments = []
    print "Building experiments: ", experiment_nums
    names = []
    for name in names_orig:
        if (experiment_nums is None) or (int(name[:2]) in experiment_nums):
            names.append(name)
            experiments.append( [(0, feature_crd[name][1])])
    return names, experiments


def w2v_cluster_tweet_vocab(filename, window=0, size=0, dataname="", n_components=0, min_count=1,
                            rebuild=False):

    print "Clustering"
    x_data, y_data, stoplist, _ = make_x_y(filename, ["text"])
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
#    if dataname == "sentiment":
#    plotutils.plot_diff1_dep(dataname, withold=False)
    plotutils.plot_tweet_sentiment(dataname)
#    else:
#        plotutils.plot_kenyan_data(dataname)


def __main__():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', nargs='+', help='Filename')
    parser.add_argument('--test', action='store', dest='test_filename', default="", help='Filename')
    parser.add_argument('--dname', action='store', dest='dataname', default="log", help='Output filename')
    parser.add_argument('-n', action='store', dest='ntopics', default='10', help='Number of LDA topics')
    parser.add_argument('--size', action='store', dest='size', default='100', help='Size w2v of LDA topics')
    parser.add_argument('--window', action='store', dest='window', default='10', help='Number of LDA topics')
    parser.add_argument('--min', action='store', dest='min', default='1', help='Number of LDA topics')
    parser.add_argument('--nclusters', action='store', dest='nclusters', default='30', help='Number of LDA topics')
    parser.add_argument('--clusthresh', action='store', dest='clusthresh', default='0', help='Threshold for reclustering')
    parser.add_argument('--p', action='store', dest='p', default='1', help='Fraction of labeled data')
    parser.add_argument('--thresh', action='store', dest='thresh', default='0', help='Fraction of unlabelled data')
    parser.add_argument('--ntrial', action='store', dest='ntrial', default='0', help='Number of the trial')
    parser.add_argument('--clfbase', action='store', dest='clfbase', default='lr', help='Number of the trial')
    parser.add_argument('--clfname', action='store', dest='clfname', default='w2v', help='Number of the trial')
    parser.add_argument('--action', action='store', dest='action', default='plot', help='Number of the trial')
    parser.add_argument('--rebuild', action='store_true', dest='rebuild', help='Number of the trial')
    parser.add_argument('--exp_num', action='store', dest='exp_nums', nargs='+', help='Experiments to save')
    parser.add_argument('--diff1_max', action='store', dest='diff1_max', default='5', help='Diff 1 max')
    parser.add_argument('--diff0_max', action='store', dest='diff0_max', default='1', help='Diff 0 max')


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

    if arguments.test_filename != "":
        test_filename = arguments.test_filename
    else:
        test_filename = None

    if arguments.exp_nums:
        exp_nums = [int(n) for n in arguments.exp_nums]
    else:
        exp_nums = None


    # runs a classification experiement a given file
    if arguments.action == "classify" or arguments.action == "explore" or arguments.action == "save":
        if len(arguments.filename) > 1:
            tweet_classification(arguments.filename[0], size=size, window=window, dataname=arguments.dataname,
                             p=percentage, thresh=threshhold, n_trial=ntrial, min_count=min_count,
                             clf_name=arguments.clfname, unlabeled_filenames=arguments.filename[1:],
                             clf_base=arguments.clfbase, recluster_thresh=recluster_thresh,
                             rebuild=arguments.rebuild, action=arguments.action,
                             experiment_nums=exp_nums,
                             diff1_max=int(arguments.diff1_max), diff0_max=int(arguments.diff0_max))
        else:
            tweet_classification(arguments.filename[0], size=size, window=window, dataname=arguments.dataname,
                             p=percentage, thresh=threshhold, n_trial=ntrial, min_count=min_count,
                             clf_name=arguments.clfname, unlabeled_filenames=None,
                             clf_base=arguments.clfbase, recluster_thresh=recluster_thresh,
                             rebuild=arguments.rebuild, action=arguments.action,
                             experiment_nums=exp_nums, test_filename=test_filename,
                             diff1_max=int(arguments.diff1_max), diff0_max=int(arguments.diff0_max))

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
        print "plot"
        plot_scores(arguments.dataname)

    # merge a unlabeled dataset, with positive labels to produce a positively labeled dataset
    elif arguments.action == "make_labels":
        ioutils.make_positive_labeled_kenyan_data(arguments.dataname)

if __name__ == "__main__":
    __main__()
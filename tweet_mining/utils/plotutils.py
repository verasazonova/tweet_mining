__author__ = 'verasazonova'

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np
from six import string_types
from os.path import isfile
from operator import itemgetter
import logging
from sklearn.manifold import TSNE


def plot_words_distribution(word_vecs, n_topics, dataname=""):

    topic_vecs = np.zeros((n_topics, len(word_vecs[0])))
    for i in range(n_topics):
        topic_vecs[i] = np.sum(word_vecs[i*20:i*20+20])

    ts = TSNE(2)
    logging.info("Reducing with tsne")
    reduced_vecs = ts.fit_transform(topic_vecs)

    cmap = get_cmap(n_topics)

    plt.figure()

    for i in range(n_topics):
        plt.plot(reduced_vecs[i, 0], reduced_vecs[i, 1], marker='o', color=cmap(i), markersize=8, label=str(i))
    plt.legend()
    plt.savefig(dataname+"_words.pdf")


def get_cmap(n_colors):
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.
    """
    color_norm = colors.Normalize(vmin=0, vmax=n_colors-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='Spectral')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def plot_tweets(counts, dates, labels, clusters, dataname):

    time_labels = [date.strftime("%m-%d") for date in dates]

    n_topics = counts.shape[0]
    n_bins = counts.shape[1]
    ind = np.arange(n_bins)
    cmap = get_cmap(n_topics)

    width = 0.35
    totals_by_bin = counts.sum(axis=0)+1e-10

    log_totals = np.log10(totals_by_bin)

    plt.figure()

    plt.subplot(211)
    plt.plot(ind+width/2., totals_by_bin)
    plt.xticks([])
    plt.ylabel("Total twits")
    plt.xlim([ind[0], ind[n_bins-1]])
    plt.grid()

    plt.subplot(212)
    polys = plt.stackplot(ind, log_totals*counts/totals_by_bin, colors=[cmap(i) for i in range(n_topics)])

    legend_proxies = []
    for poly in polys:
        legend_proxies.append(plt.Rectangle((0, 0), 1, 1, fc=poly.get_facecolor()[0]))

    plt.ylabel("Topics.  % of total twits")
    plt.xticks((ind+width/2.)[::4], time_labels[::4], rotation=60)
    plt.xlim([ind[0], ind[n_bins-1]])
    plt.ylim([0, np.max(log_totals)])

    common_words = set(labels[0])
    for label in labels[1:]:
        common_words = common_words.intersection(set(label))

    logging.info("Words common to all labels: %s" % common_words)

    label_corpus = []

    clean_labels = []
    for i, label in enumerate(labels):
        legend = str(clusters[i]) + ", "
        legend_cnt = 0
        word_list = []
        for word in label:
            word_list.append(word)
            if word not in common_words:
                legend += word + " "
                legend_cnt += len(word) + 1
            if legend_cnt > 100:
                legend += '\n '
                legend_cnt = 0
        label_corpus.append(word_list)
        clean_labels.append(legend)

    logging.info("Saved in %s" % (dataname+".pdf"))
    plt.figlegend(legend_proxies, clean_labels, 'upper right', prop={'size': 6}, framealpha=0.5)
    plt.savefig(dataname+".pdf")


def extract_xy_average(data, xind, yind, cind, cval):
    data_c = data[data[:, cind] == cval]

    xvals = sorted(list(set(data_c[:, xind])))
    yvals = []
    yerr = []
    for xval in xvals:
        i = data_c[:, xind] == xval
        yvals.append(data_c[i, yind].mean())
        yerr.append(data_c[i, yind].std())
    return np.array(xvals), np.array(yvals), np.array(yerr)


def extract_data_series(data, xind, yind, cind, cval):
    data_c = data[data[:, cind] == cval]

    xvals = range(len(data_c))
    yvals = []
    yerr = []
    for xval in xvals:
        # using the row number
        i = xval
        yvals.append(data_c[i, yind].mean())
        yerr.append(data_c[i, yind].std())
    return np.array(xvals), np.array(yvals), np.array(yerr)


def extract_conditions(data, conditions=None):
    if conditions is None:
        return data
    data_c = data

    # a list of (ind, value) tuples or of (ind, [val1, val2, val3]) tuples
    for ind, val in conditions:
#        if isinstance(val, list):
#            tmp = []
#            for v in val:
#                tmp.append(data_c[data_c[:, ind] == val])
#                print tmp
 #           data_c = np.concatenate(tmp)
 #       else:
         data_c = data_c[data_c[:, ind] == val]

    return data_c


def plot_multiple_xy_averages(data_raw, xind, yind, cind, marker='o', cdict=None, conditions=None, witherror=False,
                              series=False, labels=None):

    data = extract_conditions(data_raw, conditions)

    cvals = sorted(list(set(data[:, cind])))
    for cval in cvals:

        np.set_printoptions(precision=4)  #formatter={'float': '{: 0.3f}'.format})
        if series:
            xvals, yvals, yerrs = extract_data_series(data, xind, yind, cind, cval)
            if cval in labels:
                print "%-10s" % labels[cval],
            else:
                print "%-10s" % cval,
            print "%.4f +- %.4f" % (yvals.mean(), yvals.std())
        else:
            xvals, yvals, yerrs = extract_xy_average(data, xind, yind, cind, cval)
            print cval, xvals, yvals, yerrs
        # if no color is supplied use black
        if cdict is None:
            color = 'k'
        # if cdict is a string - assume it is a color
        elif isinstance(cdict, string_types):
            color = cdict
        # by default cdict is a dictionary of color-value pairs
        else:
            color = cdict[cval]

        if labels is not None and cval in labels:
            label = labels[cval]
        else:
            label = cval
        if witherror:
            plt.errorbar(xvals, yvals, yerr=yerrs, fmt='-', marker=marker, color=color, label=label)
        else:
            plt.plot(xvals, yvals, '-', marker=marker, color=color, label=label)


def extract_base(data, xind, yind, cind, cval):

    ind = data[:, cind] == cval
    #xvals = [min(data[:, xind]), max(data[:, xind])]
    xvals = plt.xlim()
    yvals = [data[ind, yind].mean(), data[ind, yind].mean()]
    return xvals, yvals


def plot_multiple_bases(data_raw, xind, yind, cind, cdict=None, conditions=None, labels=None):

    data = extract_conditions(data_raw, conditions)

    cvals = sorted(list(set(data[:, cind])))

    for cval in cvals:
        xvals, yvals = extract_base(data, xind, yind, cind, cval)
        if cdict is None:
            color = 'k'
        elif isinstance(cdict, string_types):
            color = cdict
        else:
            color = cdict[cval]
        if labels is None or cval not in labels:
            label = cval
        else:
            label = labels[cval]
        plt.plot(xvals, yvals, '--', color=color, label=label)


def make_labels(title=""):
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    plt.xlabel("Length of w2v corpus (labeled + unlabeled data)")
    plt.ylabel("F-score for minority class")
    plt.ylim([0.64, 0.83])
    plt.title(title)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.1), ncol=3, fancybox=True, shadow=True)


def read_data(dataname, cind=2, cdict=None):
    types = {}
    def convert_str_type(s):
        s1 = s.strip()
        if s1 in types.keys():
            return types[s1]
        else:
            types[s1] = len(types) + 1
        return types[s1]

    converter = {2: convert_str_type}

    name = dataname+"_lr_fscore.txt"
    if isfile(name):
        data = np.loadtxt(name, delimiter=',', converters=converter)
        inv_types = dict([(v, k) for (k, v) in types.items()])
        if cdict is None:
            cdict = {}
        cvals = sorted(list(set(data[:, cind])) + cdict.keys())
        cmap = get_cmap(len(cvals))
        for i, cval in enumerate(cvals):
            cdict[cval] = cmap(i)
        return data, cdict, inv_types

    return None


def plot_diff1_dep(dataname, withold=False):


    data, cdict, names = read_data(dataname, 3)

    if withold:

        dataname_old = "../100D-vs-features/"+dataname
        data_old, cdict_old, names_old = read_data(dataname_old, 5)
        plot_multiple_xy_averages(data_old, 2, 8, 5, cdict=cdict_old, marker='o', witherror=True, series=False,
                                  labels={0.0: "w2v (100) 160,000", 0.4:"w2v(100) 160,000 + 640,000"})
        names = names_old
        cdict = cdict_old

    labels = [name[3:] for name in sorted(names.values())]
    plot_multiple_xy_averages(data, 2, 9, 3, cdict=cdict, marker='s',witherror=True, series=False,
                              labels={100: "w2v(100) precision", 200:"w2v(200) precision"})

    plot_multiple_xy_averages(data, 2, 10, 3, cdict=cdict, marker='o',witherror=True, series=False,
                              labels={100: "w2v(100) recall", 200:"w2v(200) recall"})

    plt.gca().set_xticks(range(1, len(labels)),)
    plt.gca().set_xticklabels(labels, rotation=45, ha='center')
    plt.gca().tick_params(axis='x', labelsize=8)
    plt.grid()

#    data_bow = np.loadtxt("../bow_f-scores-100-10.txt")
#    plot_multiple_bases(data_bow, 2, 3, 1, cdict='k', conditions=[(1, 0.1)], labels={0.1:'bow'})
    plt.ylabel("Precision / Recall ")
    plt.title("w2v vs features for different w2v sizes for mpeketoni ")
    plt.xlabel("W2V features")
    plt.legend(loc=4)
    plt.savefig(dataname+"_diff2.pdf")


def plot_tweet_sentiment(dataname):

    data, cdict, names = read_data(dataname, cind=4)

    data2, cdict, names2 = read_data("../2015-10-25-18-56/"+dataname, cind=4, cdict=cdict)

    #data = np.concatenate([data, data2])
    #names.update(names2)

    print names
    markers = ['o', '<', 's']
    for i, name in names.items():
        plt.figure()
        plt.gca().set_xscale("log", nonposx='clip')
        plot_multiple_xy_averages(data, 7, 8, 4, cdict=cdict, marker='s', witherror=True, series=False,
                                  conditions=[(2, i)],
                                  labels={0.001: "0.1%", 0.01: "1%", 0.5: "50%", 0.1: "10%", 1: "100%"})

        plt.grid()
        make_labels("Tweet sentiment data %s" % name[3:])
        plt.savefig(dataname + "_" + name + "_100_w2v.pdf")




    plt.figure()
    for i, t in enumerate([0, 0.1]):
        plot_multiple_xy_averages(data, 2, 8, 4, cdict=cdict, marker=markers[i], witherror=True, series=False,
                                  conditions=[(5, t)],
                                  labels={0.001: "0.1%% - %i%% " % (100*t), 0.01: "1%% - %i%%" % (t*100),
                                          0.5:"50%% - %i%% " % (100*t),
                                          0.1: "10%% - %i%% " % (100*t), 1: "100%% - %i%% " % (100*t)})
        labels = [name[3:] for name in sorted(names.values())]
        plt.grid()
        plt.gca().set_xticks(range(1, len(labels)),)
        plt.gca().set_xticklabels(labels, rotation=45, ha='center')
        plt.gca().tick_params(axis='x', labelsize=8)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.1), ncol=2, fancybox=True, shadow=True)
        plt.ylabel("Minority f-score")
        plt.title("Tweet sentiment data")
        plt.xlabel("Features")
    plt.savefig(dataname + "_features" + "_100_w2v.pdf")



def plot_kenyan_data(dataname):

    TYPES = {'0_avg': 0, '1_std': 1, '2_diff0_1':2, '3_diff0_2':3, '4_diff0_3':4,
             '5_diff1_1':5, '6_diff1_2':6, '7_diff1_3':7, 'cluster':8}
    converter = {2: lambda s: TYPES[s.strip()]}

    inv_types = dict([(v, k[2:]) for (k, v) in TYPES.items()])
    inv_types[8]= 'cluster'
    print inv_types

    for suffix in ["lr", "svm"]:

        filename = dataname + "_" + suffix + "_fscore.txt"
        if isfile(filename):
            data_lr = np.loadtxt(filename, delimiter=',', converters=converter)
            cvals = sorted(TYPES.values())
            cmap = get_cmap(len(cvals))
            cdict = {}
            for i, cval in enumerate(cvals):
                cdict[cval] = cmap(i)
#            plot_multiple_xy_averages(data_lr, 2, 8, 3, 's', cdict='b', witherror=False)

            for cval in [TYPES['0_avg'], TYPES['1_std'], TYPES['4_diff0_3'], TYPES['7_diff1_3'], TYPES['cluster']]:
                plot_multiple_xy_averages(data_lr, 0, 8, 2, '.', cdict=cdict, witherror=False, series=True,
                                      labels=inv_types, conditions=[(2, cval)])

            plt.grid()

            #data_base = np.loadtxt("../mpeketoni_svm_fscore bow_only_reference.txt", delimiter=',', converters=converter)
            #data_bow = np.concatenate([data_base for i in range(10)], axis=0)

            #plot_multiple_xy_averages(data_bow, 0, 8, 2, '.', cdict=cdict, witherror=False, series=True,
            #                          conditions=[(2, TYPES['bow'])])

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3,
                       fancybox=True, shadow=True)

            plt.ylabel("Minority f-score")
            plt.title("Mpeketoni dataset")
            plt.xlabel("5 trials of 5-fold xvalidation")

    plt.savefig(dataname+"w2v.pdf")

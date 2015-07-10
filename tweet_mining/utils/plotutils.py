__author__ = 'verasazonova'

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np
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

    fig = plt.figure()

    for i in range(n_topics):
        plt.plot(reduced_vecs[i, 0], reduced_vecs[i, 1], marker='o', color=cmap(i), markersize=8, label=str(i))
    plt.legend()
    plt.savefig(dataname+"_words.pdf")


def get_cmap(n_colors):
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.
    """
    color_norm = colors.Normalize(vmin=0, vmax=n_colors-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='Set1')

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

    fig = plt.figure()

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
    print cval
    data_c = data[data[:, cind]==cval]

    xvals = sorted(list(set(data_c[:, xind])))
    yvals = []
    for xval in xvals:
        i = data_c[:, xind] == xval
        print xval, data_c[i, yind]
        yvals.append( data_c[i, yind].mean())
    return np.array(xvals), np.array(yvals)


def plot_multiple_xy_averages(data, xind, yind, cind):

    cvals = sorted(list(set(data[:, cind])))
    cmap = get_cmap(len(cvals))

    for i, cval in enumerate(cvals):
        xvals, yvals = extract_xy_average(data, xind, yind, cind, cval)
        #print cval, xvals, yvals
        plt.plot(xvals, yvals, 'o-', color=cmap(i), label=cval)

    plt.legend(loc=4)


def extract_base(data, xind, yind, cind, cval):

    ind = data[:, cind]==cval
    #xvals = [min(data[:, xind]), max(data[:, xind])]
    xvals = plt.xlim()
    yvals = [data[ind, yind].mean(), data[ind, yind].mean()]
    return xvals, yvals


def plot_multiple_bases(data, xind, yind, cind):

    cvals = sorted(list(set(data[:, cind])))
    cmap = get_cmap(len(cvals))

    for i, cval in enumerate(cvals):
        xvals, yvals = extract_base(data, xind, yind, cind, cval)
        plt.plot(xvals, yvals, '--', color=cmap(i))

    plt.xlabel("W2V corpus")
    plt.ylabel("F-Score")
    plt.title("W2V (s=100, w=10) vs. BOW")
    plt.savefig("test.pdf")


def plot_curves_baseslines(data, xind, yind, curve_ind, baseline_ind):

    fig = plt.figure()
    plt.axis()

    plot_multiple_xy_averages(data, xind, yind, curve_ind)
    plot_multiple_bases(data, xind, yind, baseline_ind)

    plt.legend()
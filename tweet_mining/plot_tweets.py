__author__ = 'verasazonova'

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np
import logging
import dateutil.parser


def read_counts_bins_labels(dataname):
    counts = np.loadtxt(dataname+"_cnts.txt")
    bin_lows = []
    with open(dataname+"_bins.txt", 'r') as f:
        for line in f:
            bin_lows.append(dateutil.parser.parse(line.strip()))
    topic_definitions = []
    with open(dataname+"_labels.txt", 'r') as f:
        for line in f:
            topic_definitions.append(line.strip())
    topics = []
    with open(dataname+"_labels_weights.txt", 'r') as f:
        for line in f:
            topics.append([(tup.split(',')[1], tup.split(',')[0]) for tup in line.strip().split(' ')])

    return counts, bin_lows, topic_definitions, topics


def get_cmap(n_colors):
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.
    """
    color_norm = colors.Normalize(vmin=0, vmax=n_colors-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='Set1')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def plot_tweets(counts, dates, labels, topics, dataname):

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

    common_words = set(labels[0].split())
    for label in labels[1:]:
        common_words = common_words.intersection(set(label.split()))

    logging.info("Words common to all labels: %s" % common_words )

    label_corpus = []

#    calculate_topics_similarity(topics)

    clean_labels = []
    for label in labels:
        legend = ""
        legend_cnt = 0
        word_list = []
        for word in label.split():
            word_list.append(word)
            if word not in common_words:
                legend += str(word) + " "
                legend_cnt += len(word) + 1
            if legend_cnt > 100:
                legend += '\n '
                legend_cnt = 0
        label_corpus.append(word_list)
        clean_labels.append(legend)

#    legend_dictionary = Dictionary(label_corpus)
#    legend_dictionary.filter_extremes(no_below=1, no_above=0.95)

#    legend_bow_corpus = [legend_dictionary.doc2bow(text) for text in label_corpus]
#    for label1 in legend_bow_corpus:
#        for l

    plt.figlegend(legend_proxies, clean_labels, 'upper right', prop={'size': 6}, framealpha=0.5)
    plt.savefig(dataname+".pdf")

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
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='Set1') #'Spectral')

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
    n = []
    for xval in xvals:
        i = data_c[:, xind] == xval
        yvals.append(data_c[i, yind].mean())
        yerr.append(data_c[i, yind].std())
        n.append(len(data_c[i, yind]))
    return np.array(xvals), np.array(yvals), np.array(yerr), n


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
                              line='-',series=False, labels=None, ax=None):

    data = extract_conditions(data_raw, conditions)

    cvals = sorted(list(set(data[:, cind])))
    for cval in cvals:

        np.set_printoptions(precision=4)  #formatter={'float': '{: 0.3f}'.format})
        if series:
            xvals, yvals, yerrs = extract_data_series(data, xind, yind, cind, cval)
            if labels is not None and cval in labels:
                print "%-10s" % labels[cval],
            else:
                print "%-10s" % cval,
            print "%.4f +- %.4f" % (yvals.mean(), yvals.std())
        else:
            xvals, yvals, yerrs, n = extract_xy_average(data, xind, yind, cind, cval)
            print cval, n, xvals, yvals, yerrs
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
        if ax is None:
            ax = plt.gca()
        if witherror:
            ax.errorbar(xvals, yvals, yerr=yerrs, fmt=line, marker=marker, color=color, label=label, elinewidth=0.3,
                        markersize=5)
        else:
            ax.plot(xvals, yvals, line, marker=marker, color=color, label=label)


def extract_base(data, xind, yind, cind, cval):

    ind = data[:, cind] == cval
    #xvals = [min(data[:, xind]), max(data[:, xind])]
    xvals = plt.xlim()
    yvals = [data[ind, yind].mean(), data[ind, yind].mean()]
    return xvals, yvals


def plot_multiple_bases(data_raw, xind, yind, cind, cdict=None, conditions=None, labels=None, ax=None):

    data = extract_conditions(data_raw, conditions)

    cvals = sorted(list(set(data[:, cind])))

    if ax is None:
        print "using gca"
        ax = plt.gca()
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
        ax.plot(xvals, yvals, ':', color=color, label=label)


def make_labels(title=""):
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    plt.xlabel("Length of w2v corpus (labeled + unlabeled data)")
    plt.ylabel("F-score")
    plt.ylim([0.66, 0.85])
    plt.xlim([-1e5, 1.8e6])
    plt.title(title)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fancybox=True, shadow=True)


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

    data_bow, _, _ = read_data("../BOW/sentiment", cind=4)  #= np.loadtxt("../bow_f-scores-100-10.txt")
    y_ind = 11

    data[:, 7] = 100 * (data[:, 4] + data[:, 5] - data[:, 4] * data[:, 5])  # 1600359

    colors=['r', 'g', 'b']

    size=100

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3.5))
    plt.tight_layout()
    ax1, ax2 = axes.ravel()

    labels = [r'$\mu$',r'$\sigma$']
    for i in range(1,10):
        labels.append(r'$\tau^%i$' % i)
        labels.append(r'$\tau_\sigma^%i$' % i)

    for i in range(data.shape[0]):
        if data[i, 4] == 1:
            data[i, 5] = 0
    ls_100 = dict( [(p, "%i%% (%i)" % (100*p, 100)) for p in [0.1, 0.5, 1]])
    ls_300 = dict( [(p, "%i%% (%i)" % (100*p, 300)) for p in [0.1, 0.5, 1]])

    for p in [0.1, 0.5, 1]:
        plot_multiple_xy_averages(data, 2, y_ind, 4, cdict=cdict, marker='.', witherror=True, series=False,
                              conditions=[(5, 0), (3, 100), (4, p)], ax=ax1, labels=ls_100)
        plot_multiple_xy_averages(data, 2, y_ind, 4, cdict=cdict, marker='.', line='--', witherror=True, series=False,
                              conditions=[(5, 0), (3, 300), (4, p)], ax=ax1, labels=ls_300)
    #ax1.grid()
    ax1.set_ylim([0.74, 0.79])
    ax1.set_xlim([0, len(labels)])
    ax1.set_xticks(range(1, len(labels)),)
    ax1.set_xticklabels(labels, rotation=0, ha='center')
    ax1.tick_params(axis='x', labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.set_ylabel("F-score", fontsize='x-small')
    ax1.set_xlabel("Features",fontsize='x-small')
    # get handles
    handles, ls = ax1.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # use them in the legend
    ax1.legend(handles,ls, loc='upper center', ncol=3, fancybox=True, shadow=False, fontsize='xx-small',
               bbox_to_anchor=(0.5, 1.1))


    print names
    for t,ax,c in zip([2], [ax2], colors):
        #plt.figure()

        #f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
        #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
        #f.subplots_adjust(hspace=0, wspace=0)
        #plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

        ax1.axvspan(t-0.15, t+0.15, facecolor=c, alpha=0.3, edgecolor='none')

        plot_multiple_xy_averages(data, 7, y_ind, 4, cdict=cdict, marker='s', witherror=True, series=False,
                                  conditions=[(2, t), (3, 100)], ax=ax,
                                  labels={0.001: "0.1%", 0.01: "1%", 0.1: "10%", 0.5:"50%", 1:"100%"})
        plot_multiple_xy_averages(data, 7, y_ind, 4, cdict=cdict, marker='v', witherror=True, series=False, line='--',
                                  conditions=[(2, t), (3, 300)], ax=ax,
                                  labels={0.001: "0.1%", 0.01: "1%", 0.1: "10%", 0.5:"50%", 1:"100%"})

        ax.text(1, 0.77, labels[t-1], bbox=dict(facecolor=c, alpha=0.3, boxstyle="round,pad=.2",))

        ax.set_ylim([0.55, 0.79])
        ax.set_xlim([-5, 105])
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xlabel("W2v corpus size (% of total)",fontsize='x-small')
        ax.set_ylabel("F-score", fontsize='x-small')
        #plt.legend(loc="lower right",  ncol=2, fancybox=True, shadow=True, fontsize='x-small')
        #plt.savefig("%s_%i_%i.pdf" % (dataname, size, t))

    plot_multiple_bases(data_bow, 7, y_ind, 4, cdict=cdict, ax=ax2)
    #plot_multiple_bases(data_bow, 2, 3,1, cdict=cdict, ax=ax3)
    #plot_multiple_bases(data_bow, 2, 3,1, cdict=cdict, ax=ax4)
    plt.savefig("all.pdf")

    size=100

    lines = ['-', '-']

    #plt.figure()
    for j, t in enumerate([0, 1]):
        for i in range(data.shape[0]):
            if data[i, 4] == 1:
                data[i, 5] = t
        plt.figure()
        f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

        plot_multiple_xy_averages(data, 2, y_ind, 4, cdict=cdict, marker='o', witherror=True, series=False,
                                  conditions=[(5, t), (3, 100)], ax=ax1)
                                  #labels={100:"%i%%-%i (100)" % (p,t), 300: "%i%% -%i(300)" % (p,t)})
        #plt.xlim([0, 18])
        plt.ylabel("F-score")

        plot_multiple_xy_averages(data, 2, y_ind, 4, cdict=cdict, marker='o', witherror=True, series=False,
                                  conditions=[(5, t), (3, 300)], ax=ax2,
                                  labels={0.001: "0.1%", 0.01: "1%", 0.1: "10%", 0.5:"50%", 1:"100%"})
                                  #labels={0.001: "0.1%% - %i%% " % (100*t), 0.01: "1%% - %i%%" % (t*100),
                                  #        0.5:"50%% - %i%% " % (100*t),
                                  #        0.1: "10%% - %i%% " % (100*t), 1: "100%% - %i%% " % (100*t)})
#        labels = [name[3:] for name in sorted(names.values())]
        labels = [r'$\mu$',r'$\sigma$']
        for i in range(1,10):
            labels.append(r'$\tau^%i$' % i)
            labels.append(r'$\tau_\sigma^%i$' % i)
        #plt.ylim([0.7, 0.79])
        plt.xlim([0, len(labels)])
        ax1.text(1, 0.78, " 100D ", bbox=dict(facecolor='white', alpha=1, boxstyle="round,pad=.2",))
        ax2.text(1, 0.78, " 300D ", bbox=dict(facecolor='white', alpha=1, boxstyle="round,pad=.2",))
        plt.gca().set_xticks(range(1, len(labels)),)
        plt.gca().set_xticklabels(labels, rotation=0, ha='center')
        plt.gca().tick_params(axis='x', labelsize=12)
        plt.ylabel("F-score")
        #plt.title("%i unlabelled data" % t)
        plt.xlabel("Features")
#            plt.ylim([0.65, 0.89])
        ax1.grid()
        ax1.set_ylim([0.7, 0.79])
        ax2.grid()
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fancybox=True, shadow=True)
        plt.savefig("%s_features_%i_w2v.pdf" % (dataname, t))
    #a = plt.axes([0.2, 0.1, .4, .4], axisbg='w')
    #p=100
    #t=1
    #plot_multiple_xy_averages(data, 2, 8, 3, cdict=cdict, marker=markers[i], line=lines[j], witherror=True, series=False,
    #                          conditions=[(5, t), (4, p)],
    #                          labels={100:"%i%%-%i (100)" % (p,t), 300: "%i%% -%i(300)" % (p,t)})

    #plt.savefig("%s_features_w2v.pdf" % (dataname))



def plot_kenyan_data(dataname):

    TYPES = {'0_avg': 0, '1_std': 1, '2_diff0_1':2, '3_diff0_2':3, '4_diff0_3':4,
             '5_diff1_1':5, '6_diff1_2':6, '7_diff1_3':7, 'cluster':8}
    converter = {2: lambda s: TYPES[s.strip()]}

    inv_types = dict([(v, k[2:]) for (k, v) in TYPES.items()])
    inv_types[8]= 'cluster'
    print inv_types

    series_data, cdict, names = read_data("./100D-features-series/mpeketoni", cind=2)

    features_data, cdict, names = read_data("./100D-diff0-diff1/mpeketoni", cind=2)
    plt.figure()

    plot_multiple_xy_averages(features_data, 2, 8, 4, cdict='r', marker='o', witherror=True, series=False)
    labels = [r'$\mu$',r'$\sigma$']
    for i in range(1,6):
        labels.append(r'$\kappa^%i$' % i)
        labels.append(r'$\kappa_\sigma^%i$' % i)
    for i in range(1,6):
        labels.append(r'$\tau^%i$' % i)
        labels.append(r'$\tau_\sigma^%i$' % i)
    plt.gca().set_xticks(range(1, len(labels)),)
    plt.gca().set_xticklabels(labels, rotation=0, ha='center')
    plt.xlim([0, len(labels)])
    plt.xlabel("Features")
    plt.ylabel("Minority f-score")
    plt.grid()
    plt.ylim([0.56, 0.86])

    ax = plt.axes([0.45, 0.17, .4, .4], axisbg='w')

    cmap = get_cmap(5)
    for i, key in enumerate([1,2,5,7,8]):
        cdict[key] = cmap(i)

    labels={1:r'$\mu$', 2:r'$\sigma$', 5:r'$\kappa^3$', 7:r'$\tau^3$', 8:'cluster'}
    for t in [1,2,5,7,8]:
        plot_multiple_xy_averages(series_data, 0, 8, 2, '.', cdict=cdict, witherror=False, series=True,
                              labels=labels, conditions=[(2, t)], ax=ax)

#    ax.grid()

    #data_base = np.loadtxt("../mpeketoni_svm_fscore bow_only_reference.txt", delimiter=',', converters=converter)
    #data_bow = np.concatenate([data_base for i in range(10)], axis=0)

    #plot_multiple_xy_averages(data_bow, 0, 8, 2, '.', cdict=cdict, witherror=False, series=True,
    #                          conditions=[(2, TYPES['bow'])])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True, shadow=True, fontsize='x-small')

    ax.set_ylabel("Minority f-score", fontsize='x-small')
    ax.set_xlabel("5 trials of 5-fold xvalidation", fontsize='x-small')

    plt.savefig(dataname+".pdf")

    data, cdict, names = read_data("./2015-10-11-12-10/mpeketoni2", cind=3)

    print names

    plt.figure()
    for size in [100, 300]:
        plot_multiple_xy_averages(data, 2, 9, 3, cdict=cdict, marker='s', witherror=True, series=False,
                                  conditions=[(3, size)],
                                  labels={100: "Precision (100)", 300: "Precision (300)"})

        plot_multiple_xy_averages(data, 2, 10, 3, cdict=cdict, marker='o', witherror=True, series=False,
                                  conditions=[(3, size)],
                                  labels={100: "Recall (100)", 300: "Recall (300)"})


    labels = [r'$\mu$',r'$\sigma$']
    for i in range(1,6):
        labels.append(r'$\tau^%i$' % i)
        labels.append(r'$\tau_\sigma^%i$' % i)
    plt.gca().set_xticks(range(1, len(labels)),)
    plt.gca().set_xticklabels(labels, rotation=0, ha='center')
    plt.xlim([0, len(labels)-1])
    plt.xlabel("Features", fontsize='x-small')
    plt.ylabel("Minority precision and recall", fontsize='x-small')
    plt.grid()
    #plt.ylim([0.56, 0.86])
    plt.legend(loc='upper left', ncol=2, fancybox=True, shadow=True, fontsize='x-small')

    ax = plt.axes([0.52, 0.17, .36, .36], axisbg='w')
    plot_multiple_xy_averages(features_data, 2, 8, 4, cdict='b', marker='o', witherror=True, series=False, ax=ax)
    labels = [r'$\mu$']
    for i in range(1,6):
        labels.append(r'$\kappa^%i$' % i)
        #labels.append(r'$\kappa_\sigma^%i$' % i)
    for i in range(1,6):
        labels.append(r'$\tau^%i$' % i)
        #labels.append(r'$\tau_\sigma^%i$' % i)
    ax.set_xticks(range(1, 2*len(labels), 2))
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlim([0, 2*len(labels)])
    ax.set_xlabel("Features", fontsize='x-small')
    ax.set_ylabel("Minority f-score", fontsize='x-small')
    ax.grid(axis='y')
    ax.set_ylim([0.56, 0.86])



    plt.savefig("mpeketoni_precision.pdf")
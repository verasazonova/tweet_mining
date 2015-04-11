__author__ = 'verasazonova'

import csv, sys, re
import matplotlib.colors as colors
import matplotlib.cm as cmx
import argparse
import numpy as np
import os
from os.path import basename
import matplotlib.pyplot as plt
import dateutil.parser
import codecs
from gensim.matutils import corpus2dense
from datetime import datetime
import os.path
from operator import itemgetter
from itertools import groupby
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer, TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.cluster import DBSCAN
from gensim.models import LdaMallet, LdaModel, TfidfModel
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases

# add space around punctuation
def normalize_punctuation(phrase):
    norm_phrase = unicode(phrase.lower(), errors='replace').strip()
    #delete url
    norm_phrase = re.sub(r'http(\S*)\B', "", norm_phrase)

    #delete known unreadable characters
    norm_phrase = re.sub(u"\uFFFD", "", norm_phrase)
    norm_phrase = re.sub(r'\&gt', "", norm_phrase)
    norm_phrase = re.sub(r'\&lt', "", norm_phrase)
    norm_phrase = re.sub(r'\&amp', "", norm_phrase)

    for punctuation in [',', ':', '.', '(', ')', '!', '?', ':', ';', '/', '\"', '*', '^', '\'']:
        norm_phrase = norm_phrase.replace(punctuation, ' ' + punctuation+' ')
    return norm_phrase

#remove one letter words
def normalize_words(words_list, stoplist):
    norm_list = [word for word in words_list if len(word) > 1 and word not in stoplist+[u'rt']]
    return norm_list

class KenyanCSVMessage():
    def __init__(self, filename, split=True, date_pos=10, text_pos=7, include_date=True, stop_path=None):
        self.filename = filename
        self.split = split
        self.date_pos = date_pos
        self.text_pos = text_pos
        self.include_date = include_date

        if os.path.isfile(stop_path):
            print "Using %s as stopword list" % stop_path
            self.stoplist = [word.strip() for word in open(stop_path, 'r').readlines()]
        else:
            self.stoplist = []

    def __iter__(self):
        with open(self.filename, 'r') as f:
            reader = csv.reader(f, dialect=csv.excel)
            reader.next()
            try:
                for row in reader:
                    facebook_comment_text = normalize_punctuation(normalize_format(row[self.text_pos]))
                    facebook_date = row[self.date_pos]
                    if self.include_date:
                        yield ( facebook_comment_text, facebook_date)
                    else:
                        if self.split:
                            yield normalize_words(facebook_comment_text.split(), self.stoplist)
                        else:
                            yield facebook_comment_text

            except csv.Error as e:
                sys.exit('file %s, line %d: %s' % (self.filename, reader.line_num, e))



def print_columnt(filename, column_num):
    with open(filename, 'r') as f:
        reader = csv.reader(f, dialect=csv.excel)
        try:
            for row in reader:
                print row[column_num]
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (filename, reader.line_num, e))


def normalize_format(phrase):
    # remove carriage return
    norm_phrase = phrase.replace('\r', '').replace('\n', ' ')
    return norm_phrase


def read_csv_twit_file(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f, dialect=csv.excel, )
        reader.next()
        try:
            for row in reader:
                print "line %s: %s" % (row[0], repr(row[2].replace('\r', '').replace('\n', ' ')))
        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (filename, reader.line_num, e))


def cpr_clusters(X):
        clusters = []
        for vecs in X:
            #clusterVec = [np.zeros(vecs[0].shape,)]         # tracks sum of vectors in a cluster
            #clusterIdx = [[0]]         # array of index arrays. e.g. [[1, 3, 5], [2, 4, 6]]
            clusterVec = []
            clusterIdx = []
            ncluster = 0
            # probablity to create a new table if new customer
            # is not strongly "similar" to any existing table
            pnew = 1.0/ (1 + ncluster)
            N = len(vecs)
            #rands = self.random_state_.rand(N)
            rands = rands / np.max(rands)

            for i, v in enumerate(vecs):
                maxSim = -np.inf
                maxIdx = 0
                #v = vecs[i]
                for j in range(ncluster):
                    sim = cosine_similarity(v, clusterVec[j])[0][0]
                    if sim >= maxSim:
                        maxIdx = j
                        maxSim = sim
                if maxSim < pnew:
                    if rands[i] < pnew:
                        clusterVec.append(v)
                        clusterIdx.append([i])
                        ncluster += 1
                        pnew = 1.0 / (1 + ncluster)
                    continue
                clusterVec[maxIdx] = clusterVec[maxIdx] + v
                clusterIdx[maxIdx].append(i)
            clusters.append(clusterIdx)
        return clusters



def calculate_topics_similarity(topics, stop_list=None):

    topics_txt = [ [tup[0] for tup in topic] for topic in topics]
    topics_dict = Dictionary(topics_txt)
    bow_topics = [topics_dict.doc2bow(text) for text in topics_txt]

    id2token = dict((v, k) for k, v in  topics_dict.token2id.iteritems())

    new_bow_topics = []
    for i, text in enumerate(bow_topics):
        bow_weights = []
        for id, count in text:
            weight_dict = dict(topics[i])
            token = id2token[id]
            weight = float(weight_dict[token])
            bow_weights.append( (id, count*weight) )
        new_bow_topics.append(bow_weights)
    print bow_topics
    print new_bow_topics

    tfidf_model = TfidfModel(new_bow_topics, id2word=id2token, dictionary=topics_dict)

    data = corpus2dense(tfidf_model[new_bow_topics],num_terms=len(topics_dict), num_docs=len(bow_topics)).T

    print data.shape

    topic_similarities = cosine_similarity(data)
    #topic_distances = cosine_distances(bow_topics.toarray())
    print topic_similarities

    thresh = 0.15
    N = len(topic_similarities)
    for i in range(N):
        print i,
        for j in range(N):
            if topic_similarities[i][j] > thresh:
                print j,
        print


    #db = DBSCAN(eps=0.3, min_samples=3, metric='precomputed', algorithm='auto').fit(topic_distances)
    #print db.labels_
    #print db.core_sample_indices_


def calculate_lda(dataset_raw, n_topics=10, lda_model_name="",
                  mallet=True, mallet_path="/Users/verasazonova/no-backup/JARS/mallet-2.0.7/bin/mallet",
                  dataname="none"):

    with open(dataname+"_log.txt", 'a') as fout:

        if dataset_raw.include_date:
            dates = [text[1] for text in dataset_raw]
            dataset = [normalize_words(text[0].split(), dataset_raw.stoplist) for text in dataset_raw]
        else:
            dates = ["" for _ in dataset_raw]
            dataset = dataset_raw

        bi_grams = Phrases(dataset, threshold=20)
        tri_grams = Phrases(bi_grams[dataset], threshold=40)
        dataset = tri_grams[bi_grams[dataset]]


        dictionary = Dictionary(dataset)
        dictionary.filter_extremes(no_below=1, no_above=0.9)

        bow_corpus = [dictionary.doc2bow(text) for text in dataset]
        fout.write("# Topics: %s\n" % n_topics)

        if not os.path.isfile(lda_model_name):

            if mallet:
                lda_model = LdaMallet(mallet_path, corpus=bow_corpus, num_topics=n_topics, id2word=dictionary, workers=4,
                                     optimize_interval=10, iterations=1000, prefix=os.path.join(os.getcwd(), 'mallet/'))
                lda_model_name = "lda_model_mallet_%s_%i" % (dataname, n_topics)
            else:
                lda_model = LdaModel(bow_corpus, id2word=dictionary, num_topics=n_topics, distributed=False,
                                    chunksize=2000, passes=5, update_every=10, alpha='asymmetric',
                                    eta=0.1, decay=0.5, eval_every=10, iterations=1000, gamma_threshold=0.001)

                lda_model_name = "lda_model_%s_%i" % (dataname, n_topics)

            lda_model.save(lda_model_name)

        else:
            if mallet:
                lda_model = LdaMallet.load(lda_model_name)
            else:
                lda_model = LdaModel.load(lda_model_name)


        # get topic definitions
        topic_definition = []
        for i, topic in enumerate(lda_model.show_topics(n_topics, num_words=20, formatted=False)):
            fout.write("%i \n" % i)
            topic_list = []
            freq_list = []
            a_list = []
            for tup in topic:
                topic_list.append(tup[1])
                freq_list.append(dictionary.dfs[ dictionary.token2id[tup[1]] ] )
                a_list.append(tup[0])


            fout.write( "%s\n\n" % repr((sorted(zip(topic_list, freq_list), key=itemgetter(1) ))))
            topic_definition.append("%i, %s" %(i, repr(" ".join(sorted(topic_list)))[2:-1]))
        fout.write("Total number of documents: %i\n" % dictionary.num_docs )


        earliest_date = dateutil.parser.parse("Sun Jun 08 00:00:00 +0000 2014")

        corpus_date_sorted = [tup for tup in  sorted(zip(bow_corpus, dates), key=get_date )
             if dateutil.parser.parse(tup[1]) > earliest_date]

        print len(corpus_date_sorted)
        print corpus_date_sorted[len(corpus_date_sorted)-1]
        latest_date = dateutil.parser.parse(corpus_date_sorted[-1][1])

        num_bins = 100

        time_span = latest_date - earliest_date
        print time_span
        time_bin = time_span / num_bins
        print time_bin

        bin_lows = [earliest_date]
        bin_high = earliest_date + time_bin
        counts = [[0 for _ in range(n_topics)] for _ in range(num_bins+1)]
        i=0
        for topic_assignements, text in zip(lda_model[zip(*corpus_date_sorted)[0]], corpus_date_sorted):
            date_str = text[1]
            if date_str is not None:
                cur_date = dateutil.parser.parse(date_str)
                if cur_date >= bin_high:
                    i+=1
                    bin_lows.append(bin_high)
                    bin_high = bin_lows[len(bin_lows)-1] + time_bin
                #counts[i][max(topic_assignments, key=itemgetter(1))[0]] += 1
                for tup in topic_assignements:
                    counts[i][tup[0]] += tup[1]

        fout.write("Number of documents assigned mostly to the topic: \n")
        fout.write("%s\n" % counts)

        corpus_date_sorted = 1.*np.array(counts)

        np.savetxt(dataname+"_cnts.txt", corpus_date_sorted)
        with open(dataname+"_bins.txt", 'w') as fout:
            for date in bin_lows:
                fout.write("%s\n" % date)
        with open(dataname+"_labels.txt", 'w') as fout:
            for label in topic_definition:
                fout.write("%s\n" % label)
        with open(dataname+"_labels_weights.txt", 'w') as fout:
            for label in lda_model.show_topics(n_topics, num_words=20, formatted=False):
                for tup in label:
                    fout.write("%s,%s " % (tup[0], tup[1]))
                fout.write("\n")

        return corpus_date_sorted, bin_lows, topic_definition


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
            topics.append([ (tup.split(',')[1], tup.split(',')[0]) for tup in line.strip().split(' ')])

    return counts, bin_lows, topic_definitions, topics


def get_date(tup):
    return dateutil.parser.parse(tup[1])


def print_date_span(dataset, output_name="output.txt"):

    with open(output_name, 'a') as fout:

        a = sorted(dataset, key=get_date )
        earliest_date = dateutil.parser.parse(a[0][1])
        latest_date = dateutil.parser.parse(a[len(a)-1][1])

        fout.write("%s\n" % earliest_date)
        fout.write("%s\n" % latest_date )

        return  (earliest_date, latest_date)


def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='Set1')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color



def plot_twits(counts, dates, labels, topics, dataname):

    time_labels = [date.strftime("%m-%d") for date in dates]

    n_topics = counts.shape[1]
    n_bins = counts.shape[0]
    ind = np.arange(n_bins)
    cmap = get_cmap(n_topics)

    width = 0.35
    totals_by_bin = counts.sum(axis=1)+1e-10

    log_totals = np.log10(totals_by_bin)

    print totals_by_bin
    print ind.shape, counts.shape

    fig = plt.figure()

    plt.subplot(211)
    plt.plot(ind+width/2., totals_by_bin)
    plt.xticks([])
    plt.ylabel("Total twits")
    plt.xlim([ind[0], ind[n_bins-1]])
    plt.grid()

    plt.subplot(212)
    polys = plt.stackplot(ind, log_totals*counts.T/totals_by_bin, colors=[cmap(i) for i in range(n_topics)])

    legendProxies = []
    for poly in polys:
        legendProxies.append(plt.Rectangle((0, 0), 1, 1, fc=poly.get_facecolor()[0]))

    plt.ylabel("Topics.  % of total twits")
    plt.xticks((ind+width/2.)[::4], time_labels[::4], rotation=60)
    plt.xlim([ind[0], ind[n_bins-1]])
    plt.ylim([0, np.max(log_totals)])

    common_words = set(labels[0].split())
    for label in labels[1:]:
        common_words = common_words.intersection(set(label.split()))

    print common_words

    label_corpus = []

    calculate_topics_similarity(topics)


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

    legend_dictionary = Dictionary(label_corpus)
    legend_dictionary.filter_extremes(no_below=1, no_above=0.95)

#    legend_bow_corpus = [legend_dictionary.doc2bow(text) for text in label_corpus]
#    for label1 in legend_bow_corpus:
#        for l

    plt.figlegend(legendProxies, clean_labels, 'upper right', prop={'size':6}, framealpha=0.5)
    plt.savefig(dataname+".pdf")


def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]

def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')


def clean_save_tweets(filename):
        labels=[]
        #read file
        with codecs.open(filename, 'r', encoding='utf-8') as f:
            reader = unicode_csv_reader(f, dialect=csv.excel)
            labels = reader.next()
            id_pos = labels.index('id_str')
            text_pos = labels.index('text')
            date_pos = labels.index('created_at')
            retweet_cnt_pos = labels.index('retweet_count')
            retweeted_pos = labels.index('retweeted')
            print text_pos, date_pos
            try:
                data = [row for row in reader]
            except csv.Error as e:
                sys.exit('file %s: %s' % (filename, e))

        print data[0][date_pos], data[0][id_pos]

        #sort tweets by date
        data_sorted = sorted(data, key=itemgetter(id_pos))

        print data_sorted[0][date_pos], data_sorted[0][id_pos]

        print "data_sorted"

        #save sorted file with newline charactes removed from the text
        with codecs.open(basename(filename).split('.')[0]+"_sorted_short.csv", 'w', encoding='utf-8') as fout:
            # write the data
            for row in data_sorted:
                fout.write("%s,\"%s\",\"%s\"\n" % ( row[id_pos], row[date_pos], normalize_format(row[text_pos])))


def get_statistics(filename):
        with codecs.open(filename, 'r', encoding='utf-8') as f:
            reader = unicode_csv_reader(f, dialect=csv.excel)
            labels = reader.next()
            id_pos = labels.index('id_str')
            text_pos = labels.index('text')
            date_pos = labels.index('created_at')
            retweet_cnt_pos = labels.index('retweet_count')
            retweeted_pos = labels.index('retweeted')
            try:
                data = [row for row in reader]
            except csv.Error as e:
                sys.exit('file %s: %s' % (filename, e))

            originals = [key for key, group in groupby(data, lambda x: x[text_pos])]

            print len(originals), len(data)


def calculate_and_plot_lda(filename, dataname, ntopics, ldamodelname):
    stop_path = "/Users/verasazonova/Work/PycharmProjects/textsim/corpus/en_swahili.txt"

    dataset = KenyanCSVMessage(filename, date_pos=1, text_pos=2, include_date=True, split=True,
                               stop_path=stop_path)

    print "Data set read"

    #print_columnt(arguments.filename, int(arguments.date_pos))

    #for x in dataset:
    #    continue


    #counts = np.loadtxt("data.txt")
    dates = [dateutil.parser.parse("Sun Jun 08 00:00:00 +0000 2014"),
             dateutil.parser.parse("Sun Jun 09 00:00:00 +0000 2014"),
             dateutil.parser.parse("Sun Jun 10 00:00:00 +0000 2014"),
             dateutil.parser.parse("Sun Jun 11 00:00:00 +0000 2014"),
             dateutil.parser.parse("Sun Jun 12 00:00:00 +0000 2014"),
             dateutil.parser.parse("Sun Jun 13 00:00:00 +0000 2014"),
             dateutil.parser.parse("Sun Jun 14 00:00:00 +0000 2014"),
             dateutil.parser.parse("Sun Jun 15 00:00:00 +0000 2014"),
             dateutil.parser.parse("Sun Jun 16 00:00:00 +0000 2014"),
             dateutil.parser.parse("Sun Jun 17 00:00:00 +0000 2014"),
             dateutil.parser.parse("Sun Jun 18 00:00:00 +0000 2014"),
             dateutil.parser.parse("Sun Jun 19 00:00:00 +0000 2014"),
             dateutil.parser.parse("Sun Jun 20 00:00:00 +0000 2014"),
             dateutil.parser.parse("Sun Jun 21 00:00:00 +0000 2014"),
             dateutil.parser.parse("Sun Jun 22 00:00:00 +0000 2014")]


    #labels = range(15)

    #counts, dates, labels =  calculate_lda(dataset, n_topics=int(ntopics),
    #                                       lda_model_name=ldamodelname,
    #                                       mallet=False, dataname=dataname)

    counts, dates, labels, topics = read_counts_bins_labels(dataname)

    print topics

    plot_twits(counts, dates, labels, topics, dataname)


def __main__():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', help='Filename')
    parser.add_argument('--fout', action='store', dest='outputname', help='Output filename')
    parser.add_argument('--ldamodel', action='store', dest='ldamodelname', default="", help='Lda model filename')
    parser.add_argument('--tpos', action='store', dest='text_pos', default='0', help='Text position')
    parser.add_argument('--dpos', action='store', dest='date_pos', default='0', help='Date position')
    parser.add_argument('-n', action='store', dest='ntopics', default='10', help='Number of LDA topics')

    arguments = parser.parse_args()

#    clean_save_tweets(arguments.filename)

    #get_statistics(arguments.filename)

    calculate_and_plot_lda(arguments.filename, arguments.outputname, arguments.ntopics, arguments.ldamodelname)

if __name__ == "__main__":
    __main__()
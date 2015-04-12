__author__ = 'verasazonova'

import numpy as np
from numpy.random import random
import os
from gensim.matutils import corpus2dense
import os.path
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import LdaMallet, LdaModel, TfidfModel
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases
import textutils as tu
import codecs
import logging


#from sklearn.cluster import DBSCAN
#from operator import itemgetter
#from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer, TfidfVectorizer, CountVectorizer


def crp_clusters(data):
        clusters = []
        for vecs in data:
            #clusterVec = [np.zeros(vecs[0].shape,)]         # tracks sum of vectors in a cluster
            #clusterIdx = [[0]]         # array of index arrays. e.g. [[1, 3, 5], [2, 4, 6]]
            cluster_vec = []
            cluster_idx = []
            ncluster = 0
            # probablity to create a new table if new customer
            # is not strongly "similar" to any existing table
            pnew = 1.0 / (1 + ncluster)
            #rands = self.random_state_.rand(N)
            rands = random.rand

            for i, v in enumerate(vecs):
                max_sim = -np.inf
                max_idx = 0
                #v = vecs[i]
                for j in range(ncluster):
                    sim = cosine_similarity(v, cluster_vec[j])[0][0]
                    if sim >= max_sim:
                        max_idx = j
                        max_sim = sim
                if max_sim < pnew:
                    if rands[i] < pnew:
                        cluster_vec.append(v)
                        cluster_idx.append([i])
                        ncluster += 1
                        pnew = 1.0 / (1 + ncluster)
                    continue
                cluster_vec[max_idx] = cluster_vec[max_idx] + v
                cluster_idx[max_idx].append(i)
            clusters.append(cluster_idx)
        return clusters


def calculate_topics_similarity(topics):

    topics_txt = [[tup[0] for tup in topic] for topic in topics]
    topics_dict = Dictionary(topics_txt)
    bow_topics = [topics_dict.doc2bow(text) for text in topics_txt]

    id2token = dict((v, k) for k, v in topics_dict.token2id.iteritems())

    new_bow_topics = []
    for i, text in enumerate(bow_topics):
        bow_weights = []
        for word_id, count in text:
            weight_dict = dict(topics[i])
            token = id2token[word_id]
            weight = float(weight_dict[token])
            bow_weights.append((word_id, count*weight))
        new_bow_topics.append(bow_weights)
    print bow_topics
    print new_bow_topics

    tfidf_model = TfidfModel(new_bow_topics, id2word=id2token, dictionary=topics_dict)

    data = corpus2dense(tfidf_model[new_bow_topics], num_terms=len(topics_dict), num_docs=len(bow_topics)).T

    print data.shape

    topic_similarities = cosine_similarity(data)
    #topic_distances = cosine_distances(bow_topics.toarray())
    print topic_similarities

    thresh = 0.15
    n_topics = len(topic_similarities)
    for i in range(n_topics):
        print i,
        for j in range(n_topics):
            if topic_similarities[i][j] > thresh:
                print j,
        print


    #db = DBSCAN(eps=0.3, min_samples=3, metric='precomputed', algorithm='auto').fit(topic_distances)
    #print db.labels_
    #print db.core_sample_indices_


def clean_and_tokenize(dataset, stoplist=None):
    # tokenize the text field in the data:
    # remove the punctuation, stopwords and words of length 1
    # text fields becomes a list of tokens instead of a string
    text_data = [tu.normalize_words(tu.normalize_punctuation(text).split(), stoplist) for text in dataset]
    return text_data


def process_text(dataset, stoplist=None):
    """
    Extracts text data from the dataset
    Cleans and tokenizes text data
    Computes most frequent phrases, creates a dictionary and converts the corpus to a BOW model
    :param dataset:
    :return: processed dataset with phrases, dictionary and BOW corpus
    """

    logging.info("Cleaned and tokenzed dataset")
    text_dataset = clean_and_tokenize(dataset, stoplist=stoplist)

    bi_grams = Phrases(text_dataset, threshold=20)
    #tri_grams = Phrases(bi_grams[text_dataset], threshold=40)
    text_dataset = bi_grams[text_dataset]

    dictionary = Dictionary(text_dataset)
    dictionary.filter_extremes(no_below=1, no_above=0.9)
    bow_corpus = [dictionary.doc2bow(text) for text in text_dataset]

    return text_dataset, dictionary, bow_corpus


def build_lda(text_corpus=None, dictionary=None, n_topics=10, mallet=True, dataname="none"):
    """
    Given a text corpus builds an LDA model (mallet or gensim) and saves it.

    :param text_corpus: text corpus *not* BOW!!
    :param dictionary: dictionary defining tokens to id
    :param n_topics:  number of tokens
    :param mallet: mallet LDA or gensim LDA
    :param dataname: basename of the LDA model
    :return: the name of the LDA model
    """

    if mallet:
        mallet_path = os.environ.get("MALLETPATH")
        lda_model = LdaMallet(mallet_path, corpus=text_corpus, num_topics=n_topics, id2word=dictionary, workers=4,
                              optimize_interval=10, iterations=1000, prefix=os.path.join(os.getcwd(), 'mallet/'))
        lda_model_name = "lda_model_mallet_%s_%i" % (dataname, n_topics)
    else:
        lda_model = LdaModel(text_corpus, id2word=dictionary, num_topics=n_topics, distributed=False,
                             chunksize=2000, passes=5, update_every=10, alpha='asymmetric',
                             eta=0.1, decay=0.5, eval_every=10, iterations=1000, gamma_threshold=0.001)

        lda_model_name = "lda_model_%s_%i" % (dataname, n_topics)

    lda_model.save(lda_model_name)
    return lda_model_name


def load_lda_model(lda_model_name=None, mallet=False):
    if os.path.isfile(lda_model_name):
        if mallet:
            lda_model = LdaMallet.load(lda_model_name)
        else:
            lda_model = LdaModel.load(lda_model_name)
        return lda_model
    return None


def apply_lda(bow_text_data, lda_model=None):
    """
    If the LDA model exists, apply it to the BOW corpus and return topic assignments
    :param bow_text_data: BOW corpus
    :param lda_model: lda_model
    :return: a list of topic assignments
    """

    if lda_model is not None:
        topic_assignments = lda_model[bow_text_data]
        return topic_assignments

    return None


def extract_topic_definitions(lda_model=None, n_topics=10, dataname=""):
    """
    Extract the definition of each topic for the lda_model provided
    :param lda_model: lda model
    :param n_topics: # of topics
    :param dataname: basename
    :return:
    """
    if lda_model is not None:
        # Process topic definitions
        topic_definition = []
        for i, topic in enumerate(lda_model.show_topics(n_topics, num_words=20, formatted=False)):
            topic_list = [word for _, word in topic]
            # The string defining the topic without the u' character
            topic_definition.append("%i, %s" % (i, repr(" ".join(sorted(topic_list)))[2:-1]))

        # Save the topic labels
        with open(dataname+"_labels.txt", 'w') as fout:
            for label in topic_definition:
                fout.write("%s\n" % label)
        # Save topic definigtions with weights
        with codecs.open(dataname+"_labels_weights.txt", 'w', encoding='utf-8') as fout:
            for label in lda_model.show_topics(n_topics, num_words=20, formatted=False):
                for tup in label:
                    fout.write("%s,%s " % (tup[0], tup[1]))
                fout.write("\n")


def date_topic_histogram(date_data=None, topics_data=None, n_topics=10, dataname=""):
    """
    Creates a histogram of counts per topic per date-bin.
    :param date_data:
    :param topics_data:
    :param n_topics:
    :param dataname:
    :return:
    """

    # Calculate time span
    earliest_date = date_data[0]
    latest_date = date_data[-1]

    num_bins = 100
    time_span = latest_date - earliest_date
    time_bin = time_span / num_bins
    logging.info("Time span: %s, time bin: %s " % (time_span, time_bin))

    bin_lows = [earliest_date]
    bin_high = earliest_date + time_bin
    logging.info("Binning %i topics into %i bins " % (n_topics, num_bins))

    # Create the histogram
    counts = np.zeros((n_topics, num_bins+1))
    bin_num = 0
    for topic_assignements, cur_date in zip(topics_data, date_data):
        if cur_date >= bin_high:
            bin_num += 1
            bin_lows.append(bin_high)
            bin_high = bin_lows[len(bin_lows)-1] + time_bin
        for topic_num, weight in topic_assignements:
            counts[topic_num, bin_num] += weight

    # Save the counts
    np.savetxt(dataname+"_cnts.txt", counts)
    # Save the date bins
    with open(dataname+"_bins.txt", 'w') as fout:
        for date in bin_lows:
            fout.write("%s\n" % date)


def bin_tweets_by_date_and_lda(dataset, n_topics=10, mallet=False, dataname="", lda_model_name=None):
    # dataset a class of KenyaCSMessage, a list of tweets, sorted by date.

    # Extract date and text.
    # Clean, tokenize it
    # Build a BOW model.
    date_pos = dataset.date_pos
    text_pos = dataset.text_pos
    data = np.array(dataset.data)
    date_data = data[:, date_pos]

    text_data, text_dict, text_bow = process_text(data[:, text_pos], stoplist=dataset.stoplist)
    logging.info("Text processed")

    # If and LDA model does not already exist - build it.
    if lda_model_name is None or not os.path.isfile(lda_model_name):

        logging.info("Building lda with %i " % int(n_topics))
        lda_model_name = build_lda(text_corpus=text_bow, dictionary=text_dict, n_topics=int(n_topics), mallet=False,
                                   dataname=dataname)
        logging.info("Lda model created in %s " % lda_model_name)

    # Load the LDA model
    lda_model = load_lda_model(lda_model_name, mallet=mallet)

    # Create the histogram of counts per topic per date.
    topic_assignments = apply_lda(bow_text_data=text_bow, lda_model=lda_model)
    date_topic_histogram(date_data, topic_assignments, n_topics=n_topics, dataname=dataname)

    # Extract and process topic definitions
    extract_topic_definitions(lda_model=lda_model, n_topics=n_topics, dataname=dataname)

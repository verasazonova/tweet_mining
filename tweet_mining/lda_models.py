from tweet_mining.utils import textutils as tu

__author__ = 'verasazonova'

import numpy as np
import os
import os.path
from gensim.models import LdaMallet, LdaModel
import codecs
import logging


# **************** LDA relating functions ******************************
def make_lda_model_name(dataname, n_topics=10, mallet=False):
    if mallet:
        return "lda_model_mallet_%s_%i" % (dataname, n_topics)
    else:
        return "lda_model_%s_%i" % (dataname, n_topics)


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
    else:
        lda_model = LdaModel(text_corpus, id2word=dictionary, num_topics=n_topics, distributed=False,
                             chunksize=2000, passes=5, update_every=10, alpha='asymmetric',
                             eta=0.1, decay=0.5, eval_every=10, iterations=1000, gamma_threshold=0.001)

    lda_model_name = make_lda_model_name(dataname, n_topics=n_topics, mallet=mallet)
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
            topic_definition.append("%s" % repr(" ".join(sorted(topic_list)))[2:-1])

        # Save the topic labels
        with open(dataname+"_labels.txt", 'w') as fout:
            for label in topic_definition:
                fout.write("%s\n" % label)

        # Save topic definitions with weights
        with codecs.open(dataname+"_labels_weights.txt", 'w', encoding='utf-8') as fout:
            for label in lda_model.show_topics(n_topics, num_words=20, formatted=False):
                for tup in label:
                    fout.write("%s,%s " % (tup[0], tup[1]))
                fout.write("\n")


# **************** Histogram relating functions ******************************

def bin_tweets_by_date_and_lda(dataset, n_topics=10, mallet=False, dataname=""):
    # dataset a class of KenyaCSMessage, a list of tweets, sorted by date.

    # Extract date and text.
    # Clean, tokenize it
    # Build a BOW model.
    date_pos = dataset.date_pos
    text_pos = dataset.text_pos
    data = np.array(dataset.data)
    date_data = data[:, date_pos]

    lda_model_name = make_lda_model_name(dataname, n_topics=n_topics, mallet=mallet)

    text_data, text_dict, text_bow = tu.process_text(data[:, text_pos], stoplist=dataset.stoplist)
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

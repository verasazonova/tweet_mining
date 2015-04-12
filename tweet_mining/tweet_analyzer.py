__author__ = 'verasazonova'

import argparse
import plot_tweets
import analyze_tweets
from io_tweets import KenyanCSVMessage
import logging


def calculate_and_plot_lda(filename, ntopics, dataname):
    stop_path = "/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/en_swahili.txt"

    dataset = KenyanCSVMessage(filename, fields=["id_str", "text", "created_at"], stop_path=stop_path)
    lda_model_name = "lda_model_%s_%i" % (dataname, int(ntopics))
    analyze_tweets.bin_tweets_by_date_and_lda(dataset, n_topics=ntopics, mallet=False, dataname=dataname,
                                              lda_model_name=lda_model_name)

    counts, bins, labels, topics = plot_tweets.read_counts_bins_labels(dataname)
    plot_tweets.plot_tweets(counts=counts, dates=bins, labels=labels, topics=topics, dataname=dataname)


def test_print_tweets(filename):
    data = KenyanCSVMessage(filename, ["id_str", "text", "created_at"])
    for row in data:
        print row


def __main__():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', help='Filename')
    parser.add_argument('--dname', action='store', dest='dataname', help='Output filename')
    parser.add_argument('--ldamodel', action='store', dest='ldamodelname', default="", help='Lda model filename')
    parser.add_argument('-n', action='store', dest='ntopics', default='10', help='Number of LDA topics')

    arguments = parser.parse_args()

#    test_print_tweets(arguments.filename)
#    clean_save_tweets(arguments.filename)
    #get_statistics(arguments.filename)

    calculate_and_plot_lda(arguments.filename, int(arguments.ntopics), arguments.dataname)


if __name__ == "__main__":
    __main__()
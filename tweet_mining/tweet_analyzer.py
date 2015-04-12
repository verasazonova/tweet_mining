__author__ = 'verasazonova'

import argparse
import plot_tweets
import analyze_tweets
import io_tweets
import logging


def calculate_and_plot_lda(filename, ntopics, dataname):
    stop_path = "/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/en_swahili.txt"

    # Load dataset
    #dataset = io_tweets.KenyanCSVMessage(filename, fields=["id_str", "text", "created_at"], stop_path=stop_path)

    # Create the histogram of LDA topics by date
    #analyze_tweets.bin_tweets_by_date_and_lda(dataset, n_topics=ntopics, mallet=False, dataname=dataname)

    # Read the resulting counts, date bins, and topics
    counts, bins, topics = io_tweets.read_counts_bins_labels(dataname)

    # Figure out which topics to cluster together
    clustered_counts, clustered_labels, clusters = analyze_tweets.build_clusters(counts, topics)

    # Plot the clustered histogram
    plot_tweets.plot_tweets(counts=clustered_counts, dates=bins, clusters=clusters,
                            labels=clustered_labels, dataname=dataname)


def calculate_and_test_w2v(filename, size, window, dataname):
    stop_path = "/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/en_swahili.txt"

    # Load dataset
    dataset = io_tweets.KenyanCSVMessage(filename, fields=["id_str", "text", "created_at"])

    # Create the histogram of LDA topics by date
    word_list = ["attack", "president", "peace", "hope", "violence", "#alshabaab", "government"]
    analyze_tweets.build_test_w2v(dataset,word_list=word_list, dataname=dataname, size=size, window=window)


def test_print_tweets(filename):
    data = io_tweets.KenyanCSVMessage(filename, ["id_str", "text", "created_at"])
    for row in data:
        print row


def __main__():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', help='Filename')
    parser.add_argument('--dname', action='store', dest='dataname', help='Output filename')
    parser.add_argument('--ldamodel', action='store', dest='ldamodelname', default="", help='Lda model filename')
    parser.add_argument('-n', action='store', dest='ntopics', default='10', help='Number of LDA topics')

    arguments = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename=arguments.dataname+"_log.txt")

#    test_print_tweets(arguments.filename)
#    clean_save_tweets(arguments.filename)
    #get_statistics(arguments.filename)

#    calculate_and_plot_lda(arguments.filename, int(arguments.ntopics), arguments.dataname)
    calculate_and_test_w2v(arguments.filename, 50, 8, arguments.dataname)


if __name__ == "__main__":
    __main__()
from tweet_mining.utils import textutils as tu

__author__ = 'verasazonova'

import csv
import logging
import os
import codecs
import sys
from operator import itemgetter
from os.path import basename
import dateutil.parser
import numpy as np


def read_counts_bins_labels(dataname):
    counts = np.loadtxt(dataname+"_cnts.txt")
    bin_lows = []
    with open(dataname+"_bins.txt", 'r') as f:
        for line in f:
            bin_lows.append(dateutil.parser.parse(line.strip()))
    topic_definitions = []
    with open(dataname+"_labels.txt", 'r') as f:
        for line in f:
            topic_definitions.append(line.strip().split())
    topics = []
    with codecs.open(dataname+"_labels_weights.txt", 'r', encoding='utf-8') as f:
        for line in f:
            topics.append([(tup.split(',')[1], tup.split(',')[0]) for tup in line.strip().split(' ')])

    return counts, bin_lows, topics


def read_tweets(filename, fields):
    """
    Read the raw csv file returns a list of tweets with certain fields

    :param filename: csv filename with a header
    :param fields: name of fields to retain
    :return: a double list of tweet data (unicode)
    """

    def fix_corrupted_csv_line(line):
        reader_str = unicode_csv_reader([line], dialect=csv.excel)
        row_str = reader_str.next()
        return row_str

    with codecs.open(filename, 'r', encoding='utf-8', errors='replace') as f:
        # forcing the reading line by line to avoid malformed csv entries
        reader = unicode_csv_reader(f, dialect=csv.excel)
        header = reader.next()
        # a list of indexes of fields
        field_positions = [header.index(unicode(field)) for field in fields]

        logging.info("Saving the following fields: %s " % zip(fields, field_positions))
        data = []
        try:
            for row in reader:
                if len(row) == len(header):
                    data.append([row[pos] for pos in field_positions])
                else:
                    row_str = ", ".join(row).split('\r\n')
                    data.append([fix_corrupted_csv_line(row_str[0])[pos] for pos in field_positions])
                    if len(row_str) > 1:
                        data.append([fix_corrupted_csv_line(row_str[1])[pos] for pos in field_positions])

            #data = [[row[pos] for pos in field_positions] for row in reader]
        except csv.Error as e:
            sys.exit('file %s: %s' % (filename, e))

    logging.info("Data read: %s" % len(data))
    logging.info("First line: %s" % data[0])
    logging.info("Last line: %s" % data[-1])
    return data


def clean_tweets(data, fields):
    """
    Normalzes the text of the tweets to remove newlines.
    Sorts the tweets by the date

    :param data: a double list of tweets
    :param fields: a list of field names corresponding to the data
    :return: normalized and sorted tweet list
    """

    # remove newline from text
    # make date a date
    text_str = "text"
    text_pos = None
    if text_str in fields:
        text_pos = fields.index(text_str)
    date_str = "created_at"
    date_pos = None
    if date_str in fields:
        date_pos = fields.index(date_str)
    id_str = "id_str"
    id_pos = None
    if id_str in fields:
        id_pos = fields.index(id_str)
    label_str = "label"
    label_pos = None
    if label_str in fields:
        label_pos = fields.index(label_str)

    for cnt in range(len(data)):
        if id_pos is not None:
            data[cnt][id_pos] = data[cnt][id_pos].strip()
        if text_pos is not None:
            data[cnt][text_pos] = tu.normalize_format(data[cnt][text_pos])
        if date_pos is not None:
            data[cnt][date_pos] = dateutil.parser.parse(data[cnt][date_pos])

    #sort tweets by date
    if date_pos is not None:
        data_sorted = sorted(data, key=itemgetter(date_pos))
        logging.info("Data sorted by date.  Span: %s - %s" % (data_sorted[0][date_pos], data_sorted[-1][date_pos]))
    else:
        data_sorted = data

    logging.info("Data pre-processed")
    logging.info("First line: %s" % data_sorted[0])
    logging.info("Last line: %s" % data_sorted[-1])

    return data_sorted, date_pos, text_pos, id_pos, label_pos


def clean_save_tweet_text(filename, fields):

    data, date_pos, text_pos, id_pos, label_pos = clean_tweets(read_tweets(filename, fields), fields)
    print data[0]
    print text_pos
    if "text" in fields:
        processed_data = [tu.normalize_punctuation(text[text_pos]) for text in data]

    filename_out = basename(filename).split('.')[0] + "_short_"
    if "created_at" in fields:
        filename_out += "sorted_"
    filename_out += "_".join(fields) + ".csv"

    #save sorted file with newline charactes removed from the text
    with codecs.open(filename_out, 'w', encoding='utf-8') as fout:
        # write the data
        fout.write("\",\"".join(fields)+"\n")
        for row in data:
            fout.write(" ".join(row)+"\n")


def save_liblinear_format_data(filename, x_data, y_data):

    with open(filename, 'w') as fout:
        for x, y in zip(x_data, y_data):
            fout.write("%i " % int((y-0.5)*2) )
            for i, coordinate in enumerate(x):
                fout.write("%i:%f " % (i+1, coordinate))
            fout.write("\n")


class KenyanCSVMessage():

    def __init__(self, filename, fields=None, stop_path="", start_date=None, end_date=None):
        """
        A class that reads and encapsulates csv twitter (or facebook) data.

        :param filename: a csv data filename.  A first row is a header explaining the fields
        :param fields: a list of field_names to include in the data
        :param stop_path: a path to a file containing the stopword list
        :return: a list of tweets sorted by date if such field exists.  newlines removed from text field (if exists)
        """
        self.filename = filename
        self.fields = fields

        if os.path.isfile(stop_path):
            logging.info("Using %s as stopword list" % stop_path)
            self.stoplist = [unicode(word.strip()) for word in
                             codecs.open(stop_path, 'r', encoding='utf-8').readlines()]
        else:
            self.stoplist = []

        self.data, self.date_pos, self.text_pos, self.id_pos, self.label_pos = \
            clean_tweets(read_tweets(self.filename, self.fields), self.fields)
        self.data = np.array(self.data)
        if start_date is None:
            self.start_date = self.data[0][self.date_pos]
        else:
            self.start_date = start_date
        if end_date is None:
            self.end_date = self.data[-1][self.date_pos]
        else:
            self.end_date = end_date

    def __iter__(self):

        for row in self.data:
            if self.date_pos is None:
                yield row
            elif row[self.date_pos] >= self.start_date and row[self.date_pos] <= self.end_date:
                yield row


def is_politica_tweet(tweet_text, political_words):

    for word in tweet_text:
        if word in political_words:
            return True

    return False


def extract_political_tweets(filename):
    fields = ["id_str", "text", "created_at"]
    data = KenyanCSVMessage(filename, fields)
    political_path = "/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/political.txt"

    if os.path.isfile(political_path):
            logging.info("Using %s as stopword list" % political_path)
            political_list = [unicode(word.strip()) for word in
                              codecs.open(political_path, 'r', encoding='utf-8').readlines()]
    else:
        political_list = []

    with codecs.open(basename(filename).split('.')[0]+"_political.csv", 'w', encoding='utf-8') as fout:
        fout.write("\",\"".join(fields)+"\n")
        for row in data:
            if is_politica_tweet(row[data.text_pos], political_list):
                # write the data
                fout.write("\",\"".join(row)+"\n")


def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]


def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8', errors='replace')


def make_positive_labeled_kenyan_data(dataname):
    dataset_positive = KenyanCSVMessage(dataname+"_positive.csv", fields=["id_str"])
    data_positive = [tweet[dataset_positive.text_pos] for tweet in dataset_positive]

    print len(data_positive)
    dataset = KenyanCSVMessage(dataname+".csv", fields=["text", "id_str"])

    cnt_pos = 0
    with codecs.open(dataname+"_annotated_positive.csv", 'w', encoding='utf-8') as fout:
        fout.write("id_str,text,label\n")
        for cnt, tweet in enumerate(dataset):
            if cnt % 10000 == 0:
                print cnt, cnt_pos
            if tweet[dataset.id_pos] in data_positive:
                cnt_pos += 1
                fout.write(tweet[dataset.id_pos] + ",\"" + tweet[dataset.text_pos].replace("\"", "\"\"") + "\",T\n")
            else:
                fout.write(tweet[dataset.id_pos] + ",\"" + tweet[dataset.text_pos].replace("\"", "\"\"") + "\",F\n")

    print cnt_pos


def save_positives(positives, dataname):
    with codecs.open(dataname+"_additional_positives.csv", 'w', encoding='utf-8') as fout:
        fout.write("id_str,text\n")
        for tweet, id in positives:
            fout.write(id + ",\"" + tweet.replace("\"", "\"\"") + "\"\n")

def save_words_representations(filename, word_list, vec_list):
    # saving word representations
    # word, w2v vector
    with codecs.open(filename, 'w', encoding="utf-8") as fout:
        for word, vec in zip(word_list, vec_list):
            fout.write(word + "," + ",".join(["%.8f" % x for x in vec]) + "\n")


# save cluster information: size and central words
def save_cluster_info(filename, cluster_info):
    with codecs.open(filename, 'w', encoding="utf-8") as fout:
        for cluster_dict in cluster_info:
            fout.write("%2i, %5i,   : " % (cluster_dict['cnt'], cluster_dict['size']))
            for j, word in enumerate(cluster_dict['words']):
                if j < 10:
                    fout.write("%s " % word)
            fout.write("\n")

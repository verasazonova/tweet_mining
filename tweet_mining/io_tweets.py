__author__ = 'verasazonova'

import csv
import logging
import os
import codecs
import sys
from operator import itemgetter
from os.path import basename
import textutils as tu
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

    with codecs.open(filename, 'r', encoding='utf-8') as f:
        reader = unicode_csv_reader(f, dialect=csv.excel)
        header = reader.next()
        # a list of indexes of fields
        field_positions = [header.index(unicode(field)) for field in fields]

        logging.info("Saving the following fields: %s " % zip(fields, field_positions))

        try:
            data = [[row[pos] for pos in field_positions] for row in reader]
        except csv.Error as e:
            sys.exit('file %s: %s' % (filename, e))

    logging.info("Data read")
    logging.info("First line: %s" % data[0])
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

    for cnt in range(len(data)):
        if text_pos is not None:
            data[cnt][text_pos] = tu.normalize_format(data[cnt][text_pos])
        if date_pos is not None:
            data[cnt][date_pos] = dateutil.parser.parse(data[cnt][date_pos])

    #sort tweets by date
    if date_pos is not None:
        data_sorted = sorted(data, key=itemgetter(date_pos))
    else:
        data_sorted = data

    logging.info("Data sorted by date.  Span: %s - %s" % (data_sorted[0][date_pos], data_sorted[-1][date_pos]))

    return data_sorted, date_pos, text_pos


def clean_save_tweets(filename, fields):

    data = clean_tweets(read_tweets(filename, fields), fields)

    #save sorted file with newline charactes removed from the text
    with codecs.open(basename(filename).split('.')[0]+"_sorted_short.csv", 'w', encoding='utf-8') as fout:
        # write the data
        fout.write("\",\"".join(fields)+"\n")
        for row in data:
            fout.write("\",\"".join(row)+"\n")


class KenyanCSVMessage():

    def __init__(self, filename, fields=None, stop_path=""):
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

        self.data, self.date_pos, self.text_pos = clean_tweets(read_tweets(self.filename, self.fields), self.fields)

    def __iter__(self):

        for row in self.data:
            yield row


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
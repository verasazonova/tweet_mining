__author__ = 'verasazonova'

import re
from gensim.corpora import Dictionary
from gensim.models import Phrases
import logging


# add space around punctuation
# assumes unicode strings
def normalize_punctuation(phrase, url=True, username=True, hashtag=True, punctuation=True, RT=False):
    norm_phrase = phrase.lower().strip()
    #delete url
    if url:
        norm_phrase = re.sub(r'http(\S*)\b', "URL", norm_phrase)

    #delete usernames
    if username:
        norm_phrase = re.sub(r'@(\w*)\b', "USER", norm_phrase)

    #delete RT
    if RT:
        norm_phrase = re.sub(r'\bRT\b', '', norm_phrase)

    #delete #
    if hashtag:
        norm_phrase = re.sub(r'#', '', norm_phrase)

    if punctuation:
        for punctuation_char in [',', ':', '.', '(', ')', '!', '?', ':', ';', '/', '\"', '*', '^', '\'',
                                 u"\u2026", u"\u2013", '_', '[', ']', '+', '-']:
            norm_phrase = norm_phrase.replace(punctuation_char, ' ' + punctuation_char+' ')

    return norm_phrase


#remove one letter words
def normalize_words(words_list, stoplist, keep_all=False):
    if keep_all:
        norm_list = [word for word in words_list]
    else:
        norm_list = [word for word in words_list if len(word) > 1 and word not in stoplist+[u'rt']]
    return norm_list


def normalize_format(phrase):
    # remove carriage return
    norm_phrase = phrase.replace('\r', '').replace('\n', ' ')
    return norm_phrase


def clean_and_tokenize(corpus, stoplist=None, keep_all=False):
    # tokenize the text field in the data:
    # remove the punctuation, stopwords and words of length 1
    # text fields becomes a list of tokens instead of a string
    text_data = [normalize_words(normalize_punctuation(text).split(), stoplist, keep_all=keep_all) for text in corpus]
    return text_data


def process_text(corpus, stoplist=None, bigrams=None, trigrams=None, keep_all=False, no_below=10, no_above=0.8):
    """
    Extracts text data from the corpus
    Cleans and tokenizes text data
    Computes most frequent phrases, creates a dictionary and converts the corpus to a BOW model
    :param corpus:
    :return: processed corpus with phrases, dictionary and BOW corpus
    """

    logging.info("Cleaned and tokenzed dataset")
    text_dataset = clean_and_tokenize(corpus, stoplist=stoplist, keep_all=keep_all)

    if bigrams is not None:
        bi_grams = Phrases(text_dataset, threshold=bigrams, min_count=no_below)
        text_dataset = bi_grams[text_dataset]
    elif trigrams is not None:
        bi_grams = Phrases(text_dataset, threshold=bigrams)
        tri_grams = Phrases(bi_grams[text_dataset], threshold=trigrams)
        text_dataset = tri_grams[bi_grams[text_dataset]]

    dictionary = Dictionary(text_dataset)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    bow_corpus = [dictionary.doc2bow(text) for text in text_dataset]

    return text_dataset, dictionary, bow_corpus

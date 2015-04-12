__author__ = 'verasazonova'

import re


# add space around punctuation
# assumes unicode strings
def normalize_punctuation(phrase):
    norm_phrase = phrase.lower().strip()
    #delete url
    norm_phrase = re.sub(r'http(\S*)\B', "", norm_phrase)

    #delete known unreadable characters
    norm_phrase = re.sub(u"\uFFFD", "", norm_phrase)
    norm_phrase = re.sub(r'&gt', "", norm_phrase)
    norm_phrase = re.sub(r'&lt', "", norm_phrase)
    norm_phrase = re.sub(r'&amp', "", norm_phrase)

    for punctuation in [',', ':', '.', '(', ')', '!', '?', ':', ';', '/', '\"', '*', '^', '\'']:
        norm_phrase = norm_phrase.replace(punctuation, ' ' + punctuation+' ')
    return norm_phrase


#remove one letter words
def normalize_words(words_list, stoplist):
    norm_list = [word for word in words_list if len(word) > 1 and word not in stoplist+[u'rt']]
    return norm_list


def normalize_format(phrase):
    # remove carriage return
    norm_phrase = phrase.replace('\r', '').replace('\n', ' ')
    return norm_phrase


def clean_and_tokenize(dataset, stoplist=None):
    # tokenize the text field in the data:
    # remove the punctuation, stopwords and words of length 1
    # text fields becomes a list of tokens instead of a string
    text_data = [normalize_words(normalize_punctuation(text).split(), stoplist) for text in dataset]
    return text_data

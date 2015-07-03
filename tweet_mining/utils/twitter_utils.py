__author__ = 'verasazonova'

import twitter
import twitter.oauth
import time
import argparse
import datetime
import json
import codecs

CONSUMER_KEY = "PcAXASHnVgh3ee3JiZeBBByJ7"
CONSUMER_SECRET = "Rbl49F5C4Fbbwt7maZBw3IB325Q1XMa8xRQ4I54UerJXxFKQSS"

OAUTH_TOKEN = "277586033-34CoMaWtywqgC37tc6CgDKJhOHDZUfTt1kwHKJay"
OAUTH_TOKEN_SECRET = "6cvYDqYFAlCsw3WETSZ6MFb5C3GB4qe6BVT78Ua33fNPq"


def oauth_login():
    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
    return twitter.Twitter(auth=auth)


def save_json(filename, data):
    with codecs.open(filename, 'w', encoding='utf-8') as f:
        f.write(unicode(json.dumps(data, ensure_ascii=False)))


def load_json(filename):
    with codecs.open(filename, encoding='utf-8') as f:
        return f.read()


fields_str = "text,retweet_count,favorited,truncated,id_str,in_reply_to_screen_name,source,retweeted,created_at," \
             "in_reply_to_status_id_str,in_reply_to_user_id_str,lang,listed_count,verified,location,user_id_str," \
             "description,geo_enabled,user_created_at,statuses_count,followers_count,favourites_count,protected," \
             "user_url,name,time_zone,user_lang,utc_offset,friends_count,screen_name,country_code,country," \
             "place_type,full_name,place_name,place_id,place_lat,place_lon,lat,lon,expanded_url,url"
fields = fields_str.split(',')


def save_tweet(tweet, filename):
    # save the fields in the tweet defined by the fields list.  Append to a given file
    with codecs.open(filename, 'a', encoding='utf-8') as fout:
        for field in fields:
            if tweet in fields:
                fout.write(",%s" % tweet[field])
            else:
                fout.write(",NA")
        fout.write("\n")


def search_tweets(q, max_id=None, since_id=None):
    """

    Searches for older tweets defined by a search parameter q, max_id, since_id

    :param q: Search parameter
    :param max_id: the latest (max) id_str of the tweet
    :param since_id: the earliest id_str of the tweet
    :return: the oldest id_str found within the 100x10 tweets (100 tweets per page, 10 pages)
    """
    twitter_api = oauth_login()

    count = 100
    # set up the query for a search, depending whether max_id and or since_id are provided
    if max_id is not None:
        if since_id is not None:
            if since_id >= max_id:
                print max_id, since_id, "Since id later than max id"
                return None
            search_results = twitter_api.search.tweets(q=q, count=count, result_type='recent', max_id=max_id,
                                                       since_id=since_id)
        else:
            search_results = twitter_api.search.tweets(q=q, count=count, result_type='recent', max_id=max_id)
    else:
        if since_id is not None:
            search_results = twitter_api.search.tweets(q=q, count=count, result_type='recent', since_id=since_id)
        else:
            search_results = twitter_api.search.tweets(q=q, count=count, result_type='recent')

    print "Searching for %s with max_id of %s" % (q, max_id)

    try:
        statuses = search_results['statuses']

        # Iterate through 100 more batches of results by following the cursor
        for _ in range(100):
            print "Length of statuses", len(statuses)
            try:
                next_results = search_results['search_metadata']['next_results']
            except KeyError, e:  # No more results when next_results doesn't exist
                print "No more tweets %s" % e
                break

            # Create a dictionary from next_results, which has the following form:
            # ?max_id=313519052523986943&q=NCAA&include_entities=1
            kwargs = dict([kv.split('=') for kv in next_results[1:].split("&")])

            search_results = twitter_api.search.tweets(**kwargs)
            statuses += search_results['statuses']

        # search will be saved into a file named by query phrase, max_id and since_id
        filename = "%s_search_%s_%s.txt" % (q, max_id, since_id)
        print "Saving to %s" % filename

        cnt = 0
        oldest_id = ""
        # save tweets and find the oldest tweet id.
        for status in statuses:
            save_tweet(status, filename)
            if cnt == 0:
                oldest_id = status['id_str']
            elif status['id_str'] < oldest_id:
                oldest_id = status['id_str']
            cnt += 1

        return oldest_id

    except KeyError, e:
        print "No tweets found %s" % e
        return None


def connect_stream(q):
    """
    Connects to a twitter stream defined by a query phrase and saves it to a file

    :param q: search phrase
    :return: nothing
    """

    # Connect to a stream defined by the query phrase 1
    twitter_api = oauth_login()
    twitter_stream = twitter.TwitterStream(auth=twitter_api.auth)
    stream = twitter_stream.statuses.filter(track=q)

    filename = "%s_stream.txt" % q

    cnt = 0
    print datetime.datetime.now()

    for tweet in stream:
        save_tweet(tweet, filename)
        print cnt, datetime.datetime.now()
        cnt += 1


def __main__():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--maxid', action='store', dest='maxid', help='Filename')
    parser.add_argument('--sinceid', action='store', dest='sinceid', help='Filename')
    parser.add_argument('--stream', action='store_true', dest='stream', help='Stream Search')
    parser.add_argument('-q', action='store', dest='q', help='Stream Search')
    arguments = parser.parse_args()

    q = arguments.q

    if arguments.stream:
        connect_stream(q)
    else:
        maxid = arguments.maxid
        sinceid = arguments.sinceid
        maxid = search_tweets(q, maxid, sinceid)
        while maxid is not None:
            if maxid is not None:
                print maxid, sinceid
                print datetime.datetime.now()
                for i in range(3):
                    print "%i min of sleep rest" % ((3-i)*5)
                    time.sleep(5*60)
            maxid = search_tweets(q, maxid, sinceid)

if __name__ == "__main__":
    __main__()

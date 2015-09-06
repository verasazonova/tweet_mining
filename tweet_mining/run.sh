#!/usr/bin/env bash
# This is a test running file for tweet_mining package
#

echo $1

MAIN_PY="/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/tweet_analyzer.py"
DATA_DIR="/Users/verasazonova/Work/HateSpeech/"$1
DATA="../"$1"_annotated_positive.csv"

DATE=`date +%Y-%m-%d-%H-%M`
mkdir $DATA_DIR/$DATE
pushd $DATA_DIR/$DATE

# check model for consistency in finding similar words
# the test file is hardcoded
DATE=`date +%Y-%m-%d-%H-%M`
DNAME=$1"_"$DATE
OUTPUT=$DNAME".txt"

touch $OUTPUT
python $MAIN_PY -f $DATA --dname $DNAME  --size 100  --min 5  --window 10 --nclusters 30 --clfname w2v  --clfbase lr  --action classify --rebuild >> $OUTPUT
#python $MAIN_PY -f $DATA --dname $DNAME  --size 100  --min 5  --window 10 --nclusters 30 --clfname bow  --clfbase lr  --action classify >> $OUTPUT
python $MAIN_PY --dname $DNAME --action plot >> $OUTPUT

popd
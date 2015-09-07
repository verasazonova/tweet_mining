#!/usr/bin/env bash
# This is a test running file for tweet_mining package
#

echo $1
MAIN_PY="/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/tweet_analyzer.py"

if [$1 == "train.csv"]; then
    # For sentiment dataset
    DATA_DIR="/Users/verasazonova/Work/TweetSentiment"
    DATA="/Users/verasazonova/Work/TweetSentiment/trainingandtestdata/train.csv"

    DATE=`date +%Y-%m-%d-%H-%M`
    mkdir $DATA_DIR/$DATE
    pushd $DATA_DIR/$DATE

    # check model for consistency in finding similar words
    # the test file is hardcoded
    DATE=`date +%Y-%m-%d-%H-%M`
    DNAME="sentiment"
    OUTPUT=$DNAME".txt"

    touch $OUTPUT

    PS="0.001 0.01 0.1"
    THRESHS="0.0 0.1 0.4 0.8"
    NS="0 1 2 3 4"
    CLFS="w2v"

    for P in $PS; do
        for THRESH in $THRESHS; do
            for N in $NS; do
                for CLF in $CLFS; do
                    NAME=$DNAME_${P/0./}_${THRESH/0./}".txt"
                    python $MAIN_PY -f train.csv --dname $DNAME --size 100 --window 10 --min 1 --nclusters 30 --clusthresh 0  --p $P --thresh $THRESH --ntrial $N --clfname $CLF --action classify --rebuild >> $NAME
                done
            done
        done
    done



    python $MAIN_PY -f $DATA --dname $DNAME  --size 100  --min 5  --window 10 --nclusters 30 --clusthresh 800 --clfname w2v  --clfbase lr  --action classify --rebuild

else

    DATA_DIR="/Users/verasazonova/Work/HateSpeech/"$1
    DATA="../"$1"_annotated_positive.csv"

    DATE=`date +%Y-%m-%d-%H-%M`
    mkdir $DATA_DIR/$DATE
    pushd $DATA_DIR/$DATE

    # check model for consistency in finding similar words
    # the test file is hardcoded
    DATE=`date +%Y-%m-%d-%H-%M`
    DNAME=$1
    OUTPUT=$DNAME".txt"

    touch $OUTPUT
    python $MAIN_PY -f $DATA --dname $DNAME  --size 100  --min 5  --window 10 --nclusters 30 --clusthresh 800 --clfname w2v  --clfbase lr  --action classify --rebuild
    #python $MAIN_PY -f $DATA --dname $DNAME  --size 100  --min 5  --window 10 --nclusters 30 --clfname bow  --clfbase lr  --action classify >> $OUTPUT
    python $MAIN_PY --dname $DNAME --action plot >> $OUTPUT

    popd

fi
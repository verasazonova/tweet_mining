#!/usr/bin/env bash
# This is a test running file for tweet_mining package
#

echo $1
MAIN_PY="/Users/verasazonova/Work/PycharmProjects/tweet_mining/tweet_mining/tweet_analyzer.py"

if [ "$1" == "sentiment" ]; then
    # For sentiment dataset
    DATA_DIR="/Users/verasazonova/Work/TweetSentiments"
    DATA=$DATA_DIR"/trainingandtestdata/train.csv"
    TEST_DATA=$DATA_DIR"/trainingandtestdata/test.csv"

    DATE=`date +%Y-%m-%d-%H-%M`

    if [ "$#" -le 1 ]; then

        CUR_DIR=$DATA_DIR/$DATE
        mkdir $CUR_DIR

    else
        CUR_DIR=$DATA_DIR/$2

    fi

    pushd $CUR_DIR

    # check model for consistency in finding similar words
    # the test file is hardcoded
    DNAME_BASE="sentiment"
    OUTPUT=$DNAME".txt"

    touch $OUTPUT

    PS="0.1"
    THRESHS="0"
    NS="0"
    CLFS="bow"

    for P in $PS; do
        DNAME=$DNAME_BASE #"_"${P/0./}
        for THRESH in $THRESHS; do
            for N in $NS; do
                for CLF in $CLFS; do
                    NAME=$DNAME_${P/0./}_${THRESH/0./}".txt"
                    COMMAND="$MAIN_PY -f $DATA --dname $DNAME --test $TEST_DATA --size 300 --window 10 --min 1 --nclusters 30 --clusthresh 0  --p $P --thresh $THRESH --ntrial $N --clfname $CLF --clfbase lr --action classify"
                    echo $COMMAND
                    python $COMMAND
                done
            done
        done
    done

#    python $MAIN_PY --dname $DNAME --action plot

    popd

else

    DATA_DIR="/Users/verasazonova/Work/HateSpeech/"$1
    DATA="../"$1"_annotated_positive.csv"

    if [ "$#" -ge 2 ]; then

        DATA2="../../"$2"/"$2".csv"

    else
        DATA2=""

    fi

    echo $DATA2

    DATE=`date +%Y-%m-%d-%H-%M`

    if [ "$#" -le 1 ]; then

        CUR_DIR=$DATA_DIR/$DATE
        mkdir $CUR_DIR

    else
        CUR_DIR=$DATA_DIR/$2

    fi

    pushd $CUR_DIR

    # check model for consistency in finding similar words
    # the test file is hardcoded
    DNAME=$1
    OUTPUT=$DNAME".txt"

    NS="0 1 2 3 4"
    SIZES="100"

    touch $OUTPUT
    for SIZE in $SIZES; do
        for N in $NS; do
            python $MAIN_PY -f $DATA $DATA2 --dname $DNAME  --size $SIZE  --min 1  --window 10  --ntrial $N --clfname w2v  --clfbase lr  --action classify --p 1 --thresh 0
            rm *.npy
            rm w2v_model_*
        done
    done
    python $MAIN_PY -f $DATA $DATA2 --dname $DNAME  --size $SIZE  --min 1  --window 10 --ntrial $N --clfname bow  --clfbase lr  --action classify --p 1 --thresh 0
    #python $MAIN_PY -f $DATA --dname $DNAME  --size 100  --min 5  --window 10 --nclusters 30 --clfname bow  --clfbase lr  --action classify >> $OUTPUT
    #python $MAIN_PY --dname $DNAME --action plot >> $OUTPUT

    popd

fi
#!/usr/bin/env bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Copyright 2016  Radboud University (Author: Emre Yilmaz)

# Apache 2.0

corpus=$1
set -e -o pipefail
if [ -z "$corpus" ] ; then
    echo >&2 "The script $0 expects one parameter -- the location of the council speech corpus"
    exit 1
fi
if [ ! -d "$corpus" ] ; then
    echo >&2 "The directory $corpus does not exist"
fi

echo "Preparing train, development and test data"
mkdir -p data data/local data/train data/dev data/test



for x in train cgn_train dev test dev_nl dev_fr dev_mx test_nl test_fr test_mx fame_dev_nl fame_dev_fr fame_dev_mx fame_test_nl fame_test_fr fame_test_mx; do
    echo "Copy spk2utt, utt2spk, wav.scp, text for $x"
    cp $corpus/data/$x/text     data/$x/text    || exit 1;
    cp $corpus/data/$x/utt2spk  data/$x/utt2spk || exit 1;
    cp $corpus/data/$x/segments  data/$x/segments || exit 1;
    cp $corpus/data/$x/wav.scp  data/$x/wav.scp || exit 1;


    # fix_data_dir.sh fixes common mistakes (unsorted entries in wav.scp,
    # duplicate entries and so on). Also, it regenerates the spk2utt from
    # utt2sp
    utils/fix_data_dir.sh data/$x 
	utils/validate_data_dir.sh data/$x --no-feats
    printf "--- done with $x\n\n"
done

gzip -c $corpus/FAME_council_mix_50.lm > data/local/LM.gz

echo "Data preparation completed."

